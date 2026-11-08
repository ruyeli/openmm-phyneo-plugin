#include "Engine.h"
#include "DispersionMath.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>
#include <algorithm>
namespace NativeDispersion {
static void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
static void fftcheck(cufftResult e) { if(e!=CUFFT_SUCCESS) throw std::runtime_error("cuFFT error "+std::to_string(e)); }
__global__ void realKernel(int n,const double* x,const double* q,double lx,double ly,double lz,double cutoff,double alpha,double* f,double* energy) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double box[3]={lx,ly,lz},fi[3]={0,0,0},ei=0;
    for(int j=0;j<n;j++) if(j!=i) {
        double dr[3],r2=0;
        for(int k=0;k<3;k++) { dr[k]=x[3*i+k]-x[3*j+k]; dr[k]-=nearbyint(dr[k]/box[k])*box[k]; r2+=dr[k]*dr[k]; }
        if(r2>=cutoff*cutoff)continue;
        if(r2<1e-20) { ei=NAN; continue; }
        double ff=0;
        for(int c=0;c<3;c++) { double e,v; realCorrection(r2,alpha,c,e,v); double qq=q[3*i+c]*q[3*j+c]; ei+=.5*qq*e; ff+=qq*v; }
        for(int k=0;k<3;k++) fi[k]+=ff*dr[k];
    }
    for(int c=0;c<3;c++) ei+=q[3*i+c]*q[3*i+c]*selfFactor(alpha,c);
    energy[i]=ei;
    for(int k=0;k<3;k++)f[3*i+k]=fi[k];
}
__global__ void spreadGather(int n,int nx,int ny,int nz,const double* x,const double* q,
                             double lx,double ly,double lz,int component,cufftDoubleComplex* mesh,double* f,bool gather) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    int grid[3]={nx,ny,nz},ref[3]; double box[3]={lx,ly,lz},w[3][6],dw[3][6],ff[3]={0,0,0};
    for(int k=0;k<3;k++) {
        double u=(x[3*i+k]/box[k]-floor(x[3*i+k]/box[k]))*grid[k]; ref[k]=int(ceil(u));
        for(int s=0;s<6;s++) { w[k][s]=spline(ref[k]-u+s); dw[k][s]=dspline(ref[k]-u+s); }
    }
    for(int a=0;a<6;a++) for(int b=0;b<6;b++) for(int c=0;c<6;c++) {
        size_t idx=(size_t(wrap(ref[0]+a-3,nx))*ny+wrap(ref[1]+b-3,ny))*nz+wrap(ref[2]+c-3,nz);
        if(!gather) atomicAdd(&mesh[idx].x,q[3*i+component]*w[0][a]*w[1][b]*w[2][c]);
        else {
            double v=-2*q[3*i+component]*mesh[idx].x;
            ff[0]+=v*dw[0][a]*w[1][b]*w[2][c]*nx/lx;
            ff[1]+=v*w[0][a]*dw[1][b]*w[2][c]*ny/ly;
            ff[2]+=v*w[0][a]*w[1][b]*dw[2][c]*nz/lz;
        }
    }
    if(gather)for(int k=0;k<3;k++)f[3*i+k]+=ff[k];
}
__global__ void reciprocalKernel(size_t count,int nx,int ny,int nz,double lx,double ly,double lz,double alpha,int component,cufftDoubleComplex* mesh,double* energy) {
    __shared__ double sum[256];
    size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    double e=0;
    if(idx<count) {
        int a=idx/(ny*nz),b=(idx/nz)%ny,c=idx%nz;
        double g=green(a,b,c,nx,ny,nz,lx,ly,lz,alpha,component);
        e=-g*(mesh[idx].x*mesh[idx].x+mesh[idx].y*mesh[idx].y);
        mesh[idx].x*=g;mesh[idx].y*=g;
    }
    sum[threadIdx.x]=e; __syncthreads();
    for(int s=128;s>0;s/=2) { if(threadIdx.x<s)sum[threadIdx.x]+=sum[threadIdx.x+s]; __syncthreads(); }
    if(threadIdx.x==0)energy[blockIdx.x]=sum[0];
}
class CudaEngine : public Engine {
    Parameters p;
    size_t count;
    double *x=nullptr,*q=nullptr,*f=nullptr,*e=nullptr;
    cufftDoubleComplex* mesh=nullptr;
    cufftHandle plan=0;
public:
    CudaEngine(const Parameters& parameters):p(parameters),count(size_t(p.grid[0])*p.grid[1]*p.grid[2]) {
        try {
            check(cudaMalloc(&x,3*p.n*sizeof(double))); check(cudaMalloc(&q,3*p.n*sizeof(double)));
            check(cudaMalloc(&f,3*p.n*sizeof(double)));
            check(cudaMalloc(&e,std::max(size_t(p.n),(count+255)/256)*sizeof(double)));
            check(cudaMalloc(&mesh,count*sizeof(cufftDoubleComplex)));
            check(cudaMemcpy(q,p.q.data(),3*p.n*sizeof(double),cudaMemcpyHostToDevice));
            fftcheck(cufftPlan3d(&plan,p.grid[0],p.grid[1],p.grid[2],CUFFT_Z2Z));
        } catch(...) { release(); throw; }
    }
    void release() { if(plan)cufftDestroy(plan);cudaFree(x);cudaFree(q);cudaFree(f);cudaFree(e);cudaFree(mesh); }
    ~CudaEngine() { release(); }
    double evaluate(const double* positions,const double* box,double* forces) override {
        check(cudaMemcpy(x,positions,3*p.n*sizeof(double),cudaMemcpyHostToDevice));
        realKernel<<<(p.n+127)/128,128>>>(p.n,x,q,box[0],box[1],box[2],p.cutoff,p.alpha,f,e);
        std::vector<double> values(std::max(size_t(p.n),(count+255)/256));
        check(cudaGetLastError());check(cudaMemcpy(values.data(),e,p.n*sizeof(double),cudaMemcpyDeviceToHost));
        double energy=0;for(int i=0;i<p.n;i++)energy+=values[i];
        for(int c=0;c<3;c++) {
            check(cudaMemset(mesh,0,count*sizeof(cufftDoubleComplex)));
            spreadGather<<<(p.n+127)/128,128>>>(p.n,p.grid[0],p.grid[1],p.grid[2],x,q,box[0],box[1],box[2],c,mesh,f,false);
            fftcheck(cufftExecZ2Z(plan,mesh,mesh,CUFFT_FORWARD));
            size_t blocks=(count+255)/256;
            reciprocalKernel<<<blocks,256>>>(count,p.grid[0],p.grid[1],p.grid[2],box[0],box[1],box[2],p.alpha,c,mesh,e);
            check(cudaGetLastError());check(cudaMemcpy(values.data(),e,blocks*sizeof(double),cudaMemcpyDeviceToHost));
            for(size_t k=0;k<blocks;k++)energy+=values[k];
            fftcheck(cufftExecZ2Z(plan,mesh,mesh,CUFFT_INVERSE));
            spreadGather<<<(p.n+127)/128,128>>>(p.n,p.grid[0],p.grid[1],p.grid[2],x,q,box[0],box[1],box[2],c,mesh,f,true);
        }
        check(cudaGetLastError());check(cudaMemcpy(forces,f,3*p.n*sizeof(double),cudaMemcpyDeviceToHost));
        return energy;
    }
};
Engine* createCuda(const Parameters& p) { return new CudaEngine(p); }
}
