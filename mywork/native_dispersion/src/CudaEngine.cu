#include "Engine.h"
#include "DispersionMath.h"
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <memory>
#include "NeighborList.cuh"
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
// One warp owns an atom. Directed pairs retain the original half-energy
// convention and avoid unordered cross-atom force atomics.
__global__ void neighborRealKernel(int n, const double* x, const double* q,
                                   const size_t* offsets, const int* neighbors,
                                   double lx, double ly, double lz, double cutoff, double alpha,
                                   double* f, double* energy) {
    int thread = blockIdx.x*blockDim.x+threadIdx.x;
    int i = thread/32, lane = threadIdx.x%32;
    if (i >= n) return; // whole warp exits together
    double box[3] = {lx, ly, lz}, fi[3] = {0, 0, 0}, ei = 0;
    for (size_t pair = offsets[i]+lane; pair < offsets[i+1]; pair += 32) {
        int j = neighbors[pair];
        double dr[3], r2 = 0;
        for (int k = 0; k < 3; k++) {
            dr[k] = x[3*i+k]-x[3*j+k];
            dr[k] -= nearbyint(dr[k]/box[k])*box[k];
            r2 += dr[k]*dr[k];
        }
        if (r2 >= cutoff*cutoff) continue; // skin never changes the physical cutoff
        if (r2 < 1e-20) { ei = NAN; continue; }
        double ff = 0;
        for (int c = 0; c < 3; c++) {
            double e, v;
            realCorrection(r2, alpha, c, e, v);
            double qq = q[3*i+c]*q[3*j+c];
            ei += .5*qq*e;
            ff += qq*v;
        }
        for (int k = 0; k < 3; k++) fi[k] += ff*dr[k];
    }
    for (int shift = 16; shift > 0; shift /= 2) {
        ei += __shfl_down_sync(0xffffffff, ei, shift);
        for (int k = 0; k < 3; k++) fi[k] += __shfl_down_sync(0xffffffff, fi[k], shift);
    }
    if (lane == 0) {
        for (int c = 0; c < 3; c++) ei += q[3*i+c]*q[3*i+c]*selfFactor(alpha, c);
        energy[i] = ei;
        for (int k = 0; k < 3; k++) f[3*i+k] = fi[k];
    }
}
__device__ inline double& meshReal(double& value) { return value; }
__device__ inline double& meshReal(cufftDoubleComplex& value) { return value.x; }
template<class GridValue>
__global__ void spreadGather(int n,int nx,int ny,int nz,const double* x,const double* q,
                             double lx,double ly,double lz,int component,GridValue* mesh,double* f,bool gather) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    int grid[3]={nx,ny,nz},ref[3]; double box[3]={lx,ly,lz},w[3][6],dw[3][6],ff[3]={0,0,0};
    for(int k=0;k<3;k++) {
        double u=(x[3*i+k]/box[k]-floor(x[3*i+k]/box[k]))*grid[k]; ref[k]=int(ceil(u));
        for(int s=0;s<6;s++) { w[k][s]=spline(ref[k]-u+s); dw[k][s]=dspline(ref[k]-u+s); }
    }
    for(int a=0;a<6;a++) for(int b=0;b<6;b++) for(int c=0;c<6;c++) {
        size_t idx=(size_t(wrap(ref[0]+a-3,nx))*ny+wrap(ref[1]+b-3,ny))*nz+wrap(ref[2]+c-3,nz);
        if(!gather) atomicAdd(&meshReal(mesh[idx]),q[3*i+component]*w[0][a]*w[1][b]*w[2][c]);
        else {
            double v=-2*q[3*i+component]*meshReal(mesh[idx]);
            ff[0]+=v*dw[0][a]*w[1][b]*w[2][c]*nx/lx;
            ff[1]+=v*w[0][a]*dw[1][b]*w[2][c]*ny/ly;
            ff[2]+=v*w[0][a]*w[1][b]*dw[2][c]*nz/lz;
        }
    }
    if(gather)for(int k=0;k<3;k++)f[3*i+k]+=ff[k];
}
template<bool halfSpectrum>
__global__ void reciprocalKernel(size_t count,int nx,int ny,int nz,double lx,double ly,double lz,double alpha,int component,cufftDoubleComplex* mesh,double* energy) {
    __shared__ double sum[256];
    size_t idx=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    double e=0;
    if(idx<count) {
        int storedZ=halfSpectrum ? nz/2+1 : nz;
        int a=idx/(ny*storedZ),b=(idx/storedZ)%ny,c=idx%storedZ;
        double g=green(a,b,c,nx,ny,nz,lx,ly,lz,alpha,component);
        e=-g*(mesh[idx].x*mesh[idx].x+mesh[idx].y*mesh[idx].y);
        // The omitted negative-kz half contributes equally to the energy.
        // kz=0 and the even-grid Nyquist plane are stored in full, not doubled.
        if (halfSpectrum && c != 0 && !(nz%2 == 0 && c == nz/2)) e *= 2;
        // Do NOT apply that multiplicity to the potential used by the inverse FFT.
        mesh[idx].x*=g;mesh[idx].y*=g;
    }
    sum[threadIdx.x]=e; __syncthreads();
    for(int s=128;s>0;s/=2) { if(threadIdx.x<s)sum[threadIdx.x]+=sum[threadIdx.x+s]; __syncthreads(); }
    if(threadIdx.x==0)energy[blockIdx.x]=sum[0];
}
class CudaEngine : public Engine {
    Parameters p;
    size_t count, spectrumCount;
    double *x=nullptr,*q=nullptr,*f=nullptr,*e=nullptr;
    cufftDoubleComplex* mesh=nullptr;
    double* realMesh=nullptr;
    cufftHandle plan=0, inversePlan=0;
    std::unique_ptr<NeighborList> neighborList;
public:
    CudaEngine(const Parameters& parameters):p(parameters),count(size_t(p.grid[0])*p.grid[1]*p.grid[2]) {
#ifdef NATIVE_DISPERSION_COMPLEX_FFT
        spectrumCount = count;
#else
        spectrumCount = size_t(p.grid[0])*p.grid[1]*(p.grid[2]/2+1);
#endif
        try {
            check(cudaMalloc(&x,3*p.n*sizeof(double))); check(cudaMalloc(&q,3*p.n*sizeof(double)));
            check(cudaMalloc(&f,3*p.n*sizeof(double)));
            check(cudaMalloc(&e,std::max(size_t(p.n),(spectrumCount+255)/256)*sizeof(double)));
            check(cudaMalloc(&mesh,spectrumCount*sizeof(cufftDoubleComplex)));
            check(cudaMemcpy(q,p.q.data(),3*p.n*sizeof(double),cudaMemcpyHostToDevice));
#ifdef NATIVE_DISPERSION_COMPLEX_FFT
            fftcheck(cufftPlan3d(&plan,p.grid[0],p.grid[1],p.grid[2],CUFFT_Z2Z));
#else
            check(cudaMalloc(&realMesh,count*sizeof(double)));
            fftcheck(cufftPlan3d(&plan,p.grid[0],p.grid[1],p.grid[2],CUFFT_D2Z));
            fftcheck(cufftPlan3d(&inversePlan,p.grid[0],p.grid[1],p.grid[2],CUFFT_Z2D));
#endif
#ifndef NATIVE_DISPERSION_ALL_PAIRS
            neighborList.reset(new NeighborList(p.n, p.cutoff));
#endif
        } catch(...) { release(); throw; }
    }
    void release() { if(plan)cufftDestroy(plan);if(inversePlan)cufftDestroy(inversePlan);cudaFree(x);cudaFree(q);cudaFree(f);cudaFree(e);cudaFree(mesh);cudaFree(realMesh); }
    ~CudaEngine() { release(); }
    size_t getNeighborListBuildCount() const override { return neighborList ? neighborList->getBuildCount() : 0; }
    size_t getNeighborListPairCount() const override { return neighborList ? neighborList->getPairCount() : 0; }
    double evaluate(const double* positions,const double* box,double* forces) override {
        check(cudaMemcpy(x,positions,3*p.n*sizeof(double),cudaMemcpyHostToDevice));
#ifdef NATIVE_DISPERSION_ALL_PAIRS
        realKernel<<<(p.n+127)/128,128>>>(p.n,x,q,box[0],box[1],box[2],p.cutoff,p.alpha,f,e);
#else
        neighborList->update(x, box);
        neighborRealKernel<<<(p.n+3)/4,128>>>(p.n, x, q, neighborList->getOffsets(), neighborList->getNeighbors(),
                                             box[0], box[1], box[2], p.cutoff, p.alpha, f, e);
#endif
        std::vector<double> values(std::max(size_t(p.n),(spectrumCount+255)/256));
        check(cudaGetLastError());check(cudaMemcpy(values.data(),e,p.n*sizeof(double),cudaMemcpyDeviceToHost));
        double energy=0;for(int i=0;i<p.n;i++)energy+=values[i];
        for(int c=0;c<3;c++) {
#ifdef NATIVE_DISPERSION_COMPLEX_FFT
            check(cudaMemset(mesh,0,count*sizeof(cufftDoubleComplex)));
            spreadGather<<<(p.n+127)/128,128>>>(p.n,p.grid[0],p.grid[1],p.grid[2],x,q,box[0],box[1],box[2],c,mesh,f,false);
            fftcheck(cufftExecZ2Z(plan,mesh,mesh,CUFFT_FORWARD));
            size_t blocks=(spectrumCount+255)/256;
            reciprocalKernel<false><<<blocks,256>>>(spectrumCount,p.grid[0],p.grid[1],p.grid[2],box[0],box[1],box[2],p.alpha,c,mesh,e);
#else
            check(cudaMemset(realMesh,0,count*sizeof(double)));
            spreadGather<<<(p.n+127)/128,128>>>(p.n,p.grid[0],p.grid[1],p.grid[2],x,q,box[0],box[1],box[2],c,realMesh,f,false);
            fftcheck(cufftExecD2Z(plan,realMesh,mesh));
            size_t blocks=(spectrumCount+255)/256;
            reciprocalKernel<true><<<blocks,256>>>(spectrumCount,p.grid[0],p.grid[1],p.grid[2],box[0],box[1],box[2],p.alpha,c,mesh,e);
#endif
            check(cudaGetLastError());check(cudaMemcpy(values.data(),e,blocks*sizeof(double),cudaMemcpyDeviceToHost));
            for(size_t k=0;k<blocks;k++)energy+=values[k];
#ifdef NATIVE_DISPERSION_COMPLEX_FFT
            fftcheck(cufftExecZ2Z(plan,mesh,mesh,CUFFT_INVERSE));
            spreadGather<<<(p.n+127)/128,128>>>(p.n,p.grid[0],p.grid[1],p.grid[2],x,q,box[0],box[1],box[2],c,mesh,f,true);
#else
            // cuFFT inverses are unnormalized, matching the original Z2Z/DMFF convention.
            fftcheck(cufftExecZ2D(inversePlan,mesh,realMesh));
            spreadGather<<<(p.n+127)/128,128>>>(p.n,p.grid[0],p.grid[1],p.grid[2],x,q,box[0],box[1],box[2],c,realMesh,f,true);
#endif
        }
        check(cudaGetLastError());check(cudaMemcpy(forces,f,3*p.n*sizeof(double),cudaMemcpyDeviceToHost));
        return energy;
    }
};
Engine* createCuda(const Parameters& p) { return new CudaEngine(p); }
}
