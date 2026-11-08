#include "Engine.h"
#include "DispersionMath.h"
#include "pocketfft_hdronly.h"
#include <complex>
#include <algorithm>
#include <array>
#include <stdexcept>
namespace NativeDispersion {
class CpuEngine : public Engine {
    Parameters p;
    std::vector<std::complex<double>> mesh;
public:
    CpuEngine(const Parameters& parameters):p(parameters),mesh(size_t(p.grid[0])*p.grid[1]*p.grid[2]) {}
    double evaluate(const double* x,const double* box,double* forces) override {
        std::fill(forces,forces+3*p.n,0.0);
        double energy=0;
        // Baseline correctness implementation. Real-space neighbor-list optimization is pending.
        for(int i=0;i<p.n;i++) for(int j=i+1;j<p.n;j++) {
            double dr[3],r2=0;
            for(int k=0;k<3;k++) { dr[k]=x[3*i+k]-x[3*j+k]; dr[k]-=nearbyint(dr[k]/box[k])*box[k]; r2+=dr[k]*dr[k]; }
            if(r2>=p.cutoff*p.cutoff) continue;
            if(r2<1e-20) throw std::runtime_error("Coincident particles in dispersion correction");
            double fsum=0;
            for(int c=0;c<3;c++) { double e,f; realCorrection(r2,p.alpha,c,e,f); double q=p.q[3*i+c]*p.q[3*j+c]; energy+=q*e; fsum+=q*f; }
            for(int k=0;k<3;k++) { forces[3*i+k]+=fsum*dr[k]; forces[3*j+k]-=fsum*dr[k]; }
        }
        auto visit=[&](int i,auto action) {
            int ref[3]; double w[3][6],dw[3][6];
            for(int k=0;k<3;k++) {
                double u=(x[3*i+k]/box[k]-floor(x[3*i+k]/box[k]))*p.grid[k];
                ref[k]=int(ceil(u));
                for(int s=0;s<6;s++) { w[k][s]=spline(ref[k]-u+s); dw[k][s]=dspline(ref[k]-u+s); }
            }
            for(int a=0;a<6;a++) for(int b=0;b<6;b++) for(int c=0;c<6;c++) {
                size_t idx=(size_t(wrap(ref[0]+a-3,p.grid[0]))*p.grid[1]+wrap(ref[1]+b-3,p.grid[1]))*p.grid[2]+wrap(ref[2]+c-3,p.grid[2]);
                action(idx,w[0][a]*w[1][b]*w[2][c],dw[0][a]*w[1][b]*w[2][c],w[0][a]*dw[1][b]*w[2][c],w[0][a]*w[1][b]*dw[2][c]);
            }
        };
        pocketfft::shape_t shape{size_t(p.grid[0]),size_t(p.grid[1]),size_t(p.grid[2])},axes{0,1,2};
        pocketfft::stride_t stride{ptrdiff_t(sizeof(std::complex<double>)*p.grid[1]*p.grid[2]),ptrdiff_t(sizeof(std::complex<double>)*p.grid[2]),ptrdiff_t(sizeof(std::complex<double>))};
        for(int component=0;component<3;component++) {
            std::fill(mesh.begin(),mesh.end(),std::complex<double>(0,0));
            for(int i=0;i<p.n;i++) visit(i,[&](size_t idx,double w,double,double,double) { mesh[idx]+=p.q[3*i+component]*w; });
            pocketfft::c2c(shape,stride,stride,axes,true,mesh.data(),mesh.data(),1.0,1);
            for(int a=0;a<p.grid[0];a++) for(int b=0;b<p.grid[1];b++) for(int c=0;c<p.grid[2];c++) {
                size_t idx=(size_t(a)*p.grid[1]+b)*p.grid[2]+c;
                double g=green(a,b,c,p.grid[0],p.grid[1],p.grid[2],box[0],box[1],box[2],p.alpha,component);
                energy-=g*std::norm(mesh[idx]); mesh[idx]*=g;
            }
            pocketfft::c2c(shape,stride,stride,axes,false,mesh.data(),mesh.data(),1.0,1);
            for(int i=0;i<p.n;i++) {
                visit(i,[&](size_t idx,double,double dx,double dy,double dz) {
                    double v=-2*p.q[3*i+component]*mesh[idx].real();
                    forces[3*i]+=v*dx*p.grid[0]/box[0];
                    forces[3*i+1]+=v*dy*p.grid[1]/box[1];
                    forces[3*i+2]+=v*dz*p.grid[2]/box[2];
                });
                energy+=p.q[3*i+component]*p.q[3*i+component]*selfFactor(p.alpha,component);
            }
        }
        return energy;
    }
};
Engine* createCpu(const Parameters& p) { return new CpuEngine(p); }
}
