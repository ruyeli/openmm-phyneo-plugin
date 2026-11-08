// Dispersion PME formulas adapted from DMFF (LGPL-3.0); see NOTICE.txt.
#pragma once
#include <cmath>
#ifdef __CUDACC__
#define ND_HD __host__ __device__
#else
#define ND_HD
#endif
namespace NativeDispersion {
constexpr double pi = 3.1415926535897932384626433832795;
ND_HD inline int wrap(int i, int n) { return (i%n+n)%n; }
ND_HD inline double spline(double u) {
    // Cardinal B6, evaluated in the numerically stable half of its support.
    if (u<0 || u>=6) return 0;
    if (u>3) u=6-u;
    double v=u*u*u*u*u;
    if (u>=1) v-=6*pow(u-1,5);
    if (u>=2) v+=15*pow(u-2,5);
    return v/120;
}
ND_HD inline double dspline(double u) {
    if (u<0 || u>=6) return 0;
    double sign=1;
    if (u>3) { u=6-u; sign=-1; }
    double v=pow(u,4);
    if (u>=1) v-=6*pow(u-1,4);
    if (u>=2) v+=15*pow(u-2,4);
    return sign*v/24;
}
ND_HD inline double theta(int index, int n) {
    double w=2*pi*index/n;
    return (66+52*cos(w)+2*cos(2*w))/120;
}
ND_HD inline double green(int ix,int iy,int iz,int nx,int ny,int nz,
                          double lx,double ly,double lz,double alpha,int component) {
    int kx=ix<=(nx-1)/2?ix:ix-nx;
    int ky=iy<=(ny-1)/2?iy:iy-ny;
    int kz=iz<=(nz-1)/2?iz:iz-nz;
    double k2=4*pi*pi*(kx*kx/(lx*lx)+ky*ky/(ly*ly)+kz*kz/(lz*lz));
    double t=k2/(4*alpha*alpha), x=sqrt(t), ex=exp(-t), f, denom;
    if (component==0) { f=(1-2*t)*ex+2*x*t*sqrt(pi)*erfc(x); denom=3; }
    else if (component==1) { f=(3-2*t+4*t*t)*ex-4*x*t*t*sqrt(pi)*erfc(x); denom=45; }
    else { f=(15-6*t+4*t*t-8*t*t*t)*ex+8*x*t*t*t*sqrt(pi)*erfc(x); denom=1260; }
    double th=theta(ix,nx)*theta(iy,ny)*theta(iz,nz);
    return sqrt(pi)*pi/(2*lx*ly*lz)*pow(alpha,3+2*component)*f/(denom*th*th);
}
ND_HD inline void realCorrection(double r2, double alpha, int component, double& e, double& f) {
    int m=3+component, p=2*m;
    double t=alpha*alpha*r2, term=1, sum=1, factorial=1;
    for(int j=1;j<m;j++) { term*=t/j; sum+=term; factorial*=j; }
    double h;
    if(t<.5) {
        term*=t/m; h=term;
        for(int j=m+1;j<m+30;j++) { term*=t/j; h+=term; }
        h*=exp(-t);
    } else h=1-exp(-t)*sum;
    double inv=pow(r2,-m);
    e=h*inv;
    // F_i = f*(r_i-r_j) for the positive real-space PME-minus-cutoff correction.
    f=(p*h-2*exp(-t)*pow(t,m)/factorial)*inv/r2;
}
ND_HD inline double selfFactor(double alpha,int component) {
    double denominators[3]={12,48,240};
    return pow(alpha,6+2*component)/denominators[component];
}
}
