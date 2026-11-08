#include "Engine.h"
#include "DispersionMath.h"
#include <memory>
#include <iostream>
#include <cmath>
#include <stdexcept>
using namespace NativeDispersion;
static void close(double a,double b,double tol) { if(std::abs(a-b)>tol)throw std::runtime_error("Numerical assertion failed"); }
int main() {
    try {
        Parameters p; p.n=2;p.grid[0]=16;p.grid[1]=18;p.grid[2]=20;p.alpha=3.;p.cutoff=.8;
        p.q={.03,.01,.002,.04,.012,.003};
        double x[]={.33,.57,.89,.61,.74,.93},box[]={2.,2.2,2.4},f[6];
        std::unique_ptr<Engine> e(createCpu(p));
        double energy=e->evaluate(x,box,f);
        for(int i=0;i<6;i++) {
            double tmp[6],h=1e-6;x[i]+=h;double plus=e->evaluate(x,box,tmp);
            x[i]-=2*h;double minus=e->evaluate(x,box,tmp);x[i]+=h;
            close(f[i],-(plus-minus)/(2*h),1e-6);
        }
        double shifted[6];x[0]+=box[0];close(energy,e->evaluate(x,box,shifted),1e-10);x[0]-=box[0];
        for(int i=0;i<6;i++)close(f[i],shifted[i],1e-9);
        double sum=0,fsum[6]={};
        for(int c=0;c<3;c++) {
            Parameters single=p;
            for(int i=0;i<6;i++)if(i%3!=c)single.q[i]=0;
            std::unique_ptr<Engine> isolated(createCpu(single));double ff[6];sum+=isolated->evaluate(x,box,ff);
            for(int i=0;i<6;i++)fsum[i]+=ff[i];
        }
        close(sum,energy,1e-10);for(int i=0;i<6;i++)close(fsum[i],f[i],1e-9);
        for(double u=.01;u<5.99;u+=.071)close(dspline(u),(spline(u+1e-6)-spline(u-1e-6))/2e-6,1e-8);
        std::cout<<"PASS: energy/force finite differences, image invariance, C6/C8/C10 additivity, B6 derivative\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl;return 1; }
}
