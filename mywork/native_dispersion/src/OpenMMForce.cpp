#include "Engine.h"
#include "openmm/Force.h"
#include "openmm/System.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <memory>
#include <cmath>
#include <stdexcept>
#include <limits>
using namespace OpenMM;
namespace NativeDispersion {
class DispersionForce;
class DispersionImpl : public ForceImpl {
    const DispersionForce& owner;
    std::unique_ptr<Engine> engine;
    CudaContext* cu=nullptr;
    std::unique_ptr<CudaArray> extraForce;
    CUfunction addKernel;
    std::vector<double> xyz,forces;
public:
    DispersionImpl(const DispersionForce& f):owner(f) {}
    ~DispersionImpl() { if(cu)cu->setAsCurrent(); engine.reset(); extraForce.reset(); }
    void initialize(ContextImpl& context) override;
    const Force& getOwner() const override;
    double calcForcesAndEnergy(ContextImpl& context,bool incForce,bool incEnergy,int groups) override;
    std::map<std::string,double> getDefaultParameters() override { return {}; }
    std::vector<std::string> getKernelNames() override { return {}; }
};
class DispersionForce : public Force {
public:
    Parameters params;
    explicit DispersionForce(const Parameters& p):params(p) { setName("NativeDispersionPMEMinusCutoff"); }
    bool usesPeriodicBoundaryConditions() const override { return true; }
protected:
    ForceImpl* createImpl() const override { return new DispersionImpl(*this); }
};
const Force& DispersionImpl::getOwner() const { return owner; }
void DispersionImpl::initialize(ContextImpl& context) {
    if(context.getSystem().getNumParticles()!=owner.params.n)throw std::runtime_error("Dispersion particle count mismatch");
    xyz.resize(3*owner.params.n); forces.resize(3*owner.params.n);
    auto name=context.getPlatform().getName();
    if(name=="CUDA") {
        auto& data=*static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
        if(data.contexts.size()!=1)throw std::runtime_error("Prototype dispersion supports one CUDA device only");
        cu=data.contexts[0];cu->setAsCurrent();
        if(!cu->getUseDoublePrecision())throw std::runtime_error("Prototype dispersion requires CUDA double precision");
        engine.reset(createCuda(owner.params));
        extraForce.reset(CudaArray::create<double>(*cu,forces.size(),"dispersionExtraForce"));
        CUmodule module=cu->createModule(R"(
extern "C" __global__ void addDispersion(const double* f,const int* index,unsigned long long* out,int n,int padded) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) for(int k=0;k<3;k++) atomicAdd(&out[i+k*padded],(unsigned long long)((long long)(f[3*index[i]+k]*4294967296.0)));
})");
        addKernel=cu->getKernel(module,"addDispersion");
    } else if(name=="Reference" || name=="CPU")engine.reset(createCpu(owner.params));
    else throw std::runtime_error("Native dispersion supports Reference, CPU or CUDA only");
}
double DispersionImpl::calcForcesAndEnergy(ContextImpl& context,bool incForce,bool incEnergy,int groups) {
    if(!(groups&(1<<owner.getForceGroup())))return 0;
    if(cu)cu->setAsCurrent();
    std::vector<Vec3> positions; context.getPositions(positions);
    Vec3 a,b,c;context.getPeriodicBoxVectors(a,b,c);
    if(std::abs(a[1])+std::abs(a[2])+std::abs(b[0])+std::abs(b[2])+std::abs(c[0])+std::abs(c[1])>1e-12)
        throw std::runtime_error("Prototype dispersion supports orthorhombic boxes only");
    double box[3]={a[0],b[1],c[2]};
    for(double length:box)if(!std::isfinite(length)||length<=2*owner.params.cutoff)throw std::runtime_error("Invalid dispersion box/cutoff");
    for(int i=0;i<owner.params.n;i++)for(int k=0;k<3;k++) {
        xyz[3*i+k]=positions[i][k];if(!std::isfinite(xyz[3*i+k]))throw std::runtime_error("Nonfinite dispersion position");
    }
    double energy=engine->evaluate(xyz.data(),box,forces.data());
    if(!std::isfinite(energy))throw std::runtime_error("Nonfinite native dispersion energy");
    for(double f:forces)if(!std::isfinite(f)||std::abs(f)>1e9)throw std::runtime_error("Invalid/excessive native dispersion force");
    if(incForce) {
        if(cu) {
            extraForce->upload(forces);
            int n=owner.params.n,padded=cu->getPaddedNumAtoms();
            void* args[]={&extraForce->getDevicePointer(),&cu->getAtomIndexArray().getDevicePointer(),&cu->getForce().getDevicePointer(),&n,&padded};
            cu->executeKernel(addKernel,args,n);
        } else {
            auto& data=*static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
            for(int i=0;i<owner.params.n;i++)for(int k=0;k<3;k++)(*data.forces)[i][k]+=forces[3*i+k];
        }
    }
    return incEnergy?energy:0;
}
}
static thread_local std::string lastError;
extern "C" const char* nd_last_error() { return lastError.c_str(); }
extern "C" int nd_add_force(void* system,int n,const double* coefficients,const int* grid,double alpha,double cutoff,int group) {
    try {
        if(!system||n<1||group<0||group>31||!std::isfinite(alpha)||alpha<=0||!std::isfinite(cutoff)||cutoff<=0)
            throw std::runtime_error("Invalid native dispersion parameters");
        NativeDispersion::Parameters p; p.n=n;p.alpha=alpha;p.cutoff=cutoff;
        size_t size=1;
        for(int k=0;k<3;k++) { if(grid[k]<6||grid[k]>4096)throw std::runtime_error("Invalid PME grid");p.grid[k]=grid[k];size*=grid[k]; }
        if(size>std::numeric_limits<int>::max())throw std::runtime_error("PME grid too large");
        for(int i=0;i<3*n;i++) { if(!std::isfinite(coefficients[i])||coefficients[i]<0)throw std::runtime_error("Invalid dispersion coefficient");p.q.push_back(std::sqrt(coefficients[i])); }
        auto f=std::make_unique<NativeDispersion::DispersionForce>(p);f->setForceGroup(group);
        int index=static_cast<System*>(system)->addForce(f.get());f.release();return index;
    } catch(const std::exception& e) { lastError=e.what();return -1; }
}
