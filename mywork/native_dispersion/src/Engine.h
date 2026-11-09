#pragma once
#include <vector>
#include <cstddef>
namespace NativeDispersion {
struct Parameters {
    int n, grid[3];
    double alpha, cutoff;
    std::vector<double> q; // particle-major sqrt(C6), sqrt(C8), sqrt(C10), nm units
};
class Engine {
public:
    virtual ~Engine() {}
    virtual double evaluate(const double* positions, const double* lengths, double* forces)=0;
    // Diagnostics used by rebuild/reuse tests. The CPU reference has no list.
    virtual size_t getNeighborListBuildCount() const { return 0; }
    virtual size_t getNeighborListPairCount() const { return 0; }
};
Engine* createCpu(const Parameters& p);
Engine* createCuda(const Parameters& p);
}
