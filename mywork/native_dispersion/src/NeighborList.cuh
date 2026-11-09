#pragma once
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace NativeDispersion {
inline void neighborCheck(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__device__ inline int cellCoordinate(double x, double length, int cells) {
    double fraction = x/length-floor(x/length);
    return min(cells-1, max(0, int(fraction*cells)));
}

__global__ void binParticles(int n, const double* x, double lx, double ly, double lz,
                            int nx, int ny, int nz, int* heads, int* next) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    int a = cellCoordinate(x[3*i], lx, nx);
    int b = cellCoordinate(x[3*i+1], ly, ny);
    int c = cellCoordinate(x[3*i+2], lz, nz);
    next[i] = atomicExch(&heads[(a*ny+b)*nz+c], i);
}

__global__ void checkListDisplacement(int n, const double* x, const double* reference,
                                     double lx, double ly, double lz, double limit2, int* rebuild) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    double lengths[3] = {lx, ly, lz}, distance2 = 0;
    for (int k = 0; k < 3; k++) {
        double d = x[3*i+k]-reference[3*i+k];
        d -= nearbyint(d/lengths[k])*lengths[k];
        distance2 += d*d;
    }
    if (distance2 >= limit2) atomicExch(rebuild, 1);
}

// Count and fill share exactly the same traversal. Directed lists deliberately
// include ALL nearby atoms, including bonded pairs excluded by other forces.
template<bool fill>
__global__ void findNeighbors(int n, const double* x, double lx, double ly, double lz,
                              int nx, int ny, int nz, double radius2, const int* heads,
                              const int* next, size_t* counts, const size_t* offsets, int* neighbors) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    int cell[3] = {cellCoordinate(x[3*i], lx, nx), cellCoordinate(x[3*i+1], ly, ny),
                   cellCoordinate(x[3*i+2], lz, nz)};
    size_t count = 0;
    // min(3, ncell) avoids visiting duplicate periodic cells for one/two bins.
    for (int a = 0; a < min(3, nx); a++) {
        int cx = (cell[0]+a-1+nx)%nx;
        for (int b = 0; b < min(3, ny); b++) {
            int cy = (cell[1]+b-1+ny)%ny;
            for (int c = 0; c < min(3, nz); c++) {
                int cz = (cell[2]+c-1+nz)%nz;
                for (int j = heads[(cx*ny+cy)*nz+cz]; j != -1; j = next[j]) {
                    if (j == i) continue;
                    double dx = x[3*i]-x[3*j], dy = x[3*i+1]-x[3*j+1], dz = x[3*i+2]-x[3*j+2];
                    dx -= nearbyint(dx/lx)*lx;
                    dy -= nearbyint(dy/ly)*ly;
                    dz -= nearbyint(dz/lz)*lz;
                    if (dx*dx+dy*dy+dz*dz < radius2) {
                        if (fill) neighbors[offsets[i]+count] = j;
                        count++;
                    }
                }
            }
        }
    }
    if (!fill) counts[i] = count;
}

class NeighborList {
    int n;
    double skin = 0.2, radius, savedBox[3] = {0, 0, 0};
    bool valid = false;
    int *heads = nullptr, *next = nullptr, *neighbors = nullptr, *rebuild = nullptr;
    size_t *counts = nullptr, *offsets = nullptr;
    double* reference = nullptr;
    size_t headCapacity = 0, pairCapacity = 0, pairCount = 0, builds = 0;
public:
    NeighborList(int atoms, double cutoff): n(atoms), radius((cutoff+skin)*(1+1e-12)) {
        try {
            neighborCheck(cudaMalloc(&next, size_t(n)*sizeof(int)));
            neighborCheck(cudaMalloc(&rebuild, sizeof(int)));
            neighborCheck(cudaMalloc(&counts, (size_t(n)+1)*sizeof(size_t)));
            neighborCheck(cudaMalloc(&offsets, (size_t(n)+1)*sizeof(size_t)));
            neighborCheck(cudaMalloc(&reference, 3*size_t(n)*sizeof(double)));
        } catch (...) { release(); throw; }
    }
    NeighborList(const NeighborList&) = delete;
    NeighborList& operator=(const NeighborList&) = delete;
    ~NeighborList() { release(); }
    void release() {
        cudaFree(heads); cudaFree(next); cudaFree(neighbors); cudaFree(rebuild);
        cudaFree(counts); cudaFree(offsets); cudaFree(reference);
    }
    const size_t* getOffsets() const { return offsets; }
    const int* getNeighbors() const { return neighbors; }
    size_t getBuildCount() const { return builds; }
    size_t getPairCount() const { return pairCount; }

    void update(const double* x, const double* box) {
        bool needed = !valid || !std::equal(box, box+3, savedBox);
        if (!needed) {
            neighborCheck(cudaMemset(rebuild, 0, sizeof(int)));
            checkListDisplacement<<<(n+127)/128,128>>>(n, x, reference, box[0], box[1], box[2],
                                                       skin*skin*0.25, rebuild);
            int flag;
            neighborCheck(cudaGetLastError());
            neighborCheck(cudaMemcpy(&flag, rebuild, sizeof(int), cudaMemcpyDeviceToHost));
            needed = flag != 0;
        }
        if (!needed) return;
        valid = false;
        // Cap empty-space storage. Coarsening cells preserves cell width >= radius.
        size_t maxCells = std::min(size_t(std::numeric_limits<int>::max()), std::max(size_t(64), 4*size_t(n)));
        int cells[3];
        for (int k = 0; k < 3; k++)
            cells[k] = int(std::max(1.0, std::min(double(maxCells), std::floor(box[k]/radius))));
        auto tooMany = [&]() { return size_t(cells[0]) > maxCells/size_t(cells[1])/size_t(cells[2]); };
        while (tooMany()) {
            int k = int(std::max_element(cells, cells+3)-cells);
            cells[k] = cells[k]/2+cells[k]%2;
        }
        size_t numCells = size_t(cells[0])*cells[1]*cells[2];
        if (numCells > headCapacity) {
            neighborCheck(cudaFree(heads)); heads = nullptr;
            neighborCheck(cudaMalloc(&heads, numCells*sizeof(int)));
            headCapacity = numCells;
        }
        neighborCheck(cudaMemset(heads, 0xff, numCells*sizeof(int)));
        binParticles<<<(n+127)/128,128>>>(n, x, box[0], box[1], box[2], cells[0], cells[1], cells[2], heads, next);
        findNeighbors<false><<<(n+127)/128,128>>>(n, x, box[0], box[1], box[2], cells[0], cells[1], cells[2],
                                                 radius*radius, heads, next, counts, offsets, neighbors);
        neighborCheck(cudaGetLastError());
        neighborCheck(cudaMemset(counts+n, 0, sizeof(size_t)));
        thrust::exclusive_scan(thrust::device_pointer_cast(counts), thrust::device_pointer_cast(counts+n+1),
                               thrust::device_pointer_cast(offsets));
        neighborCheck(cudaMemcpy(&pairCount, offsets+n, sizeof(size_t), cudaMemcpyDeviceToHost));
        if (pairCount > std::numeric_limits<size_t>::max()/sizeof(int))
            throw std::runtime_error("Neighbor list allocation size overflow");
        if (pairCount > pairCapacity) {
            neighborCheck(cudaFree(neighbors)); neighbors = nullptr;
            neighborCheck(cudaMalloc(&neighbors, pairCount*sizeof(int)));
            pairCapacity = pairCount;
        }
        findNeighbors<true><<<(n+127)/128,128>>>(n, x, box[0], box[1], box[2], cells[0], cells[1], cells[2],
                                                radius*radius, heads, next, counts, offsets, neighbors);
        neighborCheck(cudaGetLastError());
        neighborCheck(cudaMemcpy(reference, x, 3*size_t(n)*sizeof(double), cudaMemcpyDeviceToDevice));
        std::copy(box, box+3, savedBox);
        builds++;
        valid = true;
    }
};
}
