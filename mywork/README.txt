Portable hexane OPLS + PhyNEO physical-potential workflow
======================================================

This branch carries the current implementation, not a performance-optimized
release. No ML bonding or JAX runtime is used by the OpenMM MD runner.
The original upstream README still describes the general plugin.

On the destination Linux GPU cluster:

  git clone -b native-dispersion-pme https://github.com/ruyeli/openmm-phyneo-plugin.git
  cd openmm-phyneo-plugin
  conda env create -f environment.yml
  conda activate phyneo_env
  # Load your site's CUDA toolkit (nvcc) and a compatible C++ compiler.
  # If needed: export CXX=/path/to/g++; export CC=/path/to/gcc
  # If nvcc is not on PATH: export CUDACXX=/path/to/cuda/bin/nvcc
  bash scripts/build_portable.sh
  python mywork/validate_build.py
  cd mywork
  python run_hexane_md_pme.py hex_vap_430/vap_conn.pdb 0.1 430 nvt

Build and validation must run in an allocation with CUDA/GPU access. The native
CUDA architecture defaults to the build GPU; rebuild for different GPUs (or set
PHYNEO_CUDA_ARCHITECTURES to an appropriate CMake architecture list). Compiler,
toolkit and driver must be mutually compatible. BUILD_JOBS defaults to 2.
This portable build path has only received static/argument checks on the source
cluster's login node; compile and validate it on the destination before MD.
This handoff adds no precompiled libraries or build caches. Legacy upstream
build artifacts may exist in the inherited tree; do not use them. Validation records are
local, ignored by Git, and tied to source, model, environment and binary hashes.
A build alone does not enable MD: validate_build.py must succeed first.

Inputs and outputs
-----------------
The four arguments are input PDB, duration in ps, temperature in K, nvt|npt.
Defaults: 1 fs, CUDA double, mutual polarization tolerance 1e-5, max 1000
iterations, 0.6 nm cutoff, 1 bar isotropic NPT. Use NVT for the fixed-box slab.
DCD/checkpoint/CSV are written every 1 ps; one final structure PDB is written,
not a PDB trajectory. Set PHYNEO_OUTPUT_DIR to a scratch/project filesystem.
Each run creates a unique directory. The runner does not automatically resume.
Transfer existing trajectories/checkpoints separately if you need them;
Git carries implementation and starting configurations, not simulation history.

For Slurm, activate the environment/load CUDA, then from repository/mywork:

  export PHYNEO_OUTPUT_DIR=/your/project/hexane_450_slab
  sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION run_phyneo.sh

Adapt GPU/resource flags to the new site. The default submission runs 10 ps of
the repaired 1323-molecule slab at 450 K, NVT. Optional four script arguments
override those defaults. The 48-hour limit is not a promise any duration fits.

Included: 20-molecule vapor, 100-molecule liquid, repaired 1323-molecule slab,
converted XML, original physical output.xml and OPLS LIG.openmm.xml, converter,
parameter-construction script, regression tests and CPU-DMFF vapor fixtures.
The converter accepts connectivity-bearing OPLS PDBs; see --help. It is for
linear hexane, not an arbitrary chemistry/topology conversion.

Numerical corrections preserved
------------------------------
1. CUDA induced-field off-diagonal tile uses atom2 = y*TILE_SIZE+tj to match
   localData[tbx+tj], not the loop counter j. The diagonal branch is unchanged.
2. Dense scale-array uploads use size_t byte counts through cuMemcpyHtoD,
   avoiding the signed-int byte overflow seen in the 26460-atom slab.
3. This model explicitly uses native mScale=pScale=dScale=[1,0,0,0,0,0].
   Native pScale and dScale both receive the DMFF pScale convention used by
   this model. This is not a general claim that all force fields share scales.
4. C6/C8/C10 native dispersion adds PME minus raw cutoff exactly once (group
   30) to the existing damped short-range force. It preserves DMFF's grid,
   alpha, sixth-order spline, self and k=0 conventions. The correction includes
   excluded bonded pairs where the cutoff/PME scaling cancellation requires it.
   Box-dependent terms are reevaluated for NPT; only orthorhombic cells work.

Native sources and LGPL provenance are in native_dispersion/NOTICE.txt and
LICENSE.DMFF; pocketfft retains its own BSD notice. Python only constructs the
force; all runtime evaluations are C++/CUDA. System XML serialization of the
custom force is not implemented; use the runner's reconstructed system and
checkpoint/State outputs. CUDA support is double precision, one GPU.

Validation and current limits
-----------------------------
Historical source-cluster tests passed full damped physical-energy comparisons
against CPU DMFF on vapor, liquid and slab. CUDA minus DMFF energy differences
were approximately 2.86e-7, 3.06e-5 and 2.07e-4 kJ/mol; slab maximum force error
was 5.76e-5 kJ/mol/nm. Small residuals include OpenMM short-range parameter
storage precision. These are historical results, not new-cluster certification.
The shipped vapor fixture tests the dispersion correction and reconstructed
raw dispersion, not a full independent reference for every force component.
validate_build.py also runs plugin/native CTest, model/scaling tests and a short
hybrid CUDA-double MD smoke test. Inspect failed logs rather than bypassing it.

Latest profiling: about 0.38 ns/day for the 1323-molecule slab on RTX PRO 6000
Blackwell (roughly 229 ms/step). Native correction costs about 166 ms/step:
all-pairs real space ~74 ms, FFT ~62 ms, reciprocal Green evaluation ~22 ms.
Reporting/actual transfer costs are small. These optimizations remain pending:
neighbor lists, NVT Green caching and same-grid FFT work. Rounding the grid is
not adopted because it changes the DMFF comparison. Raw profiler metadata stays local.
No optimization or new GPU calculation was done during this Git handoff.
