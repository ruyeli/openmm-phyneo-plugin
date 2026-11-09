Portable hexane OPLS + PhyNEO physical-potential workflow
======================================================

This branch carries the optimized implementation: GPU dispersion neighbor
lists and real FFTs for dispersion and double-precision orthorhombic
electrostatics. It remains experimental. No ML bonding or JAX runtime is used.
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
Compile and validate on the destination before MD; historical source-cluster
results do not certify a new compiler, GPU or OpenMM environment.
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
Each run creates a unique directory. For checkpoint continuation use
continue_hexane_realfft.py or continue_phyneo_realfft.sh, described below.
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
Those regression files/fixtures were already present in the earlier branch.
This optimization update adds no test files, diagnostic folders or raw logs.
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

Implemented optimizations
-------------------------
Dispersion now uses a GPU cell/Verlet neighbor list with a 0.2 nm skin and
displacement/box-change rebuilds. Required bonded correction pairs are retained.
Box-dependent reciprocal factors are cached until the box changes. Each of
C6/C8/C10 uses double real-to-complex/complex-to-real FFTs on the unchanged mesh;
half-spectrum energy weights preserve zero and Nyquist conventions.

Permanent and induced-dipole PME use double real FFTs for initially
orthorhombic boxes. Single/mixed precision and initially triclinic boxes retain
complex FFTs. The odd-grid reciprocal multiplier is conjugate-averaged to
preserve the legacy real-valued result. Complex Fourier coefficients remain
in the nonredundant half-spectrum; imaginary information is not discarded.
The redundant initial induced-field FFT pair is skipped only in Mutual mode
with iterations; convergence tolerance and iteration count are unchanged.
Five polarization iterations now mean 18 transforms per full force evaluation:
2 permanent + 10 induced + 6 dispersion, all real FFTs for this runner.
No grid rounding, warm starts, cross-component FFT reuse or model changes.

Latest source-cluster measurements, RTX PRO 6000 Blackwell, 1323 molecules:
  complex electrostatics + optimized real dispersion: 112.54 ms/step;
  real electrostatics + optimized real dispersion:     97.50 ms/step.
The latter is 0.886 ns/day at 1 fs, 15.4% higher throughput. These were paired
2000-step timings; gains for the 100-molecule box were much smaller (~1%).
Forty matched NVE/NPT snapshots agreed within 6.6e-12 kJ/mol in energy,
1.7e-9 kJ/mol/nm in force, and 8.6e-14 e nm in induced dipoles.
Changed-box and checkpoint-continuation checks passed. Full damped dispersion
agreement with CPU DMFF is retained at the tolerances described above.

Important limits: both FFT builds show substantial total-energy fluctuations
in the short NVE checks. Earlier full initcheck reported 5127 warnings in each
build, with identical counts outside the changed FFT kernels; their physical
relevance remains unresolved. Fresh-context replay also differs slightly from
stored MD evaluations in both builds. FFT equivalence is not proof of full
energy conservation, initialization safety or long-time ensemble accuracy.
Raw diagnostics, test outputs, build products and trajectories stay off Git.

Continuation with all real FFTs
------------------------------
With the local build validated, from repository/mywork:
  python -u -B continue_hexane_realfft.py /path/to/source_run
or:
  sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION continue_phyneo_realfft.sh /path/to/source_run
An optional final argument is an ADDITIONAL duration in ps. Without it, resume
until the source manifest's original total target step. For example, a source
checkpoint at 783 ps in a 10 ns run has 9217 ps remaining.

The runner requires source_run/run.json and restart.chk, plus the original
topology PDB with matching hash. If its path changed, set PHYNEO_INPUT_PDB to
the transferred identical PDB. The new libraries are selected from the local
build manifest, never from stale source-run binary paths. Physical settings,
OpenMM version, box and PME meshes must match. Binary checkpoints can still
be incompatible across GPU/platform/OpenMM environments: failure is explicit,
with no silent PDB/State fallback.

No minimization or velocity reassignment follows checkpoint loading. A new
uniquely named directory is created beside the source (or in PHYNEO_OUTPUT_DIR).
It contains DCD/CSV, atomically replaced checkpoints, provenance and final State
XML; no PDB trajectory. Standard OpenMM DCD segment numbering is retained.
The source run is never appended to or overwritten.

The 48-hour Slurm template requests a stop signal 180 seconds before expiry;
the runner checks signals every <=1000 steps and saves a restart. Hard kills
cannot be caught. At 0.886 ns/day, another 9.217 ns takes roughly 10.4 days,
requiring several allocations. Resubmit with the NEW output directory as
source_run; omitting duration preserves its outstanding target_total_steps.
No automatic job submission/chaining is performed. The source-cluster runner
passed a 10-step original-checkpoint restart followed by a 5-step restart of
that new checkpoint. Slurm wall-time expiry itself was not smoke-tested.
