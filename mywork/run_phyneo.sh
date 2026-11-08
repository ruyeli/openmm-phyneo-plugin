#!/usr/bin/env bash
# Submit from this directory with the correct account/partition for your cluster:
# sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION run_phyneo.sh
# Activate phyneo_env and load a compatible CUDA toolkit before submission.
#SBATCH --job-name=phyneo-hexane
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out
set -euo pipefail
work_dir="${PHYNEO_WORKDIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}"
env_dir="${CONDA_PREFIX:?Activate phyneo_env before submitting}"
cd "$work_dir"
[[ -f run_hexane_md_pme.py ]] || { echo 'Submit from repository/mywork or set PHYNEO_WORKDIR.' >&2; exit 1; }
export PHYNEO_OUTPUT_DIR="${PHYNEO_OUTPUT_DIR:-$work_dir/runs}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
# Default is a 10 ps NVT smoke run, not a claim of production performance.
# Override with: sbatch ... run_phyneo.sh input.pdb length_ps temperature_K nvt
exec "$env_dir/bin/python" -u run_hexane_md_pme.py \
    "${1:-hex_slab_450_2/phyneo_slab_450_whole.pdb}" "${2:-10}" "${3:-450}" "${4:-nvt}"
