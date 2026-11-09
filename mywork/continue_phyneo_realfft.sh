#!/usr/bin/env bash
# Activate phyneo_env/load CUDA before submitting from repository/mywork:
# sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION continue_phyneo_realfft.sh SOURCE_RUN [additional_ps]
#SBATCH --job-name=phyneo-realfft-restart
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=slurm-realfft-%j.out
set -euo pipefail
if (( $# < 1 || $# > 2 )); then
    echo 'Usage: sbatch continue_phyneo_realfft.sh SOURCE_RUN [additional_ps]' >&2
    exit 2
fi
work_dir="${PHYNEO_WORKDIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}}"
env_dir="${CONDA_PREFIX:?Activate phyneo_env before submitting}"
cd "$work_dir"
[[ -f continue_hexane_realfft.py ]] || { echo 'Submit from repository/mywork or set PHYNEO_WORKDIR.' >&2; exit 1; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
# Default destination is SOURCE_RUN's parent. PHYNEO_OUTPUT_DIR can override it.
# No duration means finish the original target; otherwise add this many ps.
exec "$env_dir/bin/python" -u -B continue_hexane_realfft.py "$@"
