#!/bin/bash
#SBATCH -q regular
#SBATCH -A amsc006
#SBATCH --reservation=_CAP_March_ModCon_Dry_Run_GPU
#SBATCH -N 8
#SBATCH -C gpu&hbm80g
#SBATCH --time=05:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# ============================================================
# MASTER NODE
# ============================================================
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500


module load conda


conda activate lightly

srun --ntasks-per-node=1 --gpus-per-task=4 bash -c "
torchrun \
    --nnodes=\$SLURM_NNODES \
    --nproc_per_node=4 \
    --rdzv_id=\${SLURM_JOB_ID}1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    -m train_vith_petiole
"

