#!/bin/bash
#SBATCH -q regular
#SBATCH -A amsc006
#SBATCH --reservation=_CAP_SYNAPYIDINOSAM
#SBATCH -N 1                          # 4 nodes = 16 GPUs total
#SBATCH -C gpu&hbm80g
#SBATCH --time=06:00:00               # Reduce to 1 hour (500 images takes ~10 min)
#SBATCH --ntasks-per-node=1           
#SBATCH --gpus-per-node=4             
#SBATCH --cpus-per-task=128           
#SBATCH --output=logs/job_%j.out      # %j = job ID
#SBATCH --error=logs/job_%j.err

module load python
source /global/common/software/nersc/pe/conda/24.10.0/Miniforge3-24.7.1-0/bin/activate /pscratch/sd/x/xchong/envs/lightly

# IMPORTANT: Unset these to prevent SLURM-Lightning conflicts
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --list-gpus

python train_vitl_petiole.py