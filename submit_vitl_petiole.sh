#!/bin/bash
#SBATCH -C gpu
#SBATCH -A m4880_g
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH -o output/dinov3_vitl_petiole.out
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=xchong@lbl.gov

module load python
source /global/common/software/nersc/pe/conda/24.10.0/Miniforge3-24.7.1-0/bin/activate /pscratch/sd/x/xchong/envs/lightly

# IMPORTANT: Unset these to prevent SLURM-Lightning conflicts
unset SLURM_NTASKS
unset SLURM_NTASKS_PER_NODE

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --list-gpus

python train_vitl_petiole.py