#!/bin/bash
# =============================================================================
# SLURM Job Submission Script for HPC Training of Yoga Pose Classifier
# =============================================================================
#SBATCH --job-name=yoga_pose_classifier
#SBATCH --output=slurm_logs/train_%j.out
#SBATCH --error=slurm_logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00

echo "=========================================================="
echo "Starting HPC Yoga Pose Classifier Training Job: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $(date)"
echo "=========================================================="

# 1. Create logs directory
mkdir -p slurm_logs
mkdir -p hpc_output

# 2. Activate Python Environment (adjust path as needed on your cluster)
# e.g., conda activate yoga_env or source venv/bin/activate
module load cuda/12.1 python/3.11 2>/dev/null || true

# 3. Verify PyTorch CUDA Availability
python -c "import torch; print('PyTorch CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 4. Run Distributed/GPU Training Pipeline
python train_hpc.py \
    --epochs 120 \
    --batch-size 64 \
    --lr 0.003 \
    --weight-decay 1e-4 \
    --hidden-dim 128 \
    --dropout 0.25 \
    --patience 20 \
    --device cuda \
    --output-dir hpc_output

echo "=========================================================="
echo "HPC Training Job Completed at: $(date)"
echo "Artifacts saved in: hpc_output/"
echo "=========================================================="
