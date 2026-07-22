#!/bin/bash
# Script to submit the evaluation job to RunAI

JOB_NAME="mri-final-eval"

# Delete the previous job if it exists so we can submit a fresh one
runai-bgu delete $JOB_NAME
sleep 5

# Submit the evaluate job
runai-bgu submit python \
  -n $JOB_NAME \
  -c 4 -m 32Gi -g 1 --large-shm \
  --conda brain-age \
  -- "python /gpfs0/tamyr/users/beyoav/mri_final_project/src/evaluate.py --out_dir /gpfs0/tamyr/users/beyoav/mri_final_project/results/logs --ckpt /gpfs0/tamyr/users/beyoav/mri_final_project/results/logs/checkpoints/best-vit-epoch=49-val_loss=1.69.ckpt"

echo "Evaluation job submitted. Monitor status with: runai-bgu list"
echo "Follow logs with: runai-bgu logs $JOB_NAME -f"
