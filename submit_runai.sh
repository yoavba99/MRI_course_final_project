#!/bin/bash
# Submit MRI Final Project training to RunAI

JOB_NAME="mri-final-train-$(date +%s)"
IMAGE="registry.bgu.ac.il/hpc/jupyter-notebook:latest"

# Default resources
GPUS=1
CPU=2
MEM="16G"

echo "Submitting job: $JOB_NAME"
echo "Image: $IMAGE"
echo "Resources: $GPUS GPU, $CPU CPUs, $MEM Memory"

runai training submit \
  --name $JOB_NAME \
  -i $IMAGE \
  -g $GPUS \
  --cpu $CPU \
  --memory $MEM \
  --working-dir "/gpfs0/tamyr/users/beyoav/mri_final_project" \
  --command -- \
  /bin/bash -c "cd /gpfs0/tamyr/users/beyoav/mri_final_project && python src/train.py --out_dir /gpfs0/tamyr/users/beyoav/mri_final_project/results/logs"

echo "Job submitted. Monitor status with: runai list jobs"
echo "Logs: runai logs $JOB_NAME -f"
