# Brain Age Prediction - MRI Final Project

This repository contains the codebase for the MRI final project "Brain Age Prediction".

## Overview

The objective of this project is to predict the "brain age" of a patient from their 3D MRI scans, evaluating two models:
1. **Baseline Model**: A non-deep-learning classical approach utilizing linear regression over simple image-derived features (mean intensity, max intensity, approximate brain volume).
2. **Deep Learning Model (ViT)**: A Vision Transformer (`SimpleViT3D`) leveraging MONAI's Transformer backend, custom overlapping patches, stochastic depth regularization, and a KL-Divergence loss to treat regression as a soft-classification problem across discrete age bins.

## Code Structure

- `src/dataset.py`: Defines the `MRIDataset` and Lightning `MRIDataModule`, utilizing MONAI transforms for augmentations on 3D NumPy arrays.
- `src/baseline.py`: Implements the `BaselineModel` using scikit-learn's `LinearRegression` over extracted global image statistics.
- `src/model.py`: Implements the `SimpleViT3D` deep learning model, stripping out unnecessary components to leave a clean, focused model definition based on `BaseBrainAgeModel`.
- `src/evaluate.py`: The main entrypoint for inference. It evaluates both the baseline model and the pre-trained ViT model on the test dataset, generates required scatter plots (True Age vs. Predicted Age), and extracts four qualitative samples for the final report.

## How to Run

1. **Environment Setup**: 
   Ensure you have PyTorch, PyTorch Lightning, MONAI, scikit-learn, and matplotlib installed.
2. **Train the Model**:
   You must train the ViT model from scratch first.
   ```bash
   python src/train.py \
       --train_csv /gpfs0/tamyr/projects/data/brain_age/metadata_age_prediction_train.csv \
       --valid_csv /gpfs0/tamyr/projects/data/brain_age/metadata_age_prediction_valid.csv \
       --base_path /gpfs0/tamyr/projects/data/brain_age/npy/ \
       --out_dir ../results/logs
   ```
   This will output a model checkpoint in the `../results/logs/checkpoints/` directory.
3. **Run Evaluation**:
   After training, pass the generated checkpoint path to the evaluation script.
   ```bash
   python src/evaluate.py \
       --ckpt ../results/logs/checkpoints/<YOUR_BEST_CHECKPOINT>.ckpt \
       --train_csv /gpfs0/tamyr/projects/data/brain_age/metadata_age_prediction_train.csv \
       --valid_csv /gpfs0/tamyr/projects/data/brain_age/metadata_age_prediction_valid.csv \
       --test_csv /gpfs0/tamyr/projects/data/brain_age/metadata_age_prediction_test.csv \
       --base_path /gpfs0/tamyr/projects/data/brain_age/npy/ \
       --out_dir ../results
   ```

## Outputs

After running the evaluation script, the following outputs will be saved in the `results` directory:
- `scatter_baseline.png`: Scatter plot for the baseline model with MAE and Pearson correlation score.
- `scatter_vit.png`: Scatter plot for the ViT model with MAE and Pearson correlation score.
- `qualitative_cases/`: Directory containing mid-slice images showcasing where models performed well or poorly.
