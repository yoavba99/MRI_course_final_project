# Yoav Ben Ari - 207178039
# Nitzan Elad - 211563176

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm
from dataset import MRIDataModule
from baseline import BaselineModel, evaluate_baseline
from model import SimpleViT3D
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="Path to your trained model checkpoint")
    parser.add_argument("--train_csv", type=str, default="/gpfs0/tamyr/users/beyoav/mri_final_project/metadata/student_train_metadata.csv")
    parser.add_argument("--valid_csv", type=str, default="/gpfs0/tamyr/users/beyoav/mri_final_project/metadata/student_val_metadata.csv")
    parser.add_argument("--test_csv", type=str, default="/gpfs0/tamyr/users/beyoav/mri_final_project/metadata/student_test_metadata.csv")
    parser.add_argument("--base_path", type=str, default="/gpfs0/tamyr/projects/data/brain_age/npy/")
    parser.add_argument("--out_dir", type=str, default="../results")
    return parser.parse_args()

def plot_scatter(y_true, y_pred, model_name, mae, pearson_r, out_path):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='k')
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    
    plt.xlabel("True Age (Years)")
    plt.ylabel("Predicted Age (Years)")
    plt.title(f"{model_name}. MAE: {mae:.2f} Years, Pearson r: {pearson_r:.2f}")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_qualitative_image(image_tensor, true_age, baseline_pred, vit_pred, case_name, out_path):
    """
    Save the middle slice of a 3D MRI scan along with true/predicted labels.
    """
    img = image_tensor.squeeze().cpu().numpy()
    
    # Take middle axial slice
    mid_slice = img[:, :, img.shape[2] // 2]
    mid_slice = np.rot90(mid_slice)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mid_slice, cmap='gray')
    plt.title(f"{case_name}\nTrue: {true_age:.1f} | Base: {baseline_pred:.1f} | ViT: {vit_pred:.1f}")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    datamodule = MRIDataModule(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        test_csv=args.test_csv,
        base_path=args.base_path,
        batch_size=8,
        num_workers=4
    )
    datamodule.setup()
    
    train_loader = datamodule.train_dataloader()
    test_loader = datamodule.test_dataloader()

    # 2. Baseline Model
    print("Evaluating Baseline Model...")
    baseline_model = BaselineModel()
    baseline_model.fit(train_loader)
    base_mae, base_r, base_true, base_pred = evaluate_baseline(baseline_model, test_loader)
    
    print(f"Baseline MAE: {base_mae:.2f}, Pearson r: {base_r:.2f}")
    plot_scatter(base_true, base_pred, "Baseline Model", base_mae, base_r, os.path.join(args.out_dir, "scatter_baseline.png"))

    # 3. ViT Model
    print("Evaluating ViT Model...")
    if not args.ckpt or not os.path.exists(args.ckpt):
        print(f"Notice: No valid checkpoint path provided for ViT ({args.ckpt}).")
        print("Skipping ViT evaluation. Baseline evaluation completed successfully.")
        return

    try:
        vit_model = SimpleViT3D.load_from_checkpoint(args.ckpt, map_location=device)
    except Exception as e:
        print(f"Error loading checkpoint with load_from_checkpoint: {e}")
        print("Initializing model and attempting to load state dict manually...")
        vit_model = SimpleViT3D()
        state_dict = torch.load(args.ckpt, map_location=device)
        vit_model.load_state_dict(state_dict['state_dict'] if 'state_dict' in state_dict else state_dict, strict=False)
        
    vit_model.eval()
    
    vit_true = []
    vit_pred = []
    
    all_images = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc="ViT Inference"):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            logits = vit_model(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=1)
            probs = torch.exp(log_probs)
            
            est_age = (probs * vit_model.bin_centres.to(device)).sum(dim=1)
            
            vit_true.extend(labels.cpu().flatten().numpy().tolist())
            vit_pred.extend(est_age.cpu().flatten().numpy().tolist())
            
            for i in range(images.size(0)):
                all_images.append(images[i].cpu())
                
    vit_true = np.array(vit_true)
    vit_pred = np.array(vit_pred)
    
    vit_mae = mean_absolute_error(vit_true, vit_pred)
    vit_r, _ = pearsonr(vit_true, vit_pred)
    
    print(f"ViT MAE: {vit_mae:.2f}, Pearson r: {vit_r:.2f}")
    plot_scatter(vit_true, vit_pred, "Our Model (ViT)", vit_mae, vit_r, os.path.join(args.out_dir, "scatter_vit.png"))
    
    # 4. Qualitative Analysis
    print("Extracting qualitative cases...")
    base_errors = np.abs(base_true - base_pred)
    vit_errors = np.abs(vit_true - vit_pred)
    
    # Define thresholds
    GOOD_THRESH = 5.0
    POOR_THRESH = 15.0
    
    case1_idx = np.where((base_errors < GOOD_THRESH) & (vit_errors < GOOD_THRESH))[0]
    case2_idx = np.where((base_errors > POOR_THRESH) & (vit_errors > POOR_THRESH))[0]
    case3_idx = np.where((base_errors < GOOD_THRESH) & (vit_errors > POOR_THRESH))[0]
    case4_idx = np.where((base_errors > POOR_THRESH) & (vit_errors < GOOD_THRESH))[0]
    
    cases_dir = os.path.join(args.out_dir, "qualitative_cases")
    os.makedirs(cases_dir, exist_ok=True)
    
    def get_best_idx(indices, preference="min_sum"):
        if len(indices) == 0:
            return None
        if preference == "min_sum":
            return indices[np.argmin(base_errors[indices] + vit_errors[indices])]
        elif preference == "max_sum":
            return indices[np.argmax(base_errors[indices] + vit_errors[indices])]
        elif preference == "base_good":
            return indices[np.argmin(base_errors[indices] - vit_errors[indices])]
        elif preference == "vit_good":
            return indices[np.argmin(vit_errors[indices] - base_errors[indices])]
    
    cases = {
        "Both_Models_Performed_Well": get_best_idx(case1_idx, "min_sum"),
        "Both_Models_Performed_Poorly": get_best_idx(case2_idx, "max_sum"),
        "Baseline_Well_ViT_Poor": get_best_idx(case3_idx, "base_good"),
        "ViT_Well_Baseline_Poor": get_best_idx(case4_idx, "vit_good")
    }
    
    for case_name, idx in cases.items():
        if idx is not None:
            save_qualitative_image(
                all_images[idx], 
                base_true[idx], 
                base_pred[idx], 
                vit_pred[idx], 
                case_name.replace("_", " "), 
                os.path.join(cases_dir, f"{case_name}.png")
            )
            print(f"Saved {case_name}.png")
        else:
            print(f"Could not find a suitable candidate for {case_name}")

    print("Evaluation complete.")

if __name__ == "__main__":
    main()
