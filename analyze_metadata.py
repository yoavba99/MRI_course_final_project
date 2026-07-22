# Yoav Ben Ari - 207178039
# Nitzan Elad - 211563176

import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    local_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(local_dir, "metadata")):
        base_dir = local_dir
    else:
        base_dir = "/gpfs0/tamyr/users/beyoav/mri_final_project"
    
    metadata_dir = os.path.join(base_dir, "metadata")
    out_dir = os.path.join(base_dir, "results", "analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    splits = {
        "Train": "student_train_metadata.csv",
        "Val": "student_val_metadata.csv",
        "Test": "student_test_metadata.csv"
    }
    
    dfs = []
    print("=== Dataset Summary ===")
    for split_name, file_name in splits.items():
        file_path = os.path.join(metadata_dir, file_name)
        df = pd.read_csv(file_path)
        df['Split'] = split_name
        dfs.append(df)
        
        total = len(df)
        missing_age = df['Age'].isna().sum()
        unique_projs = df['ProjName'].nunique()
        mean_age = df['Age'].mean()
        
        print(f"{split_name}: {total} subjects | {missing_age} missing Age | {unique_projs} Projects | Mean Age: {mean_age:.1f}")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Age boxplot
    plt.figure(figsize=(8, 6))
    combined_df.boxplot(column="Age", by="Split", grid=False, figsize=(8,6))
    plt.title("Age Distribution across Splits")
    plt.suptitle("")
    plt.ylabel("Age (Years)")
    plt.savefig(os.path.join(out_dir, "age_distribution_box.png"), dpi=300)
    plt.close('all')
    
    # Age KDE plot
    plt.figure(figsize=(10, 6))
    for split in ["Train", "Val", "Test"]:
        subset = combined_df[combined_df['Split'] == split]['Age'].dropna()
        subset.plot(kind='kde', label=split, linewidth=2)
    plt.title("Age Density across Splits")
    plt.xlabel("Age (Years)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "age_distribution_kde.png"), dpi=300)
    plt.close()
    
    # Project distribution
    plt.figure(figsize=(14, 8))
    proj_counts = pd.crosstab(combined_df['ProjName'], combined_df['Split'])
    proj_counts = proj_counts[['Train', 'Val', 'Test']]
    proj_counts['Total'] = proj_counts.sum(axis=1)
    proj_counts = proj_counts.sort_values(by='Total', ascending=False).drop('Total', axis=1)
    
    proj_counts.plot(kind='bar', figsize=(14, 8), width=0.8)
    plt.title("Project (Data Source) Distribution across Splits")
    plt.xlabel("Project Name")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "project_distribution.png"), dpi=300)
    plt.close('all')
    
    # Project vs split percentage matrix
    crosstab_pct = pd.crosstab(combined_df['ProjName'], combined_df['Split'], normalize='columns') * 100
    crosstab_pct = crosstab_pct[['Train', 'Val', 'Test']]
    crosstab_pct = crosstab_pct.round(1)
    
    csv_path = os.path.join(out_dir, "project_split_percentages.csv")
    crosstab_pct.to_csv(csv_path)
    
    print(f"\nAnalysis complete. Plots and data saved to: {out_dir}")

if __name__ == "__main__":
    main()
