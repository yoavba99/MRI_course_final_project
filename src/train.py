# Yoav Ben Ari - 207178039
# Nitzan Elad - 211563176

import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import os

from dataset import MRIDataModule
from model import SimpleViT3D

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="/gpfs0/tamyr/users/beyoav/mri_final_project/metadata/student_train_metadata.csv")
    parser.add_argument("--valid_csv", type=str, default="/gpfs0/tamyr/users/beyoav/mri_final_project/metadata/student_val_metadata.csv")
    parser.add_argument("--base_path", type=str, default="/gpfs0/tamyr/projects/data/brain_age/npy/")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--out_dir", type=str, default="../results/logs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Initializing DataModule...")
    datamodule = MRIDataModule(
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        base_path=args.base_path,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    print("Initializing Model...")
    model = SimpleViT3D(learning_rate=args.learning_rate)
    
    os.makedirs(args.out_dir, exist_ok=True)
    logger = WandbLogger(save_dir=args.out_dir, project="mri_final_project", name="vit_training")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, "checkpoints"),
        filename="best-vit-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    
    import torch
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )
    
    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)
    print(f"Training complete! Best checkpoint saved to: {checkpoint_callback.best_model_path}")
    print("You can now pass this checkpoint path to the --ckpt argument in evaluate.py")

if __name__ == "__main__":
    main()
