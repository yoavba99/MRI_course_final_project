# Yoav Ben Ari - 207178039
# Nitzan Elad - 211563176

import os
import numpy as np
import torch
import pandas as pd
import logging
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import monai.transforms as transforms

def get_transforms(train=True):
    """
    Returns the MONAI composition of transforms.
    Train: Random rotation (+-5deg), Random shift (+-5), Gaussian Noise.
    Test: None.
    """
    if not train:
        return None 
        
    rotate_range_in_radians = 5 * (np.pi / 180)
    
    return transforms.Compose([
        transforms.RandRotate(range_x=rotate_range_in_radians,
                                    range_y=rotate_range_in_radians,
                                    range_z=rotate_range_in_radians,
                                    prob=0.5),
        transforms.RandAffine(translate_range=(5, 5, 5), prob=0.5, padding_mode="zeros"),
        transforms.RandShiftIntensity(offsets=5, prob=0.5),
        transforms.RandScaleIntensity(factors=0.1, prob=0.5),
        transforms.RandGaussianNoise(mean=0.0, std=0.02, prob=0.5),
    ])

class MRIDataset(Dataset):
    def __init__(self, csv_file, base_path, transform=None):
        self.csv_file = csv_file
        self.base_path = base_path
        self.transform = transform 

        # Load CSV
        df = pd.read_csv(self.csv_file)
        
        # Drop rows with missing labels to prevent NaN loss
        if 'Age' in df.columns:
            df = df.dropna(subset=['Age'])
        else:
            df = df.dropna(subset=[df.columns[2]])
        
        # Expecting Subject in column 0, Age in column 2
        if 'Subject' in df.columns:
            self.subjects = df['Subject'].values
        else:
            self.subjects = df.iloc[:, 0].values

        if 'Age' in df.columns:
            self.labels = df['Age'].values.astype(np.float32)
        else:
            self.labels = df.iloc[:, 2].values.astype(np.float32)

        self.subjects = self.subjects
        self.labels = self.labels
        
        valid_subjects = []
        valid_labels = []
        valid_paths = []
        
        for subj, label in zip(self.subjects, self.labels):
            path = os.path.join(self.base_path, f"{subj}.npy")
            if os.path.exists(path):
                valid_subjects.append(subj)
                valid_labels.append(label)
                valid_paths.append(path)
                
        self.subjects = np.array(valid_subjects)
        self.labels = np.array(valid_labels)
        self.data_paths = valid_paths

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        try:
            image = np.load(self.data_paths[idx])
            image = torch.from_numpy(image).float().unsqueeze(0)
            
            if image.max() > 1.0:
                image = image / image.max()
            
            if self.transform:
                image = self.transform(image)
            
            label = torch.tensor([self.labels[idx]], dtype=torch.float32)
            return image, label
            
        except Exception as e:
            logging.error(f"Error loading file {self.data_paths[idx]}: {str(e)}")
            raise e

class MRIDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, valid_csv, test_csv=None, base_path="", 
                 batch_size=64, num_workers=4):
        super().__init__()
        self.train_csv = train_csv
        self.valid_csv = valid_csv
        self.test_csv = test_csv
        self.base_path = base_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_ds = MRIDataset(
                csv_file=self.train_csv,
                base_path=self.base_path,
                transform=get_transforms(train=True)
            )
            self.val_ds = MRIDataset(
                csv_file=self.valid_csv,
                base_path=self.base_path,
                transform=get_transforms(train=False)
            )

        if stage == "test" or stage is None:
            if self.test_csv:
                self.test_ds = MRIDataset(
                    csv_file=self.test_csv,
                    base_path=self.base_path,
                    transform=get_transforms(train=False)
                )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, 
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, 
                          num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        if hasattr(self, 'test_ds'):
            return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, 
                              num_workers=self.num_workers, persistent_workers=True, pin_memory=True)
