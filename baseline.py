import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import torch
import tqdm

class BaselineModel:
    def __init__(self):
        self.model = LinearRegression()
        
    def _extract_features(self, images):
        """
        Extract simple image-derived features for the baseline model.
        images: tensor of shape (N, C, D, H, W)
        Returns: numpy array of shape (N, num_features)
        """
        features = []
        for i in range(images.size(0)):
            img = images[i].numpy()
            mean_intensity = img.mean()
            std_intensity = img.std()
            max_intensity = img.max()
            min_intensity = img.min()
            
            # Simple threshold-based volume estimation
            brain_volume = (img > 0.1).sum()
            
            features.append([mean_intensity, std_intensity, max_intensity, min_intensity, brain_volume])
        
        return np.array(features)

    def fit(self, dataloader):
        """
        Fit the linear regression model on the training set.
        """
        print("Extracting features from training set for baseline model...")
        X, y = [], []
        for batch in tqdm.tqdm(dataloader):
            images, labels = batch
            feats = self._extract_features(images)
            X.append(feats)
            y.append(labels.numpy())
            
        X = np.concatenate(X, axis=0)
        y = np.concatenate(y, axis=0).flatten()
        
        print("Fitting Linear Regression model...")
        self.model.fit(X, y)
        print("Baseline model fitted successfully.")

    def predict(self, dataloader):
        """
        Predict on a dataset and return true ages and predicted ages.
        """
        X, y_true = [], []
        for batch in dataloader:
            images, labels = batch
            feats = self._extract_features(images)
            X.append(feats)
            y_true.append(labels.numpy())
            
        X = np.concatenate(X, axis=0)
        y_true = np.concatenate(y_true, axis=0).flatten()
        
        y_pred = self.model.predict(X)
        return y_true, y_pred

def evaluate_baseline(model, dataloader):
    y_true, y_pred = model.predict(dataloader)
    
    mae = mean_absolute_error(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)
    
    return mae, pearson_r, y_true, y_pred
