# Yoav Ben Ari - 207178039
# Nitzan Elad - 211563176

import os
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.mixture import GaussianMixture
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
            
            non_zero_img = img[img > 0]
            if len(non_zero_img) > 0:
                mean_intensity = non_zero_img.mean()
                std_intensity = non_zero_img.std()
                max_intensity = non_zero_img.max()
                min_intensity = non_zero_img.min()
            else:
                mean_intensity = std_intensity = max_intensity = min_intensity = 0.0
            

            # GMM features (3 components for CSF, GM, WM)
            data = img[img > 0].reshape(-1, 1)
            
            # Subsample voxels for GMM fitting
            if len(data) > 10000:
                indices = np.random.choice(len(data), 10000, replace=False)
                data_sub = data[indices]
            else:
                data_sub = data
                
            if len(data_sub) > 3:
                gmm = GaussianMixture(n_components=3, random_state=42)
                gmm.fit(data_sub)
                
                # Sort by mean intensity
                order = np.argsort(gmm.means_.flatten())
                gmm_means = gmm.means_.flatten()[order]
                gmm_covs = gmm.covariances_.flatten()[order]
                gmm_weights = gmm.weights_.flatten()[order]
            else:
                gmm_means = np.zeros(3)
                gmm_covs = np.zeros(3)
                gmm_weights = np.zeros(3)
            
            features.append([
                mean_intensity, std_intensity, max_intensity, min_intensity, 
                *gmm_means, *gmm_covs, *gmm_weights
            ])
        
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
