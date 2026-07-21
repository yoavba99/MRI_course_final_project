import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.utils.checkpoint import checkpoint
from monai.networks.nets import ViT
from monai.networks.blocks import TransformerBlock
import logging
import json
import csv
import pathlib
import math

# Define Age Bins for Soft Classification
BIN_WIDTH = 1.0
AGE_MIN, AGE_MAX = 5, 89
BIN_CENTRES = torch.arange(AGE_MIN, AGE_MAX + BIN_WIDTH, BIN_WIDTH)
N_BINS = len(BIN_CENTRES)

def inject_drop_path(model, drop_path_rate=0.1):
    """
    Dynamically injects Drop Path (Stochastic Depth) into MONAI's Transformer Blocks.
    """
    import timm.models.layers as timm_layers
    
    blocks = [module for module in model.modules() if isinstance(module, TransformerBlock)]
    dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(blocks))]
    
    for i, block in enumerate(blocks):
        block.drop_path = timm_layers.DropPath(dpr[i]) if dpr[i] > 0. else nn.Identity()

def load_pretrained_vit_weights(model: nn.Module, pretrained_path: str):
    """
    Robust utility to load pre-trained 3D weights, handling dimension mismatches.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Attempting to load pre-trained weights from {pretrained_path}")
    try:
        checkpoint_data = torch.load(pretrained_path, map_location="cpu")
        state_dict = checkpoint_data.get("state_dict", checkpoint_data)
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in state_dict.items():
            clean_key = k.replace("vit.", "").replace("encoder.", "") 
            if clean_key in model_dict and v.shape == model_dict[clean_key].shape:
                pretrained_dict[clean_key] = v
            else:
                logger.warning(f"Skipping key {k} - Shape mismatch or not found.")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        logger.info(f"Successfully loaded {len(pretrained_dict)} layers from pre-trained weights.")
    except FileNotFoundError:
        logger.error(f"Pre-trained weights file not found at {pretrained_path}. Training from scratch.")

class BaseBrainAgeModel(pl.LightningModule):
    """Base class for shared logic (loss, metrics, optimization)"""
    def __init__(self, learning_rate=1e-4, sigma_bins=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.criterion = nn.KLDivLoss(reduction='batchmean', log_target=False)
        self.sigma = sigma_bins
        self.register_buffer("bin_centres", BIN_CENTRES.float())

    def age_to_soft(self, age_batch, sigma=1.0, eps=1e-8):
        age_batch = age_batch.squeeze(-1).float()
        d2 = (age_batch[:, None] - self.bin_centres[None, :])**2
        p = torch.softmax(-d2 / (2 * sigma**2), dim=-1)
        p = p.clamp_min(eps)
        return p / p.sum(dim=-1, keepdim=True)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        log_probs = F.log_softmax(logits, dim=1)
        soft_labels = self.age_to_soft(y, self.sigma)
        loss = self.criterion(log_probs, soft_labels)
        
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        
        with torch.no_grad():
            probs = torch.exp(log_probs)
            est_age = (probs * self.bin_centres).sum(dim=1)
            mae = torch.abs(est_age - y.squeeze()).mean()
            self.log(f"{stage}_mae", mae, prog_bar=True, sync_dist=True)
            
            if stage == "test":
                self.test_outputs.append({
                    "true_age": y.cpu(),
                    "est_age": est_age.cpu(),
                    "probs": probs.cpu()
                })
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        if not hasattr(self, "test_outputs"):
            self.test_outputs = []
        return self._shared_step(batch, "test")

    def on_test_epoch_end(self):
        log_dir = getattr(self.logger, "log_dir", ".")
        if log_dir is None:
            log_dir = "."
        
        run_name = "vit_final_project"
        
        out_file = pathlib.Path(log_dir) / f"test_results_{run_name}.csv"
        with open(out_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true_age", "est_age", "abs_error", "probs_json"])
            for batch_out in self.test_outputs:
                for i in range(len(batch_out["true_age"])):
                    true_age = float(batch_out["true_age"][i])
                    est_age = float(batch_out["est_age"][i])
                    probs = batch_out["probs"][i].tolist()
                    writer.writerow([true_age, est_age, abs(true_age - est_age), json.dumps(probs)])
        self.test_outputs = []

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=0.1)
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-10
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

class SimpleViT3D(BaseBrainAgeModel):
    def __init__(self, img_size=(96, 96, 96), patch_size=(16, 16, 16), stride=None,
                 hidden_size=1024, mlp_dim=2048, num_layers=8, num_heads=16,
                 dropout_rate=0.2, learning_rate=1e-4, sigma_bins=1.0):
        super().__init__(learning_rate, sigma_bins)
        
        if stride is None:
            stride = patch_size

        # MONAI ViT defaults to spatial_dims=3 based on img_size length
        self.vit = ViT(
            in_channels=1,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            classification=True,
            dropout_rate=dropout_rate,
            num_classes=N_BINS,
            post_activation=None
        )
        
        # Override patch embeddings if stride is different from patch_size
        if tuple(stride) != tuple(patch_size):
            self.vit.patch_embedding.patch_embeddings = nn.Conv3d(
                in_channels=1,
                out_channels=hidden_size,
                kernel_size=patch_size,
                stride=stride
            )
            
            n_patches = math.prod([math.floor((img_size[i] - patch_size[i]) / stride[i]) + 1 for i in range(3)])
            
            self.vit.patch_embedding.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
            nn.init.trunc_normal_(self.vit.patch_embedding.position_embeddings, std=0.02)
            logging.info(f"Overlapping patches applied: patch_size={patch_size}, stride={stride}. Total patches={n_patches}")

        # Inject Drop Path for regularization
        inject_drop_path(self.vit, drop_path_rate=0.2)
        
        # Disable classification flag temporarily so it returns the tuple to checkpoint
        self.vit.classification = False

    def forward(self, x):
        # Use gradient checkpointing to save memory
        out, _ = checkpoint(self.vit, x, use_reentrant=False)
        return out
