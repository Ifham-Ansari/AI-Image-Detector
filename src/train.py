import logging
from pathlib import Path
import sys
import random
import os
import traceback

# Suppress HuggingFace Hub symlink warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add project root to sys.path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from config import (
    CHECKPOINT_DIR,
    DEVICE,
    DROPOUT,
    EPOCHS,
    LEARNING_RATE,
    MODEL_NAME,
    PATIENCE,
    PRETRAINED,
    SEED,
    WEIGHT_DECAY,
    OUTPUT_DIR,
)
from src.dataset import load_data
from src.evaluate import calculate_metrics
from src.model import EfficientNetFFTClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(OUTPUT_DIR / "train.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _get_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1]
    return batch

def train_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Train", leave=False)
    
    try:
        for i, batch in enumerate(progress_bar):
            inputs, targets = _get_batch(batch)
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Use Mixed Precision (AMP)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Scale loss and step optimizer
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            
            if i % 10 == 0:
                progress_bar.set_postfix(batch=i, loss=loss.item())
                
    except Exception as e:
        logger.error(f"Error during training epoch: {str(e)}")
        logger.error(traceback.format_exc())
        raise e
        
    return running_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for batch in tqdm(dataloader, desc="Valid", leave=False):
                inputs, targets = _get_batch(batch)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)

def save_checkpoint(state: dict, filename: Path) -> None:
    torch.save(state, filename)

def train_model():
    try:
        set_seed()
        logger.info(f"Starting training on device: {DEVICE}")
        
        loaders = load_data()
        for split, loader in loaders.items():
            logger.info(f"Dataset {split} size: {len(loader.dataset)} images ({len(loader)} batches)")
            
        logger.info(f"Initializing model: {MODEL_NAME}")
        model = EfficientNetFFTClassifier(
            model_name=MODEL_NAME,
            pretrained=PRETRAINED,
            num_classes=2,
            dropout=DROPOUT,
        ).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
        
        # Initialize Gradient Scaler for AMP
        scaler = torch.cuda.amp.GradScaler()

        best_auc = 0.0
        patience = PATIENCE
        best_path = CHECKPOINT_DIR / "best_model.pth"
        last_path = CHECKPOINT_DIR / "last_model.pth"

        for epoch in range(1, EPOCHS + 1):
            logger.info(f"Epoch {epoch}/{EPOCHS}")
            
            train_loss = train_epoch(model, loaders["train"], criterion, optimizer, DEVICE, scaler)
            valid_loss = validate_epoch(model, loaders["valid"], criterion, DEVICE)
            valid_metrics = calculate_metrics(model, loaders["valid"], DEVICE)
            scheduler.step()

            logger.info(
                f"Epoch {epoch} results: train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
                f"valid_acc={valid_metrics['accuracy']:.4f} valid_auc={valid_metrics['roc_auc']:.4f}"
            )

            if valid_metrics["roc_auc"] > best_auc:
                best_auc = valid_metrics["roc_auc"]
                save_checkpoint(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_auc": best_auc,
                    },
                    best_path,
                )
                patience = PATIENCE
                logger.info(f"New best model saved with AUC: {best_auc:.4f}")
            else:
                patience -= 1
                if patience <= 0:
                    logger.info("Early stopping triggered")
                    break

            save_checkpoint(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_auc": best_auc,
                },
                last_path,
            )

        logger.info(f"Training complete. Best ROC-AUC: {best_auc:.4f}")
        
    except Exception as e:
        logger.error(f"Critical error in train_model: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    train_model()
