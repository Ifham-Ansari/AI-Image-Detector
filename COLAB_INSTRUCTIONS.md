# 🚀 Google Colab Setup Instructions

This guide helps you run the AI Image Detector training on Google Colab with GPU acceleration.

## Quick Start (3 Steps)

### Step 1: Push to GitHub
```bash
# On your local machine
git add .
git commit -m "Initial commit: AI Image Detector project"
git push origin main
```

### Step 2: Open Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com)
2. Click `File` → `Open notebook` → `GitHub` tab
3. Search for your repository: `YOUR_USERNAME/ai-image-detector`
4. Open `Colab_Training_Setup.ipynb`

### Step 3: Configure & Run
1. Update `GITHUB_URL` in **Section 2** cell with your repository URL
2. Run cells sequentially from top to bottom
3. Monitor GPU usage and training progress

---

## Detailed Steps

### Prerequisites
- ✅ GitHub account with repository uploaded
- ✅ Project size: ~120 MB (including data and code)
- ✅ All files committed to GitHub

### File Structure on GitHub
```
ai-image-detector/
├── .gitignore                 (excludes large files)
├── config.py                  (Colab-compatible)
├── app.py
├── requirements.txt
├── Colab_Training_Setup.ipynb (main notebook)
├── data/
│   ├── train/FAKE, REAL/
│   ├── valid/FAKE, REAL/
│   └── test/FAKE, REAL/
├── src/
│   ├── train.py
│   ├── model.py
│   ├── dataset.py
│   └── ...
└── outputs/
    └── checkpoints/
```

### What the Notebook Does

| Section | What It Does | Time |
|---------|-------------|------|
| 1. Install Libraries | Installs PyTorch, timm, gradcam, etc. | ~2-3 min |
| 2. Clone Repository | Clones your GitHub repo to Colab | ~30 sec |
| 3. Mount Drive (Optional) | Mounts Google Drive if needed | ~30 sec |
| 4. Verify Paths | Checks data directories exist | ~10 sec |
| 5. Train | Runs training script with GPU | 1-2 hours |
| 6. Save Checkpoints | Saves trained model to Drive | ~1 min |

---

## Important Notes

### 🔑 Key Differences from Local Training

**config.py automatically detects Colab** and:
- ✅ Sets `PROJECT_ROOT` to `/content/ai-image-detector`
- ✅ Mounts Google Drive if code requests it
- ✅ Uses GPU automatically (`cuda` if available)

**No code changes needed!** The same `config.py` works locally AND on Colab.

### 💾 Data Size
- Total project size: ~120 MB
- GitHub stores entire dataset (no need for separate upload)
- Colab downloads everything in Section 2

### ⏱️ Training Time
- **GPU (T4)**: ~1-2 hours for 12 epochs
- **GPU (A100)**: ~30-45 minutes for 12 epochs
- **CPU**: ❌ Not recommended (very slow)

### 📊 GPU Selection
When opening Colab:
1. Click `Runtime` → `Change runtime type`
2. Select GPU: Prefer A100 > V100 > T4
3. Standard free tier usually gets T4

---

## Troubleshooting

### ❌ "Repository not found"
- Check `GITHUB_URL` is correct
- Make sure repository is **public** OR use GitHub token for private repos

### ❌ "No space left on device"
- Colab free tier has ~78 GB available
- Your project + data + checkpoints = ~5-10 GB (fine)
- If error occurs, delete old outputs: `rm -rf /content/ai-image-detector/outputs/*`

### ❌ "Out of memory"
- Reduce batch size in `config.py`: `BATCH_SIZE = 8` (instead of 16)
- Run one epoch at a time

### ❌ "Module not found: src"
- Make sure you `cd /content/ai-image-detector` before running training
- Notebook does this automatically in Section 2

---

## Saving Results

### Option A: Download Checkpoints (Manual)
1. After training completes
2. Click Files icon (left sidebar) → `ai-image-detector/outputs/checkpoints`
3. Right-click → Download

### Option B: Auto-save to Google Drive (Optional)
Uncomment Section 6 and run:
```python
import shutil
shutil.copytree(
    CHECKPOINT_DIR,
    Path('/content/drive/MyDrive/ai-image-detector-checkpoints'),
    dirs_exist_ok=True
)
```

---

## Advanced Tips

### Resume Training from Checkpoint
In the notebook, add before Section 5:
```python
import torch
checkpoint = torch.load(CHECKPOINT_DIR / "best_model.pth")
# Load into model and resume training
```

### Monitor Training in Real-time
TensorBoard can be added to notebook:
```python
%load_ext tensorboard
%tensorboard --logdir {OUTPUT_DIR / "logs"}
```

### Use Private GitHub Repo
Generate GitHub token and clone with:
```bash
!git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/ai-image-detector.git
```

---

## What's Colab-Compatible in This Project?

✅ **config.py**: Auto-detects Colab environment  
✅ **train.py**: Runs unchanged (GPU auto-detected)  
✅ **dataset.py**: Reads from any path  
✅ **.gitignore**: Excludes unnecessary files  

---

## Session Management

### Session Timeout
- Free tier Colab sessions timeout after **12 hours**
- Runtime auto-disconnects after **30 minutes of inactivity**

### Save Progress
- Checkpoints are saved to `outputs/checkpoints/` during training
- Download regularly to local machine

---

## Next Steps

1. ✅ Push this project to GitHub
2. ✅ Open `Colab_Training_Setup.ipynb` in Colab
3. ✅ Run cells sequentially
4. ✅ Monitor training (GPU usage visible in upper-left)
5. ✅ Download trained models

**Happy training! 🚀**

---

## Support

For issues:
- Check GPU availability: `!nvidia-smi`
- Verify data: `!ls -la /content/ai-image-detector/data/train/`
- Check config: Run Section 4 "Verify Dataset and Model Paths"
