# 🚀 Deployment Guide — GitHub + Hugging Face Spaces

This guide walks you through pushing your project to GitHub and deploying a live, mobile-accessible app on **Hugging Face Spaces** — completely free.

---

## PART 1 — Push to GitHub

### Step 1: Initialize Git in your project folder

Open a terminal inside `C:\Users\vishv\Desktop\breast-cancer-detection\` and run:

```bash
git init
git add .
git commit -m "Initial commit: MammoScan AI breast cancer detection"
```

### Step 2: Create a new GitHub repository

1. Go to → **https://github.com/paradise-007**
2. Click the green **"New"** button (top-right)
3. Name it: `breast-cancer-detection`
4. Set to **Public** (required for free HuggingFace deployment)
5. **Do NOT** tick "Add README" (you already have one)
6. Click **"Create repository"**

### Step 3: Connect and push

GitHub will show you these commands — paste them in your terminal:

```bash
git remote add origin https://github.com/paradise-007/breast-cancer-detection.git
git branch -M main
git push -u origin main
```

✅ Your code is now live on GitHub.

---

## PART 2 — Deploy to Hugging Face Spaces (Free Live App)

Hugging Face Spaces gives you a **free public URL** that works on mobile, tablet, and desktop.

### Step 1: Create a Hugging Face account

→ **https://huggingface.co/join** (free, takes 1 minute)

### Step 2: Create a new Space

1. Go to → **https://huggingface.co/new-space**
2. Fill in:
   - **Space name**: `mammoscan-ai`
   - **License**: MIT
   - **SDK**: Select **Streamlit**
   - **Hardware**: CPU Basic (free)
   - **Visibility**: Public
3. Click **"Create Space"**

### Step 3: Push your code to the Space

Hugging Face Spaces are Git repositories. Run these commands:

```bash
# Clone your new Space (replace YOUR_HF_USERNAME)
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/mammoscan-ai

# Copy all project files into the cloned folder
# (or just add the HF remote to your existing repo)
cd breast-cancer-detection
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/mammoscan-ai
git push hf main
```

**Or use the drag-and-drop uploader** (easiest for beginners):
1. Open your Space on HuggingFace
2. Click **"Files"** tab → **"Add file"** → **"Upload files"**
3. Drag all your project files in
4. Click **"Commit changes"**

### Step 4: Wait for the build

HuggingFace will automatically:
- Install `requirements.txt`
- Run `streamlit run streamlit_app.py`
- Give you a public URL like: `https://YOUR_HF_USERNAME-mammoscan-ai.hf.space`

Build takes **2–5 minutes** the first time.

### Step 5: Share the link

Your app is now live! The URL works on:
- 📱 iPhone / Android (mobile browser)
- 💻 Desktop / laptop
- 🖥 Any device with internet

---

## PART 3 — Add Trained Model Weights (Optional)

If you've run `train.py` and have `.keras` files, upload them so the app runs in **live mode** instead of demo mode.

### Option A: Git LFS (for large files)

```bash
git lfs install
git lfs track "*.keras"
git add .gitattributes
git add models/saved/*.keras models/saved/eval_metrics.json
git commit -m "Add trained model weights"
git push hf main
```

### Option B: Upload via HuggingFace UI

1. Go to your Space → **Files** tab
2. Navigate to `models/saved/`
3. Upload `VGG16.keras`, `ResNet50V2.keras`, `InceptionV3.keras`, `eval_metrics.json`

---

## Summary

| Step | What | Where |
|------|------|--------|
| 1 | Source code + version control | GitHub: `github.com/paradise-007/breast-cancer-detection` |
| 2 | Live web app (free, mobile) | HuggingFace: `YOUR_HF_USERNAME-mammoscan-ai.hf.space` |
| 3 | Model weights (optional) | HuggingFace Files tab or Git LFS |

---

## Troubleshooting

**Build fails on HuggingFace?**
- Check the **"Logs"** tab in your Space for the error
- Most common cause: a package in `requirements.txt` that needs a specific version

**App too slow on free CPU?**
- TensorFlow inference on CPU is slow for large models
- Consider using `tensorflow-cpu` (already set in requirements.txt)
- Or upgrade to a paid GPU Space on HuggingFace

**Models not loading?**
- Make sure weight files are named exactly: `VGG16.keras`, `ResNet50V2.keras`, `InceptionV3.keras`
- Place them in `models/saved/`
- The app will run in demo mode if they're missing (that's OK for demos!)
