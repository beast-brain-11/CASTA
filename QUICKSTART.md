# 🚀 CASTA Quick Start Guide

## ✅ Correct Command Syntax

### Camera (Webcam) Detection
```bash
# With Gemini AI enrichment
python app.py --source 0 --use-gemini

# Without Gemini (faster)
python app.py --source 0 --no-gemini

# Save recording
python app.py --source 0 --use-gemini --save-camera
```

### Video File Detection
```bash
python app.py --source video.mp4 --show
```

### Image Detection
```bash
python app.py --source image.jpg
```

## ⚠️ Common Mistakes

### ❌ WRONG (double dash before source):
```bash
python app.py -- source 0 --use-gemini
```

### ✅ CORRECT (single dash):
```bash
python app.py --source 0 --use-gemini
```

## 🎮 GPU Verification

Your RTX 4060 Laptop GPU is detected! ✅

The app will automatically use your GPU for:
- YOLO object detection (via Roboflow inference)
- Faster frame processing
- Real-time video analysis

Check GPU usage with:
```bash
# During app execution, open another terminal:
nvidia-smi
```

## 🔑 Environment Setup Reminder

Make sure your `.env` file exists with:
```env
ROBOFLOW_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
```

## 🎬 Quick Test Commands

```bash
# Test camera (no recording)
python app.py --source 0 --use-gemini

# Test camera with recording
python app.py --source 0 --use-gemini --save-camera

# Press 'q' to quit
# Press 's' to take screenshot (camera mode)
```

## 📊 Performance Tips

**For best performance:**
- Use `--no-gemini` for faster processing (no AI analysis)
- Lower `--conf` threshold if too many false detections
- Your RTX 4060 should handle 20-30 FPS easily

**Monitor GPU usage:**
```bash
# In another terminal
watch -n 1 nvidia-smi
```

## 🐛 Troubleshooting

**Problem:** `error: the following arguments are required: --source`
**Solution:** Use single dash `--source` not `-- source`

**Problem:** Not using GPU
**Solution:** Check CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

**Problem:** Camera not found
**Solution:** Try different camera IDs: `--source 1` or `--source 2`
