# 🚁 CASTA

**AI-Driven Cognitive Aerial Spatio-Temporal Analysis**

*AI-Driven Cognitive Spatio-Temporal Analysis of Aerial Threats*

Combines Roboflow YOLOv12 object detection with Google Gemini Vision AI for intelligent drone threat assessment.

---

## 🎯 Features

### Core Capabilities
- ✅ **Real-time Object Detection** - Roboflow YOLOv12 (drone/bird classification)
- ✅ **Multi-Object Tracking** - Centroid-based tracker with persistent IDs
- ✅ **Behavioral Analysis** - Velocity, acceleration, trajectory, loitering detection
- ✅ **Threat Scoring** - Rule-based 0-100% scoring with 5 threat levels
- ✅ **AI Semantic Enrichment** - Gemini Vision analyzes HIGH/CRITICAL threats for payloads
- ✅ **Visual Overlays** - Color-coded boxes, trajectory lines, threat indicators
- ✅ **Structured Logging** - JSON output with complete metadata

### Threat Indicators
- 🔴 **APPROACHING** - Bounding box growing (getting closer)
- 🔴 **FAST** - High velocity movement
- 🔴 **DIRECT** - Straight-line trajectory (not random)
- 🔴 **LOITERING** - Hovering/stationary (reconnaissance pattern)
- 🔴 **ACCEL_SURGE** - Sudden acceleration (attack initiation)
- 🔴 **PAYLOAD_DETECTED** - Gemini identifies attachments/explosives

### Threat Levels
| Level | Color | Score | Description |
|-------|-------|-------|-------------|
| **CRITICAL** | Red | 75-100 | Multiple high-risk indicators, potential kamikaze |
| **HIGH** | Orange | 60-74 | Aggressive behavior, approaching fast |
| **MEDIUM** | Yellow | 40-59 | Suspicious patterns detected |
| **LOW** | Green | 20-39 | Minor indicators present |
| **MINIMAL** | White | 0-19 | Normal flight pattern |

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/beast-brain-11/CASTA.git
cd CASTA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
# Copy .env.example to .env
cp .env.example .env

# 4. Add your API keys to .env file
# Edit .env and add:
# ROBOFLOW_API_KEY=your_roboflow_key_here
# GEMINI_API_KEY=your_gemini_key_here
```

**Get API Keys:**
- **Roboflow API Key**: Sign up at [Roboflow](https://app.roboflow.com/) → Settings → API Keys
- **Gemini API Key**: Get from [Google AI Studio](https://ai.google.dev/)

### Usage

**Image Analysis:**
```bash
python app.py --source image.jpg
```

**Video Analysis:**
```bash
python app.py --source video.mp4 --show
```

**Live Camera Feed:**
```bash
python app.py --source 0 --use-gemini
```

**Fast Mode (No Gemini):**
```bash
python app.py --source video.mp4 --no-gemini
```

### Command Line Options

```
--source         Input: image path, video path, or camera ID (0, 1, 2...)
--conf 0.4       Confidence threshold (default: 0.4)
--output path    Custom output path (default: auto-generated)
--show           Display video/camera in real-time window
--save-camera    Save camera recording to file
--use-gemini     Enable Gemini Vision (default: ON)
--no-gemini      Disable Gemini (tracking only, faster)
```

---

## 📊 How It Works

### Pipeline Flow

```
Frame Input
    ↓
[1] Roboflow YOLO Detection
    ↓ (bounding boxes + confidence)
[2] Centroid Tracker
    ↓ (object IDs + history)
[3] Feature Extraction
    ↓ (velocity, acceleration, size growth, path straightness, loitering)
[4] Threat Scoring
    ↓ (0-100% score + CRITICAL/HIGH/MEDIUM/LOW/MINIMAL level)
[5] Gemini Enrichment (if HIGH/CRITICAL)
    ↓ (semantic tags: payload, bird verification, context)
[6] Visual Annotation
    ↓ (color boxes, trajectory, labels, indicators, score bars)
Output: Annotated Media + JSON Metadata
```

### Spatio-Temporal Features

**Spatial:**
- Bounding box size change rate (approaching detection)
- Distance to frame center
- Object area trajectory

**Temporal:**
- Velocity (pixels/frame)
- Acceleration (velocity change)
- Frame-by-frame position delta

**Combined:**
- Path straightness (zigzag vs direct)
- Loitering detection (low movement over time)
- Size growth correlation with velocity

### Gemini Vision Integration

For detections with **HIGH** or **CRITICAL** threat scores:
1. Crop drone region with 20% padding
2. Send to Gemini with context-aware prompt
3. Receive semantic analysis (2-3 sentences)
4. Extract tags: `PAYLOAD_VISIBLE`, `CONFIRMED_DRONE`, `LIKELY_BIRD`, etc.
5. Adjust threat score (+20 for payload, -30 for bird misclassification)
6. Display Gemini description on frame

---

## 📁 Output

All results saved to `threat_analysis_results/`:

**Annotated Media:**
- `hybrid_image_TIMESTAMP.jpg` - Processed images
- `hybrid_video_TIMESTAMP.mp4` - Processed videos
- `hybrid_camera_TIMESTAMP.mp4` - Camera recordings

**Metadata JSON:**
- Complete detection data per frame
- Threat scores and levels
- Behavioral features (velocity, acceleration, etc.)
- Gemini analysis results
- Trajectory coordinates
- Session statistics

**Example JSON Structure:**
```json
{
  "source": "video.mp4",
  "total_frames": 1250,
  "stats": {
    "total_detections": 342,
    "threat_counts": {
      "CRITICAL": 12,
      "HIGH": 28,
      "MEDIUM": 45,
      "LOW": 89,
      "MINIMAL": 168
    },
    "gemini_calls": 40
  },
  "frame_log": [
    {
      "frame": 150,
      "detections": [
        {
          "object_id": 5,
          "class_name": "drone",
          "yolo_confidence": 0.89,
          "threat_score": 82.5,
          "threat_level": "CRITICAL",
          "indicators": ["APPROACHING", "FAST", "DIRECT"],
          "features": {
            "velocity": 67.3,
            "acceleration": 23.1,
            "size_growth_rate": 2.1,
            "path_straightness": 0.91
          },
          "gemini_analysis": {
            "description": "Confirmed drone with visible attachment on undercarriage...",
            "semantic_tags": ["CONFIRMED_DRONE", "PAYLOAD_VISIBLE"]
          }
        }
      ]
    }
  ]
}
```

---

## ⚙️ Configuration

### API Keys
**Important:** The API keys are stored in a `.env` file (not tracked by Git).

**Setup:**
1. Copy `.env.example` to `.env`
2. Add your API keys to `.env`:
```env
ROBOFLOW_API_KEY=your_roboflow_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

**Security Note:** 
- Never commit `.env` to Git
- The `.gitignore` file already excludes `.env`
- Share `.env.example` (template) with your team, not `.env` (actual keys)

### Threat Thresholds
Edit in `app.py`:
```python
THREAT_THRESHOLDS = {
    'velocity': 50,           # pixels/frame
    'acceleration': 20,       # pixels/frame²
    'size_growth_rate': 1.5,  # growth factor
    'path_straightness': 0.8, # 0-1
    'loiter_frames': 30,      # frames
    'loiter_threshold': 5     # pixels
}
```

### Models
- **YOLO Model**: `drone-and-bird-detection-kewte/1` (Roboflow)
- **Gemini Model**: `gemini-flash-latest` (fast, cost-efficient)

---

## 🎮 Interactive Controls

### During Video/Camera Processing:
- **'q'** - Quit and save
- **'s'** - Take screenshot (camera mode)

### Visual Indicators:
- **Colored Boxes** - Threat level coding
- **Trajectory Lines** - Movement path history
- **Score Bar** - Horizontal bar under each detection
- **Labels** - ID | Class | Confidence | Threat | Score | Indicators
- **Frame Stats** - Top-left overlay with counts
- **Gemini Text** - Description below high-threat detections

---

## 🧪 Example Workflows

### 1. Analyze Test Footage
```bash
python app.py --source test_drone.mp4 --show
```

### 2. Live Monitoring (Camera)
```bash
python app.py --source 0 --save-camera
# Press 's' for screenshots, 'q' to finish
```

### 3. Batch Processing (No Display)
```bash
python app.py --source video1.mp4
python app.py --source video2.mp4
python app.py --source video3.mp4
# Check threat_analysis_results/ for outputs
```

### 4. Fast Mode for Long Videos
```bash
python app.py --source long_video.mp4 --no-gemini --conf 0.5
```

---

## 📈 Performance

**Processing Speed:**
- Image: ~0.1-0.3s per image (with Gemini)
- Video: ~15-25 FPS on GPU (NVIDIA RTX 4060)
- Camera: Real-time 20-30 FPS

**Gemini API:**
- Only called for HIGH/CRITICAL threats (selective)
- Typical: 2-5 calls per 100 frames in normal scenarios
- Max rate: ~40 calls/minute (API limits apply)

---

## 🛠️ Technical Stack

- **Detection**: Roboflow YOLOv12 (object detection API)
- **Tracking**: Custom centroid tracker with history buffers
- **AI Vision**: Google Gemini Flash (multimodal LLM)
- **Computer Vision**: OpenCV, Supervision
- **Deep Learning**: PyTorch (YOLO backend)
- **Language**: Python 3.8+

---

## 🎯 Use Cases

1. **Perimeter Security** - Detect approaching drones at restricted facilities
2. **Event Monitoring** - Track aerial threats during public gatherings
3. **Critical Infrastructure** - Airports, power plants, government buildings
4. **Research & Development** - Study drone behavior patterns
5. **Counter-UAS Training** - Simulator for threat recognition drills

---

## 📝 Notes

- **Gemini Enrichment**: Currently optional; can run pure YOLO+tracking mode
- **Tracking**: Simple centroid method (good for stationary camera scenarios)
- **Payload Detection**: Relies on Gemini vision; future: train dedicated payload model
- **Threat Scoring**: Rule-based; can extend with ML classifier trained on labeled data

---

## 🚀 Future Enhancements

- [ ] Multi-camera fusion
- [ ] Dedicated payload detection model (Stage 2)
- [ ] ML-based threat classifier (replace rules)
- [ ] Alert system (webhooks, notifications)
- [ ] Web dashboard (FastAPI + React)
- [ ] Drone trajectory prediction
- [ ] Integration with radar/RF sensors

---

## 📄 License

This is a research/educational project. 

**Dataset Credits:**
- Roboflow Model: `drone-and-bird-detection-kewte/1`

---

## 🤝 Support

For issues or questions, check:
- JSON logs in `threat_analysis_results/`
- Console output for debugging
- Adjust `--conf` threshold if too many/few detections

---

**Project:** CASTA (Cognitive Aerial Spatio-Temporal Analysis)  
**Built for:** Counter-UAS (C-UAS) Research  
**Purpose:** AI-Driven Cognitive Spatio-Temporal Analysis of Aerial Threats
