"""
CASTA - Cognitive Aerial Spatio-Temporal Analysis
AI-Driven Cognitive Spatio-Temporal Analysis of Aerial Threats

Combines Roboflow YOLOv12 detection + Gemini Vision semantic analysis
with spatio-temporal behavioral tracking and threat scoring
"""

import os
import sys
import cv2
import numpy as np
import supervision as sv
from inference import get_model
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict, deque
import json
import time
from typing import Dict, List, Tuple, Optional
import google.generativeai as genai
from PIL import Image
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Roboflow Configuration
MODEL_ID = "drone-and-bird-detection-kewte/1"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY not found in environment variables. Please check your .env file.")

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
GEMINI_MODEL = "gemini-flash-latest"

# Output Configuration
OUTPUT_DIR = Path("threat_analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Threat Thresholds
THREAT_THRESHOLDS = {
    'velocity': 50,           # pixels/frame for HIGH speed
    'acceleration': 20,       # pixels/frame^2 for sudden changes
    'size_growth_rate': 1.5,  # bbox area growth factor (approaching)
    'path_straightness': 0.8, # 0-1, higher = straighter path
    'loiter_frames': 30,      # frames with low movement = loitering
    'loiter_threshold': 5     # pixel movement threshold for loitering
}

# Color Scheme for Threat Levels
THREAT_COLORS = {
    'CRITICAL': (0, 0, 255),      # Red
    'HIGH': (0, 165, 255),        # Orange
    'MEDIUM': (0, 255, 255),      # Yellow
    'LOW': (0, 255, 0),           # Green
    'MINIMAL': (255, 255, 255)    # White
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CENTROID TRACKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CentroidTracker:
    """Simple centroid-based tracker for drone objects"""
    
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = {}  # {id: centroid}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        # Historical data for each tracked object
        self.history = defaultdict(lambda: {
            'positions': deque(maxlen=60),  # Last 60 positions
            'boxes': deque(maxlen=60),      # Last 60 bounding boxes
            'timestamps': deque(maxlen=60), # Last 60 timestamps
            'class_name': None,
            'confidences': deque(maxlen=60)
        })
    
    def register(self, centroid, bbox, timestamp, class_name, confidence):
        """Register a new object"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        
        # Initialize history
        self.history[self.next_object_id]['positions'].append(centroid)
        self.history[self.next_object_id]['boxes'].append(bbox)
        self.history[self.next_object_id]['timestamps'].append(timestamp)
        self.history[self.next_object_id]['class_name'] = class_name
        self.history[self.next_object_id]['confidences'].append(confidence)
        
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        # Keep history for potential later analysis
    
    def update(self, detections, timestamp):
        """
        Update tracker with new detections
        Returns: dict mapping detection index to object ID
        """
        if len(detections) == 0:
            # Mark all objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        # Extract centroids and boxes from detections
        input_centroids = []
        input_boxes = []
        input_classes = []
        input_confidences = []
        
        for i in range(len(detections)):
            bbox = detections.xyxy[i]
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            input_centroids.append((cx, cy))
            input_boxes.append(bbox)
            
            class_name = detections['class_name'][i] if 'class_name' in detections.data else f"class_{detections.class_id[i]}"
            input_classes.append(class_name)
            input_confidences.append(detections.confidence[i])
        
        # If no objects being tracked, register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_boxes[i], timestamp, input_classes[i], input_confidences[i])
            
            # Return mapping: detection index -> object ID
            return {i: i for i in range(len(input_centroids))}
        
        # Compute distance matrix
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        
        distance_matrix = np.zeros((len(object_centroids), len(input_centroids)))
        for i, obj_centroid in enumerate(object_centroids):
            for j, input_centroid in enumerate(input_centroids):
                distance_matrix[i, j] = np.linalg.norm(
                    np.array(obj_centroid) - np.array(input_centroid)
                )
        
        # Match objects to inputs using minimum distance
        rows = distance_matrix.min(axis=1).argsort()
        cols = distance_matrix.argmin(axis=1)[rows]
        
        used_rows = set()
        used_cols = set()
        detection_to_id = {}
        
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            
            if distance_matrix[row, col] > self.max_distance:
                continue
            
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            
            # Update history
            self.history[object_id]['positions'].append(input_centroids[col])
            self.history[object_id]['boxes'].append(input_boxes[col])
            self.history[object_id]['timestamps'].append(timestamp)
            self.history[object_id]['confidences'].append(input_confidences[col])
            
            used_rows.add(row)
            used_cols.add(col)
            detection_to_id[col] = object_id
        
        # Handle disappeared objects
        unused_rows = set(range(distance_matrix.shape[0])) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
        
        # Register new objects
        unused_cols = set(range(distance_matrix.shape[1])) - used_cols
        for col in unused_cols:
            self.register(input_centroids[col], input_boxes[col], timestamp, input_classes[col], input_confidences[col])
            detection_to_id[col] = self.next_object_id - 1
        
        return detection_to_id

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPATIO-TEMPORAL FEATURE EXTRACTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FeatureExtractor:
    """Extract spatio-temporal features for threat analysis"""
    
    @staticmethod
    def compute_velocity(positions, timestamps):
        """Compute average velocity in pixels/frame"""
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        time_delta = len(positions) - 1
        return total_distance / time_delta if time_delta > 0 else 0.0
    
    @staticmethod
    def compute_acceleration(positions, timestamps):
        """Compute acceleration (change in velocity)"""
        if len(positions) < 3:
            return 0.0
        
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocities.append(np.sqrt(dx**2 + dy**2))
        
        if len(velocities) < 2:
            return 0.0
        
        accelerations = []
        for i in range(1, len(velocities)):
            accelerations.append(abs(velocities[i] - velocities[i-1]))
        
        return np.mean(accelerations) if accelerations else 0.0
    
    @staticmethod
    def compute_size_change_rate(boxes):
        """Compute rate of bounding box size change (approaching indicator)"""
        if len(boxes) < 2:
            return 1.0
        
        areas = []
        for box in boxes:
            width = box[2] - box[0]
            height = box[3] - box[1]
            areas.append(width * height)
        
        if areas[0] == 0:
            return 1.0
        
        return areas[-1] / areas[0]  # Growth factor
    
    @staticmethod
    def compute_path_straightness(positions):
        """Compute path straightness (0-1, higher = straighter)"""
        if len(positions) < 3:
            return 1.0
        
        # Direct distance from start to end
        start = np.array(positions[0])
        end = np.array(positions[-1])
        direct_distance = np.linalg.norm(end - start)
        
        if direct_distance == 0:
            return 0.0
        
        # Actual path distance
        path_distance = 0.0
        for i in range(1, len(positions)):
            p1 = np.array(positions[i-1])
            p2 = np.array(positions[i])
            path_distance += np.linalg.norm(p2 - p1)
        
        if path_distance == 0:
            return 0.0
        
        return direct_distance / path_distance
    
    @staticmethod
    def detect_loitering(positions, threshold=5):
        """Detect if object is loitering (low movement over time)"""
        if len(positions) < 10:
            return False, 0
        
        # Check movement in recent frames
        recent_positions = list(positions)[-30:]  # Last 30 frames
        movements = []
        
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            movements.append(np.sqrt(dx**2 + dy**2))
        
        avg_movement = np.mean(movements) if movements else 0
        loitering = avg_movement < threshold
        loiter_frames = len(recent_positions) if loitering else 0
        
        return loitering, loiter_frames

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# THREAT SCORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ThreatScorer:
    """Compute threat scores based on behavioral indicators"""
    
    def __init__(self, thresholds=THREAT_THRESHOLDS):
        self.thresholds = thresholds
    
    def compute_threat_score(self, features: Dict) -> Tuple[float, str, List[str]]:
        """
        Compute threat score (0-100) and threat level
        Returns: (score, level, indicators)
        """
        score = 0
        indicators = []
        
        # Velocity indicator (0-25 points)
        velocity = features.get('velocity', 0)
        if velocity > self.thresholds['velocity']:
            score += 25
            indicators.append('FAST')
        elif velocity > self.thresholds['velocity'] * 0.5:
            score += 15
            indicators.append('MODERATE_SPEED')
        
        # Acceleration indicator (0-20 points)
        acceleration = features.get('acceleration', 0)
        if acceleration > self.thresholds['acceleration']:
            score += 20
            indicators.append('ACCEL_SURGE')
        
        # Approaching indicator (0-30 points)
        size_growth = features.get('size_growth_rate', 1.0)
        if size_growth > self.thresholds['size_growth_rate']:
            score += 30
            indicators.append('APPROACHING')
        elif size_growth < 0.7:  # Getting smaller = moving away
            score -= 10
        
        # Path straightness (0-15 points)
        straightness = features.get('path_straightness', 0)
        if straightness > self.thresholds['path_straightness']:
            score += 15
            indicators.append('DIRECT')
        
        # Loitering (0-10 points, but indicates reconnaissance)
        is_loitering = features.get('is_loitering', False)
        loiter_frames = features.get('loiter_frames', 0)
        if is_loitering and loiter_frames > self.thresholds['loiter_frames']:
            score += 10
            indicators.append('LOITERING')
        
        # Clamp score to 0-100
        score = max(0, min(100, score))
        
        # Determine threat level
        if score >= 75:
            level = 'CRITICAL'
        elif score >= 60:
            level = 'HIGH'
        elif score >= 40:
            level = 'MEDIUM'
        elif score >= 20:
            level = 'LOW'
        else:
            level = 'MINIMAL'
        
        return score, level, indicators

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GEMINI VISION CLIENT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GeminiClient:
    """Google Gemini Vision API client for semantic analysis"""
    
    def __init__(self, api_key: str, model_name: str = GEMINI_MODEL):
        self.api_key = api_key
        self.model_name = model_name
        self.enabled = False
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.enabled = True
            print(f"✓ Gemini Vision initialized: {model_name}")
        except Exception as e:
            print(f"⚠️  Gemini initialization failed: {e}")
            print("   Continuing without Gemini enrichment")
    
    def analyze_drone(self, image_crop: np.ndarray, class_name: str, threat_level: str) -> Dict:
        """
        Analyze a drone detection using Gemini Vision
        Returns semantic tags and confidence adjustments
        """
        if not self.enabled:
            return {'semantic_tags': [], 'gemini_confidence': None, 'description': None}
        
        try:
            # Convert numpy array to PIL Image
            if len(image_crop.shape) == 2:  # Grayscale
                pil_image = Image.fromarray(image_crop)
            else:  # BGR to RGB
                rgb_crop = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_crop)
            
            # Construct prompt based on threat level
            if threat_level in ['HIGH', 'CRITICAL']:
                prompt = f"""Analyze this aerial object detected as '{class_name}' with threat level {threat_level}.

Focus on:
1. Is this actually a drone or a bird?
2. Are there any visible payloads, attachments, or suspicious features?
3. Any behavioral context visible (aggressive posture, hovering, etc.)?

Provide a concise analysis in 2-3 sentences with specific threat-relevant observations."""
            else:
                prompt = f"""Quickly verify: Is this detected '{class_name}' actually a drone or a bird? Any notable features? Keep response to 1 sentence."""
            
            # Generate response
            response = self.model.generate_content([prompt, pil_image])
            
            # Parse response
            description = response.text.strip() if response.text else "No description"
            
            # Extract semantic tags from description
            semantic_tags = []
            description_lower = description.lower()
            
            if 'payload' in description_lower or 'attachment' in description_lower:
                semantic_tags.append('PAYLOAD_VISIBLE')
            if 'bird' in description_lower:
                semantic_tags.append('LIKELY_BIRD')
            if 'drone' in description_lower and 'not' not in description_lower:
                semantic_tags.append('CONFIRMED_DRONE')
            if 'aggressive' in description_lower or 'threat' in description_lower:
                semantic_tags.append('AGGRESSIVE_BEHAVIOR')
            if 'hovering' in description_lower or 'stationary' in description_lower:
                semantic_tags.append('HOVERING')
            
            return {
                'semantic_tags': semantic_tags,
                'gemini_confidence': 0.9 if 'CONFIRMED_DRONE' in semantic_tags else 0.7,
                'description': description
            }
            
        except Exception as e:
            print(f"   Gemini analysis error: {e}")
            return {'semantic_tags': [], 'gemini_confidence': None, 'description': None}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# HYBRID DETECTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class HybridDetector:
    """Main hybrid detection system combining Roboflow + Gemini + Tracking"""
    
    def __init__(self, 
                 model_id: str = MODEL_ID,
                 roboflow_key: str = ROBOFLOW_API_KEY,
                 gemini_key: str = GEMINI_API_KEY,
                 confidence: float = 0.4,
                 use_gemini: bool = True):
        
        # Setup Roboflow
        os.environ["ROBOFLOW_API_KEY"] = roboflow_key
        print(f"🔧 Loading Roboflow model: {model_id}")
        self.yolo_model = get_model(model_id=model_id)
        self.confidence = confidence
        print(f"✓ Roboflow model loaded")
        
        # Setup Gemini
        self.gemini = GeminiClient(gemini_key) if use_gemini else None
        self.use_gemini = use_gemini and (self.gemini.enabled if self.gemini else False)
        
        # Setup tracker and analyzers
        self.tracker = CentroidTracker(max_disappeared=30)
        self.feature_extractor = FeatureExtractor()
        self.threat_scorer = ThreatScorer()
        
        # Frame counter
        self.frame_count = 0
        
        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'threat_counts': defaultdict(int),
            'gemini_calls': 0
        }
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process a single frame through the full pipeline
        Returns: (annotated_frame, detections_with_metadata)
        """
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        # Step 1: YOLO Detection
        results = self.yolo_model.infer(frame, confidence=self.confidence)[0]
        detections = sv.Detections.from_inference(results)
        
        if len(detections) == 0:
            self.tracker.update(detections, timestamp)
            return frame, []
        
        self.stats['total_detections'] += len(detections)
        
        # Step 2: Track objects
        detection_to_id = self.tracker.update(detections, timestamp)
        
        # Step 3: Compute features and threat scores for each tracked object
        enriched_detections = []
        
        for det_idx, object_id in detection_to_id.items():
            history = self.tracker.history[object_id]
            
            # Extract features
            features = {
                'velocity': self.feature_extractor.compute_velocity(
                    history['positions'], history['timestamps']
                ),
                'acceleration': self.feature_extractor.compute_acceleration(
                    history['positions'], history['timestamps']
                ),
                'size_growth_rate': self.feature_extractor.compute_size_change_rate(
                    history['boxes']
                ),
                'path_straightness': self.feature_extractor.compute_path_straightness(
                    history['positions']
                ),
            }
            
            is_loitering, loiter_frames = self.feature_extractor.detect_loitering(
                history['positions'], threshold=THREAT_THRESHOLDS['loiter_threshold']
            )
            features['is_loitering'] = is_loitering
            features['loiter_frames'] = loiter_frames
            
            # Compute threat score
            threat_score, threat_level, indicators = self.threat_scorer.compute_threat_score(features)
            self.stats['threat_counts'][threat_level] += 1
            
            # Get detection data
            bbox = detections.xyxy[det_idx]
            class_name = detections['class_name'][det_idx] if 'class_name' in detections.data else f"class_{detections.class_id[det_idx]}"
            yolo_confidence = detections.confidence[det_idx]
            
            # Step 4: Gemini enrichment (for HIGH/CRITICAL threats)
            gemini_result = None
            if self.use_gemini and threat_level in ['HIGH', 'CRITICAL']:
                # Crop detection region with padding
                x1, y1, x2, y2 = map(int, bbox)
                h, w = frame.shape[:2]
                
                # Add 20% padding
                pad_w = int((x2 - x1) * 0.2)
                pad_h = int((y2 - y1) * 0.2)
                
                x1_pad = max(0, x1 - pad_w)
                y1_pad = max(0, y1 - pad_h)
                x2_pad = min(w, x2 + pad_w)
                y2_pad = min(h, y2 + pad_h)
                
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if crop.size > 0:
                    gemini_result = self.gemini.analyze_drone(crop, class_name, threat_level)
                    self.stats['gemini_calls'] += 1
                    
                    # Adjust threat based on Gemini tags
                    if 'PAYLOAD_VISIBLE' in gemini_result.get('semantic_tags', []):
                        threat_score = min(100, threat_score + 20)
                        threat_level = 'CRITICAL'
                        indicators.append('PAYLOAD_DETECTED')
                    
                    if 'LIKELY_BIRD' in gemini_result.get('semantic_tags', []):
                        threat_score = max(0, threat_score - 30)
                        if threat_score < 20:
                            threat_level = 'MINIMAL'
            
            # Package detection metadata
            detection_meta = {
                'object_id': object_id,
                'bbox': bbox.tolist(),
                'class_name': class_name,
                'yolo_confidence': float(yolo_confidence),
                'threat_score': threat_score,
                'threat_level': threat_level,
                'indicators': indicators,
                'features': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                            for k, v in features.items()},
                'trajectory': [pos for pos in history['positions']],
                'gemini_analysis': gemini_result
            }
            
            enriched_detections.append(detection_meta)
        
        # Step 5: Annotate frame
        annotated_frame = self.annotate_frame(frame, enriched_detections)
        
        return annotated_frame, enriched_detections
    
    def annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw annotations on frame with threat visualization"""
        annotated = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            threat_level = det['threat_level']
            color = THREAT_COLORS[threat_level]
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            
            # Draw trajectory
            trajectory = det['trajectory']
            if len(trajectory) > 1:
                points = np.array(trajectory, dtype=np.int32)
                cv2.polylines(annotated, [points], False, color, 2)
            
            # Create label
            label_parts = [
                f"ID:{det['object_id']}",
                det['class_name'],
                f"{det['yolo_confidence']:.2f}",
                f"{threat_level}",
                f"{det['threat_score']:.0f}%"
            ]
            
            if det['indicators']:
                label_parts.append(f"[{','.join(det['indicators'][:2])}]")
            
            label = " | ".join(label_parts)
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated, 
                (x1, y1 - label_h - 10), 
                (x1 + label_w, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (0, 0, 0), 
                2
            )
            
            # Draw threat score bar
            bar_width = x2 - x1
            bar_height = 8
            score_width = int(bar_width * (det['threat_score'] / 100))
            
            cv2.rectangle(
                annotated,
                (x1, y2 + 5),
                (x1 + bar_width, y2 + 5 + bar_height),
                (50, 50, 50),
                -1
            )
            cv2.rectangle(
                annotated,
                (x1, y2 + 5),
                (x1 + score_width, y2 + 5 + bar_height),
                color,
                -1
            )
            
            # Add Gemini description if available
            if det.get('gemini_analysis') and det['gemini_analysis'].get('description'):
                desc = det['gemini_analysis']['description']
                if len(desc) > 50:
                    desc = desc[:47] + "..."
                
                cv2.putText(
                    annotated,
                    f"Gemini: {desc}",
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1
                )
        
        # Add frame stats overlay
        stats_y = 30
        cv2.putText(
            annotated,
            f"Frame: {self.frame_count} | Detections: {len(detections)}",
            (10, stats_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        stats_y += 30
        threat_summary = " | ".join([f"{level}: {self.stats['threat_counts'][level]}" 
                                     for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']])
        cv2.putText(
            annotated,
            threat_summary,
            (10, stats_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        if self.use_gemini:
            stats_y += 25
            cv2.putText(
                annotated,
                f"Gemini Calls: {self.stats['gemini_calls']}",
                (10, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 255, 100),
                1
            )
        
        return annotated
    
    def run_image(self, image_path: str, output_path: str = None):
        """Process a single image"""
        print(f"\n{'='*80}")
        print(f"PROCESSING IMAGE: {image_path}")
        print(f"{'='*80}\n")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not read image")
            return
        
        # Process
        annotated, detections = self.process_frame(image, time.time())
        
        # Save
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"hybrid_image_{timestamp}.jpg"
        
        cv2.imwrite(str(output_path), annotated)
        print(f"✓ Saved to: {output_path}")
        
        # Save JSON
        json_path = Path(str(output_path).replace('.jpg', '.json'))
        with open(json_path, 'w') as f:
            json.dump({
                'source': image_path,
                'timestamp': datetime.now().isoformat(),
                'detections': detections,
                'stats': dict(self.stats)
            }, f, indent=2)
        
        print(f"✓ Metadata saved to: {json_path}")
        
        # Display
        print(f"\n📊 Results: {len(detections)} detections")
        for det in detections:
            print(f"   ID {det['object_id']}: {det['class_name']} | "
                  f"Threat: {det['threat_level']} ({det['threat_score']:.0f}%) | "
                  f"Indicators: {', '.join(det['indicators']) if det['indicators'] else 'None'}")
        
        cv2.imshow('Hybrid Threat Analysis', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run_video(self, video_path: str, output_path: str = None, show: bool = True):
        """Process a video file"""
        print(f"\n{'='*80}")
        print(f"PROCESSING VIDEO: {video_path}")
        print(f"{'='*80}\n")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video")
            return
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✓ Video: {width}x{height} @ {fps} FPS ({total_frames} frames)")
        
        # Setup output
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"hybrid_video_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # JSON log
        json_log = []
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = time.time()
                annotated, detections = self.process_frame(frame, timestamp)
                
                out.write(annotated)
                
                if detections:
                    json_log.append({
                        'frame': self.frame_count,
                        'timestamp': timestamp,
                        'detections': detections
                    })
                
                if show:
                    cv2.imshow('Hybrid Threat Analysis', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n⚠️  Stopped by user")
                        break
                
                if self.frame_count % 30 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    print(f"   Progress: {self.frame_count}/{total_frames} ({progress:.1f}%)")
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        print(f"\n✓ Processed {self.frame_count} frames")
        print(f"✓ Video saved to: {output_path}")
        
        # Save JSON log
        json_path = Path(str(output_path).replace('.mp4', '.json'))
        with open(json_path, 'w') as f:
            json.dump({
                'source': video_path,
                'total_frames': self.frame_count,
                'stats': dict(self.stats),
                'frame_log': json_log
            }, f, indent=2)
        
        print(f"✓ JSON log saved to: {json_path}")
        self.print_final_stats()
    
    def run_camera(self, camera_id: int = 0, save_output: bool = False):
        """Run real-time camera detection"""
        print(f"\n{'='*80}")
        print(f"CAMERA DETECTION: Camera {camera_id}")
        print(f"{'='*80}\n")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"✓ Camera: {width}x{height} @ {fps} FPS")
        print(f"✓ Press 'q' to quit, 's' to screenshot")
        
        out = None
        if save_output:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = OUTPUT_DIR / f"hybrid_camera_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"✓ Recording to: {output_path}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = time.time()
                annotated, detections = self.process_frame(frame, timestamp)
                
                if out is not None:
                    out.write(annotated)
                
                cv2.imshow('Hybrid Threat Analysis - Camera', annotated)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⚠️  Stopped by user")
                    break
                elif key == ord('s'):
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = OUTPUT_DIR / f"screenshot_{timestamp_str}.jpg"
                    cv2.imwrite(str(screenshot_path), annotated)
                    print(f"📸 Screenshot: {screenshot_path}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
        
        print(f"\n✓ Processed {self.frame_count} frames")
        self.print_final_stats()
    
    def print_final_stats(self):
        """Print final statistics"""
        print(f"\n{'='*80}")
        print("FINAL STATISTICS")
        print(f"{'='*80}")
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"Total Detections: {self.stats['total_detections']}")
        print(f"\nThreat Distribution:")
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
            count = self.stats['threat_counts'][level]
            if count > 0:
                print(f"  {level}: {count}")
        if self.use_gemini:
            print(f"\nGemini API Calls: {self.stats['gemini_calls']}")
        print(f"{'='*80}\n")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN CLI
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args():
    parser = argparse.ArgumentParser(
        description="CASTA - Cognitive Aerial Spatio-Temporal Analysis (Roboflow + Gemini)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Image detection
  python app.py --source image.jpg
  
  # Video with Gemini enrichment
  python app.py --source video.mp4 --use-gemini
  
  # Real-time camera
  python app.py --source 0 --use-gemini
  
  # Disable Gemini (YOLO + tracking only)
  python app.py --source video.mp4 --no-gemini
        """
    )
    
    parser.add_argument('--source', type=str, required=True,
                       help='Input: image, video path, or camera ID (0, 1, etc.)')
    parser.add_argument('--conf', type=float, default=0.4,
                       help='YOLO confidence threshold (default: 0.4)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: auto in threat_analysis_results/)')
    parser.add_argument('--show', action='store_true',
                       help='Display output in real-time')
    parser.add_argument('--save-camera', action='store_true',
                       help='Save camera recording')
    parser.add_argument('--use-gemini', action='store_true', default=True,
                       help='Enable Gemini Vision enrichment (default: enabled)')
    parser.add_argument('--no-gemini', action='store_true',
                       help='Disable Gemini (YOLO + tracking only)')
    
    return parser.parse_args()

def main():
    print("\n╔" + "═"*78 + "╗")
    print("║" + " "*33 + "CASTA v1.0" + " "*35 + "║")
    print("║" + " "*15 + "Cognitive Aerial Spatio-Temporal Analysis" + " "*22 + "║")
    print("║" + " "*20 + "Roboflow YOLO + Gemini Vision" + " "*29 + "║")
    print("╚" + "═"*78 + "╝\n")
    
    args = parse_args()
    
    # Determine Gemini usage
    use_gemini = args.use_gemini and not args.no_gemini
    
    # Initialize detector
    detector = HybridDetector(
        model_id=MODEL_ID,
        roboflow_key=ROBOFLOW_API_KEY,
        gemini_key=GEMINI_API_KEY,
        confidence=args.conf,
        use_gemini=use_gemini
    )
    
    source = args.source
    
    # Route to appropriate handler
    if source.isdigit():
        # Camera
        detector.run_camera(int(source), save_output=args.save_camera)
    elif source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
        # Video
        if not Path(source).exists():
            print(f"❌ Error: Video file not found")
            return
        detector.run_video(source, output_path=args.output, show=args.show)
    elif source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
        # Image
        if not Path(source).exists():
            print(f"❌ Error: Image file not found")
            return
        detector.run_image(source, output_path=args.output)
    else:
        print(f"❌ Error: Unsupported source format")
        print("Supported: .jpg, .png (images), .mp4, .avi (videos), 0/1/2 (camera)")
        return
    
    print("\n✅ Analysis complete!")
    print(f"📁 Results in: {OUTPUT_DIR.absolute()}\n")

if __name__ == "__main__":
    main()
