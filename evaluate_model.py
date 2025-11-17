"""
Model Evaluation Script - Generate Confusion Matrix
Evaluates Roboflow model performance on test dataset
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from inference import get_model
from roboflow import Roboflow
from sklearn.metrics import confusion_matrix, classification_report
from dotenv import load_dotenv
import json
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

load_dotenv()

# Configuration
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "drone-and-bird-detection-kewte/1"
WORKSPACE = "priaansh"
PROJECT = "drone-and-bird-detection-kewte"
VERSION = 1
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

class ModelEvaluator:
    """Evaluate Roboflow model and generate confusion matrix"""
    
    def __init__(self, api_key: str, model_id: str, conf_threshold: float = 0.4):
        self.api_key = api_key
        self.model_id = model_id
        self.conf_threshold = conf_threshold
        
        # Initialize Roboflow
        os.environ["ROBOFLOW_API_KEY"] = api_key
        self.rf = Roboflow(api_key=api_key)
        self.model = get_model(model_id=model_id)
        
        # Get dataset
        print(f"🔧 Loading project: {WORKSPACE}/{PROJECT}")
        self.project = self.rf.workspace(WORKSPACE).project(PROJECT)
        self.dataset = self.project.version(VERSION)
        
        # Class mapping
        self.classes = []
        self.class_to_idx = {}
        
    def download_test_data(self, output_dir: str = "test_dataset"):
        """Download test dataset from Roboflow"""
        output_path = Path(output_dir)
        
        if output_path.exists():
            print(f"✓ Test dataset already exists at: {output_path}")
            return str(output_path)
        
        print(f"📥 Downloading test dataset...")
        
        # Try YOLOv12 first (your model), then fall back to other YOLO formats
        # All YOLO formats use the same annotation structure
        formats_to_try = ["yolov12", "yolov11", "yolov8", "yolov5"]
        
        for fmt in formats_to_try:
            try:
                print(f"   Attempting format: {fmt}...")
                dataset_obj = self.dataset.download(fmt, location=output_dir)
                # The actual path is the location parameter we provided
                actual_path = Path(output_dir) / self.project.name
                print(f"✓ Successfully downloaded using format: {fmt}")
                print(f"✓ Dataset location: {actual_path}")
                return str(actual_path)
            except Exception as e:
                print(f"   ✗ Format '{fmt}' failed: {str(e)[:50]}...")
                continue
        
        raise Exception("❌ Could not download dataset - no compatible YOLO format found")
    
    def load_ground_truth(self, dataset_path: str):
        """Load ground truth annotations from test set"""
        test_images_dir = Path(dataset_path) / "test" / "images"
        test_labels_dir = Path(dataset_path) / "test" / "labels"
        
        if not test_images_dir.exists():
            raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")
        
        # Load data.yaml to get class names
        data_yaml = Path(dataset_path) / "data.yaml"
        if data_yaml.exists():
            import yaml
            with open(data_yaml, 'r') as f:
                data = yaml.safe_load(f)
                self.classes = data.get('names', [])
                self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        
        print(f"✓ Classes: {self.classes}")
        
        ground_truth = {}
        image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        
        print(f"📋 Loading ground truth for {len(image_files)} images...")
        
        for img_path in tqdm(image_files, desc="Loading annotations"):
            label_path = test_labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            # Parse YOLO format annotations
            annotations = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        annotations.append({
                            'class_id': class_id,
                            'class_name': self.classes[class_id],
                            'bbox': [x_center, y_center, width, height]
                        })
            
            ground_truth[str(img_path)] = annotations
        
        print(f"✓ Loaded {len(ground_truth)} annotated images")
        return ground_truth
    
    def run_predictions(self, image_paths: list):
        """Run model predictions on all test images"""
        predictions = {}
        
        print(f"🔍 Running predictions on {len(image_paths)} images...")
        
        for img_path in tqdm(image_paths, desc="Predicting"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            results = self.model.infer(img, confidence=self.conf_threshold)[0]
            
            preds = []
            for pred in results.predictions:
                preds.append({
                    'class_name': pred.class_name,
                    'confidence': pred.confidence,
                    'bbox': [pred.x, pred.y, pred.width, pred.height]
                })
            
            predictions[str(img_path)] = preds
        
        print(f"✓ Completed predictions")
        return predictions
    
    def compute_iou(self, box1, box2):
        """Compute IoU between two boxes (YOLO format: x_center, y_center, w, h)"""
        # Convert to corner format
        x1_min = box1[0] - box1[2]/2
        y1_min = box1[1] - box1[3]/2
        x1_max = box1[0] + box1[2]/2
        y1_max = box1[1] + box1[3]/2
        
        x2_min = box2[0] - box2[2]/2
        y2_min = box2[1] - box2[3]/2
        x2_max = box2[0] + box2[2]/2
        y2_max = box2[1] + box2[3]/2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        # Union
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def match_predictions_to_ground_truth(self, ground_truth, predictions, iou_threshold=0.5):
        """Match predictions to ground truth using IoU"""
        y_true = []
        y_pred = []
        
        for img_path, gt_annotations in ground_truth.items():
            pred_annotations = predictions.get(img_path, [])
            
            matched_gt = set()
            matched_pred = set()
            
            # Match predictions to ground truth
            for pred_idx, pred in enumerate(pred_annotations):
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gt_annotations):
                    if gt_idx in matched_gt:
                        continue
                    
                    # Normalize pred bbox to YOLO format (0-1 scale)
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    
                    pred_norm = [
                        pred['bbox'][0] / w,
                        pred['bbox'][1] / h,
                        pred['bbox'][2] / w,
                        pred['bbox'][3] / h
                    ]
                    
                    iou = self.compute_iou(gt['bbox'], pred_norm)
                    
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_gt_idx >= 0:
                    # Matched prediction
                    matched_gt.add(best_gt_idx)
                    matched_pred.add(pred_idx)
                    y_true.append(gt_annotations[best_gt_idx]['class_name'])
                    y_pred.append(pred['class_name'])
                else:
                    # False positive (background class)
                    y_true.append('background')
                    y_pred.append(pred['class_name'])
            
            # Add false negatives (missed detections)
            for gt_idx, gt in enumerate(gt_annotations):
                if gt_idx not in matched_gt:
                    y_true.append(gt['class_name'])
                    y_pred.append('background')
        
        return y_true, y_pred
    
    def generate_confusion_matrix(self, y_true, y_pred, output_dir="evaluation_results"):
        """Generate and save confusion matrix visualization"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get unique classes
        all_classes = sorted(list(set(y_true + y_pred)))
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=all_classes)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=all_classes, yticklabels=all_classes,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Roboflow Model Evaluation', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save
        cm_path = output_path / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {cm_path}")
        plt.close()
        
        # Generate classification report
        report = classification_report(y_true, y_pred, labels=all_classes, 
                                      target_names=all_classes, zero_division=0)
        
        report_path = output_path / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write("Classification Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(report)
        
        print(f"✓ Classification report saved to: {report_path}")
        print(f"\n{report}")
        
        # Save raw confusion matrix
        cm_json_path = output_path / "confusion_matrix.json"
        with open(cm_json_path, 'w') as f:
            json.dump({
                'classes': all_classes,
                'confusion_matrix': cm.tolist(),
                'total_samples': len(y_true)
            }, f, indent=2)
        
        print(f"✓ Raw data saved to: {cm_json_path}")
        
        return cm, all_classes
    
    def evaluate(self, dataset_path: str = None):
        """Full evaluation pipeline"""
        print("\n" + "="*80)
        print("ROBOFLOW MODEL EVALUATION - CONFUSION MATRIX GENERATION")
        print("="*80 + "\n")
        
        # Download test data if needed
        if dataset_path is None:
            dataset_path = self.download_test_data()
        
        # Load ground truth
        ground_truth = self.load_ground_truth(dataset_path)
        
        # Get test image paths
        test_images_dir = Path(dataset_path) / "test" / "images"
        test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
        
        # Run predictions
        predictions = self.run_predictions(test_images)
        
        # Match predictions to ground truth
        print(f"\n🔗 Matching predictions to ground truth (IoU threshold: {IOU_THRESHOLD})...")
        y_true, y_pred = self.match_predictions_to_ground_truth(
            ground_truth, predictions, iou_threshold=IOU_THRESHOLD
        )
        
        print(f"✓ Matched {len(y_true)} detections")
        
        # Generate confusion matrix
        print(f"\n📊 Generating confusion matrix...")
        cm, classes = self.generate_confusion_matrix(y_true, y_pred)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        
        return cm, classes, y_true, y_pred

def main():
    """Main evaluation script"""
    evaluator = ModelEvaluator(
        api_key=ROBOFLOW_API_KEY,
        model_id=MODEL_ID,
        conf_threshold=CONFIDENCE_THRESHOLD
    )
    
    # Run evaluation
    cm, classes, y_true, y_pred = evaluator.evaluate()
    
    print(f"\n✅ Results saved to: evaluation_results/")
    print(f"   - confusion_matrix.png")
    print(f"   - classification_report.txt")
    print(f"   - confusion_matrix.json")

if __name__ == "__main__":
    main()
