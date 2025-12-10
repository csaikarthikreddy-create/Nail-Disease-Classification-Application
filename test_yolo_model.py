#!/usr/bin/env python3
"""
Test script for YOLO nail detection model
Tests the best.pt model downloaded from the main branch
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.nail_detector import NailDetector

def test_yolo_model(image_path=None):
    """Test the YOLO nail detection model"""
    
    # Initialize detector with the best.pt model
    model_path = 'models/yolo_nail_detector_best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"📦 Loading YOLO model from: {model_path}")
    print(f"   File size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
    
    try:
        detector = NailDetector(model_path=model_path)
        print("✅ YOLO model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Test with a sample image if provided
    if image_path and os.path.exists(image_path):
        print(f"\n🔍 Testing detection on: {image_path}")
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not read image: {image_path}")
                return False
            
            print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
            
            # Run detection
            results = detector.detect_nails(image)
            
            if results is None or len(results['nails']) == 0:
                print("⚠️  No nails detected in the image")
                return True  # Model works, just no detections
            
            print(f"✅ Detected {len(results['nails'])} nail(s):")
            for i, nail in enumerate(results['nails'], 1):
                bbox = nail['bbox']
                conf = nail['confidence']
                print(f"   Nail {i}: bbox=({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}), confidence={conf:.3f}")
            
            # Save visualization
            vis_path = 'test_detection_result.jpg'
            cv2.imwrite(vis_path, results['visualization'])
            print(f"💾 Visualization saved to: {vis_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n✅ Model loaded successfully (no test image provided)")
        print("   To test with an image, run:")
        print(f"   python test_yolo_model.py <image_path>")
        return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test YOLO nail detection model')
    parser.add_argument('image', nargs='?', help='Path to test image (optional)')
    args = parser.parse_args()
    
    success = test_yolo_model(args.image)
    sys.exit(0 if success else 1)


