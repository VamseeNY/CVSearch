#!/usr/bin/env python3
"""
YOLO Detection Module
Detects persons in video frames
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pickle
import argparse
import os


class PersonDetector:
    """YOLO-based person detector"""
    
    def __init__(self, model_name='yolo11n.pt', conf_threshold=0.75):
        """
        Initialize YOLO detector
        
        Args:
            model_name: YOLO model to use
            conf_threshold: Detection confidence threshold
        """
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        print(f"YOLO loaded with confidence threshold: {conf_threshold}")
    
    def detect_persons(self, frame):
        """
        Detect persons in a frame
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            list: Detections with format [{'bbox': [x1,y1,x2,y2], 'conf': float}]
        """
        results = self.model(frame, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Class 0 = person
                    if int(box.cls) == 0 and float(box.conf) > self.conf_threshold:
                        bbox = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf)
                        
                        detections.append({
                            'bbox': bbox,
                            'conf': conf
                        })
        
        return detections


def detect_video(video_path, camera_id, output_dir, conf_threshold=0.75):
    """
    Run detection on entire video and save results
    
    Args:
        video_path: Path to input video
        camera_id: Camera identifier
        output_dir: Directory to save detection results
        conf_threshold: Detection confidence threshold
    
    Returns:
        dict: Detection results
    """
    
    print(f"Processing Camera {camera_id}: {video_path}")
    
    # Setup output directory
    camera_dir = f"{output_dir}/camera_{camera_id}"
    detections_dir = f"{camera_dir}/detections"
    os.makedirs(detections_dir, exist_ok=True)
    
    # Initialize detector
    detector = PersonDetector(conf_threshold=conf_threshold)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Storage for all detections
    all_detections = {}
    frame_idx = 0
    total_detections = 0
    
    print("Running detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect persons
        detections = detector.detect_persons(frame)
        all_detections[frame_idx] = detections
        total_detections += len(detections)
        
        # Progress update
        if frame_idx % 100 == 0 and frame_idx > 0:
            progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            avg_detections = total_detections / frame_idx if frame_idx > 0 else 0
            print(f"Frame {frame_idx}/{total_frames} ({progress:.1f}%) - "
                  f"Avg detections: {avg_detections:.2f}")
        
        frame_idx += 1
    
    cap.release()
    
    # Prepare results
    results = {
        'camera_id': camera_id,
        'video_path': video_path,
        'total_frames': frame_idx,
        'total_detections': total_detections,
        'avg_detections_per_frame': total_detections / frame_idx if frame_idx > 0 else 0,
        'fps': fps,
        'resolution': (width, height),
        'detections': all_detections
    }
    
    # Save detections
    detections_file = f"{detections_dir}/detections.pkl"
    print(f"Saving detections to {detections_file}")
    with open(detections_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n Detection complete for Camera {camera_id}")
    print(f"Processed {frame_idx} frames")
    print(f"Total person detections: {total_detections}")
    print(f"Average detections per frame: {results['avg_detections_per_frame']:.2f}")
    
    return results


def main():
    """Standalone execution"""
    parser = argparse.ArgumentParser(description='YOLO person detection for video')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--camera_id', type=int, required=True, help='Camera ID')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    parser.add_argument('--conf_threshold', type=float, default=0.75, 
                       help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    results = detect_video(
        args.video,
        args.camera_id,
        args.output_dir,
        args.conf_threshold
    )
    
    print(f"\n Results saved to {args.output_dir}/camera_{args.camera_id}/detections/")


if __name__ == '__main__':
    main()
