#!/usr/bin/env python3
"""
Tracking Module
Uses DeepSORT with Torchreid ReID features for person tracking
"""

import cv2
import numpy as np
import torch
import torchreid
from deep_sort_realtime.deepsort_tracker import DeepSort
import pickle
import argparse
import os
from collections import defaultdict


class PersonTracker:
    """DeepSORT tracker with Torchreid features"""
    
    def __init__(self, config=None):
        """
        Initialize tracker with Torchreid feature extractor
        
        Args:
            config: Configuration dict for tracker parameters
        """
        if config is None:
            config = {}
        
        print("Initializing tracking system...")
        
        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️ Using device: {self.device}")
        
        # Load Torchreid feature extractor
        print("Loading Torchreid OSNet feature extractor...")
        self.feature_extractor = torchreid.utils.FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='',
            device=self.device
        )
        print("Torchreid loaded")
        
        # Initialize DeepSORT
        print("Initializing DeepSORT tracker...")
        max_age = config.get('max_age', 50)
        n_init = config.get('n_init', 3)
        max_cosine_distance = config.get('max_cosine_distance', 0.3)
        nn_budget = config.get('nn_budget', 100)
        
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder='mobilenet',
            half=False,
            bgr=True
        )
        
        print(f"DeepSORT initialized (max_age={max_age}, n_init={n_init}, "
              f"max_cosine_distance={max_cosine_distance})")
    
    def extract_features(self, frame, detections):
        """
        Extract Torchreid features for detections
        
        Args:
            frame: Input frame (BGR)
            detections: List of detections with bbox
        
        Returns:
            list: Feature vectors (512-dim) for each detection
        """
        if len(detections) == 0:
            return []
        
        features = []
        h, w = frame.shape[:2]
        
        for det in detections:
            bbox = det['bbox'].astype(int)
            x1, y1, x2, y2 = bbox
            
            # Ensure bbox within frame
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 > x1 and y2 > y1:
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    try:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        feat = self.feature_extractor([crop_rgb])[0].cpu().numpy()
                        features.append(feat)
                    except Exception as e:
                        features.append(np.zeros(512))
                else:
                    features.append(np.zeros(512))
            else:
                features.append(np.zeros(512))
        
        return features
    
    def convert_to_deepsort_format(self, detections):
        """Convert detections to DeepSORT format: ([x, y, w, h], confidence)"""
        if len(detections) == 0:
            return []
        
        deepsort_detections = []
        for det in detections:
            bbox = det['bbox']
            conf = det['conf']
            
            # Convert [x1,y1,x2,y2] to [x1,y1,w,h]
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            deepsort_detections.append(([x1, y1, w, h], conf))
        
        return deepsort_detections
    
    def update(self, frame, detections):
        """
        Update tracker with new detections
        
        Args:
            frame: Current frame
            detections: List of detections
        
        Returns:
            list: Confirmed tracks with format [{'track_id': int, 'bbox': [x1,y1,x2,y2]}]
        """
        if len(detections) == 0:
            # Update with empty detections to age existing tracks
            tracks = self.tracker.update_tracks([], frame=frame)
            return []
        
        # Extract features
        features = self.extract_features(frame, detections)
        
        # Convert to DeepSORT format
        deepsort_detections = self.convert_to_deepsort_format(detections)
        
        # Update tracker
        tracks = self.tracker.update_tracks(
            raw_detections=deepsort_detections,
            embeds=features,
            frame=frame
        )
        
        # Extract confirmed tracks
        confirmed_tracks = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltwh = track.to_ltwh()
            x1, y1, w, h = ltwh
            x2, y2 = x1 + w, y1 + h
            
            confirmed_tracks.append({
                'track_id': track_id,
                'bbox': [x1, y1, x2, y2],
                'ltwh': ltwh
            })
        
        return confirmed_tracks


def track_video(video_path, camera_id, output_dir, detections_file, config=None):
    """
    Run tracking on video using pre-computed detections
    
    Args:
        video_path: Path to input video
        camera_id: Camera identifier
        output_dir: Output directory
        detections_file: Path to detections pickle file
        config: Tracker configuration
    
    Returns:
        dict: Tracking results
    """
    
    print(f"Tracking Camera {camera_id}: {video_path}")
    
    # Load detections
    print(f"Loading detections from {detections_file}")
    with open(detections_file, 'rb') as f:
        detection_data = pickle.load(f)
    
    all_detections = detection_data['detections']
    fps = detection_data['fps']
    width, height = detection_data['resolution']
    
    print(f"Loaded detections for {len(all_detections)} frames")
    
    # Setup directories
    camera_dir = f"{output_dir}/camera_{camera_id}"
    crops_dir = f"{camera_dir}/crops"
    os.makedirs(crops_dir, exist_ok=True)
    
    # Initialize tracker
    tracker = PersonTracker(config)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = f"{camera_dir}/tracked_video.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Storage for tracking results
    all_tracks = {}
    track_data = {}
    track_embeddings = defaultdict(list)
    
    frame_idx = 0
    
    print("Running tracking...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections for this frame
        detections = all_detections.get(frame_idx, [])
        
        # Update tracker
        tracks = tracker.update(frame, detections)
        
        # Process tracks
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            x1, y1, x2, y2 = [int(v) for v in bbox]
            
            # Ensure within frame
            x1 = max(0, min(x1, width-1))
            y1 = max(0, min(y1, height-1))
            x2 = max(x1+1, min(x2, width))
            y2 = max(y1+1, min(y2, height))
            
            if x2 > x1 and y2 > y1:
                # Save crop
                crop = frame[y1:y2, x1:x2]
                
                if crop.size > 0:
                    track_dir = f"{crops_dir}/track_{track_id}"
                    os.makedirs(track_dir, exist_ok=True)
                    
                    crop_filename = f"{track_dir}/frame_{frame_idx:06d}.jpg"
                    cv2.imwrite(crop_filename, crop)
                    
                    # Extract and store feature
                    try:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        feat = tracker.feature_extractor([crop_rgb])[0].cpu().numpy()
                        track_embeddings[track_id].append(feat)
                    except Exception:
                        pass
                    
                    # Update track metadata
                    if track_id not in track_data:
                        track_data[track_id] = {
                            'first_frame': frame_idx,
                            'last_frame': frame_idx,
                            'crop_paths': [],
                            'num_detections': 0
                        }
                    
                    track_data[track_id]['last_frame'] = frame_idx
                    track_data[track_id]['crop_paths'].append(crop_filename)
                    track_data[track_id]['num_detections'] += 1
                
                # Draw on frame
                color = get_color_for_id(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'ID: {track_id}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Store tracks for this frame
        all_tracks[frame_idx] = tracks
        
        # Add frame info
        info_text = f"Camera {camera_id} | Frame: {frame_idx:06d} | Tracks: {len(tracks)}"
        cv2.putText(frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Progress
        if frame_idx % 100 == 0 and frame_idx > 0:
            print(f"Frame {frame_idx} - Active tracks: {len(tracks)}, "
                  f"Total unique IDs: {len(track_data)}")
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    print(f"\nTracking complete for Camera {camera_id}")
    print(f"Processed {frame_idx} frames")
    print(f"Total unique tracks: {len(track_data)}")
    
    # Compute track signatures
    print("Computing track signatures from embeddings...")
    for track_id, embeddings in track_embeddings.items():
        if len(embeddings) > 0:
            signature = np.mean(embeddings, axis=0)
            signature = signature / (np.linalg.norm(signature) + 1e-8)
            track_data[track_id]['signature'] = signature
            track_data[track_id]['temporal_range'] = (
                track_data[track_id]['first_frame'],
                track_data[track_id]['last_frame']
            )
        else:
            track_data[track_id]['signature'] = np.zeros(512)
            track_data[track_id]['temporal_range'] = (
                track_data[track_id]['first_frame'],
                track_data[track_id]['last_frame']
            )
    
    # Prepare results
    results = {
        'camera_id': camera_id,
        'video_path': video_path,
        'total_frames': frame_idx,
        'fps': fps,
        'resolution': (width, height),
        'tracks': track_data,
        'frame_tracks': all_tracks
    }
    
    # Save tracking data
    tracking_file = f"{camera_dir}/tracking_data.pkl"
    print(f"Saving tracking data to {tracking_file}")
    with open(tracking_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"Output video: {output_video_path}")
    
    return results


def get_color_for_id(track_id):
    """Generate consistent color for track ID"""
    # Convert to int if it's a string
    if isinstance(track_id, str):
        # Hash the string to get a consistent integer
        track_id_int = hash(track_id) % (2**31)
    else:
        track_id_int = int(track_id)
    
    np.random.seed(track_id_int)
    color = tuple(np.random.randint(0, 255, 3).tolist())
    return color



def main():
    """Standalone execution"""
    parser = argparse.ArgumentParser(description='DeepSORT + Torchreid tracking')
    parser.add_argument('--video', required=True, help='Path to video file')
    parser.add_argument('--camera_id', type=int, required=True, help='Camera ID')
    parser.add_argument('--detections', required=True, help='Path to detections pickle file')
    parser.add_argument('--output_dir', default='./results', help='Output directory')
    parser.add_argument('--max_age', type=int, default=50, help='DeepSORT max age')
    parser.add_argument('--n_init', type=int, default=3, help='DeepSORT n_init')
    parser.add_argument('--max_cosine_distance', type=float, default=0.3, 
                       help='DeepSORT max cosine distance')
    
    args = parser.parse_args()
    
    config = {
        'max_age': args.max_age,
        'n_init': args.n_init,
        'max_cosine_distance': args.max_cosine_distance,
        'nn_budget': 100
    }
    
    results = track_video(
        args.video,
        args.camera_id,
        args.output_dir,
        args.detections,
        config
    )
    
    print(f"\nResults saved to {args.output_dir}/camera_{args.camera_id}/")


if __name__ == '__main__':
    main()
