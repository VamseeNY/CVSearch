#!/usr/bin/env python3
"""
Phase 1 Pipeline Orchestrator
Calls detect.py -> track.py -> cross_camera_reid.py -> save_results.py
"""

import argparse
import yaml
import os
import sys
from datetime import datetime
from multiprocessing import Pool

# Import individual modules
from detect import detect_video
from track import track_video
from cross_camera_reid import load_camera_tracks, perform_cross_camera_reid
from save_results import save_all_results


def load_config(config_path):
    """Load YAML configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_phase1_pipeline(config_path, parallel=False):
    """
    Run complete Phase 1 pipeline
    """
    
    start_time = datetime.now()
    
    print("=" * 70)
    print(" PHASE 1: MULTI-CAMERA TRACKING PIPELINE")
    print("=" * 70)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load config
    config = load_config(config_path)
    cameras = config['cameras']
    output_dir = config.get('output_dir', './results')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output: {output_dir}")
    print(f"Cameras: {len(cameras)}\n")
    
    # Step 1: Detection
    print("=" * 70)
    print("STEP 1: YOLO DETECTION")
    print("=" * 70 + "\n")
    
    detection_config = config.get('tracking', {})
    conf_threshold = detection_config.get('confidence_threshold', 0.75)
    
    for i, cam in enumerate(cameras, 1):
        print(f"Camera {i}/{len(cameras)}: ID {cam['id']}")
        detect_video(cam['video'], cam['id'], output_dir, conf_threshold)
        print()
    
    # Step 2: Tracking
    print("=" * 70)
    print("STEP 2: DEEPSORT + TORCHREID TRACKING")
    print("=" * 70 + "\n")
    
    tracking_config = config.get('tracking', {})
    
    for i, cam in enumerate(cameras, 1):
        print(f"Camera {i}/{len(cameras)}: ID {cam['id']}")
        detections_file = f"{output_dir}/camera_{cam['id']}/detections/detections.pkl"
        track_video(cam['video'], cam['id'], output_dir, detections_file, tracking_config)
        print()
    
    # Step 3: Cross-camera ReID
    print("=" * 70)
    print("STEP 3: CROSS-CAMERA RE-IDENTIFICATION")
    print("=" * 70 + "\n")
    
    camera_tracks = load_camera_tracks(output_dir)
    reid_config = config.get('reid', {})
    
    global_mapping = perform_cross_camera_reid(
        camera_tracks,
        similarity_threshold=reid_config.get('similarity_threshold', 0.6),
        temporal_tolerance=reid_config.get('temporal_tolerance', 300),
        use_temporal=reid_config.get('use_temporal', True)
    )
    
    # Save global mapping
    import pickle
    with open(f"{output_dir}/global_id_mapping.pkl", 'wb') as f:
        pickle.dump(global_mapping, f)
    print()
    
    # Step 4: Save results
    print("=" * 70)
    print("STEP 4: SAVING RESULTS")
    print("=" * 70 + "\n")
    
    save_all_results(output_dir)
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE!")
    print("=" * 70)
    print(f"Duration: {duration}")
    print(f"Results: {output_dir}/")
    print("\n Ready for Phase 2: Person Search")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Phase 1 Pipeline')
    parser.add_argument('--config', required=True, help='Config YAML file')
    parser.add_argument('--parallel', action='store_true', help='Parallel processing')
    
    args = parser.parse_args()
    
    run_phase1_pipeline(args.config, args.parallel)


if __name__ == '__main__':
    main()
