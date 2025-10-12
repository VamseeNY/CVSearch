#!/usr/bin/env python3
"""
Load Data Module
Loads tracking results and builds person database for searching
"""

import pickle
import os
import json
from collections import defaultdict


def load_phase1_results(results_dir):
    """
    Load all Phase 1 results
    
    Args:
        results_dir: Directory containing Phase 1 outputs
    
    Returns:
        dict: Complete data structure with camera_tracks, global_mapping, person_images
    """
    
    print("Loading Phase 1 results...")
    print(f"Results directory: {results_dir}")
    
    # Load camera tracking data
    camera_tracks = {}
    for item in sorted(os.listdir(results_dir)):
        if item.startswith('camera_'):
            camera_id = int(item.split('_')[1])
            tracking_file = f"{results_dir}/{item}/tracking_data.pkl"
            
            if os.path.exists(tracking_file):
                with open(tracking_file, 'rb') as f:
                    camera_tracks[camera_id] = pickle.load(f)
                print(f"Loaded Camera {camera_id}: {len(camera_tracks[camera_id]['tracks'])} tracks")
    
    # Load global ID mapping
    mapping_file = f"{results_dir}/global_id_mapping.pkl"
    if os.path.exists(mapping_file):
        with open(mapping_file, 'rb') as f:
            global_mapping = pickle.load(f)
        print(f"Loaded global ID mapping: {len(set(global_mapping.values()))} unique persons")
    else:
        print("No global ID mapping found, using local track IDs")
        global_mapping = None
    
    # Build person images database
    person_images = build_person_database(results_dir, camera_tracks, global_mapping)
    
    print(f"\n Database ready:")
    print(f"  • Cameras: {len(camera_tracks)}")
    print(f"  • Global persons: {len(person_images)}")
    total_images = sum(len(imgs) for imgs in person_images.values())
    print(f"  • Total person images: {total_images}")
    
    return {
        'camera_tracks': camera_tracks,
        'global_mapping': global_mapping,
        'person_images': person_images,
        'results_dir': results_dir
    }


def build_person_database(results_dir, camera_tracks, global_mapping):
    """
    Build database of person images organized by global ID
    
    Args:
        results_dir: Results directory
        camera_tracks: Camera tracking data
        global_mapping: Global ID mapping (or None for single camera)
    
    Returns:
        dict: {global_id: [list of image paths]}
    """
    
    print("\n Building person image database...")
    
    person_images = defaultdict(list)
    
    if global_mapping:
        # Multi-camera: use global IDs
        for (camera_id, local_track_id), global_id in global_mapping.items():
            crop_dir = f"{results_dir}/camera_{camera_id}/crops/track_{local_track_id}"
            
            if os.path.exists(crop_dir):
                image_paths = [
                    f"{crop_dir}/{f}" 
                    for f in sorted(os.listdir(crop_dir)) 
                    if f.endswith(('.jpg', '.png'))
                ]
                person_images[global_id].extend(image_paths)
    else:
        # Single camera: use local track IDs
        for camera_id, camera_data in camera_tracks.items():
            for local_track_id in camera_data['tracks'].keys():
                crop_dir = f"{results_dir}/camera_{camera_id}/crops/track_{local_track_id}"
                
                if os.path.exists(crop_dir):
                    # Use tuple (camera_id, track_id) as key for single camera
                    person_id = (camera_id, local_track_id)
                    image_paths = [
                        f"{crop_dir}/{f}" 
                        for f in sorted(os.listdir(crop_dir))
                        if f.endswith(('.jpg', '.png'))
                    ]
                    person_images[person_id] = image_paths
    
    print(f" Built database for {len(person_images)} persons")
    
    return dict(person_images)


def get_person_metadata(person_id, data):
    """
    Get metadata for a specific person
    
    Args:
        person_id: Global person ID
        data: Complete data structure from load_phase1_results
    
    Returns:
        dict: Person metadata
    """
    
    global_mapping = data['global_mapping']
    camera_tracks = data['camera_tracks']
    person_images = data['person_images']
    
    metadata = {
        'person_id': person_id,
        'total_images': len(person_images.get(person_id, [])),
        'appearances': []
    }
    
    if global_mapping:
        # Find all appearances of this person
        for (camera_id, local_track_id), global_id in global_mapping.items():
            if global_id == person_id:
                track_info = camera_tracks[camera_id]['tracks'][local_track_id]
                metadata['appearances'].append({
                    'camera_id': camera_id,
                    'local_track_id': local_track_id,
                    'num_detections': track_info.get('num_detections', 0),
                    'first_frame': track_info.get('first_frame', 0),
                    'last_frame': track_info.get('last_frame', 0)
                })
    
    return metadata


def main():
    """Standalone execution - test data loading"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Load Phase 1 results')
    parser.add_argument('--results_dir', required=True, help='Phase 1 results directory')
    
    args = parser.parse_args()
    
    data = load_phase1_results(args.results_dir)
    
    print("\n Data loaded successfully!")
    print(f"\nSample person IDs: {list(data['person_images'].keys())[:5]}")


if __name__ == '__main__':
    main()
