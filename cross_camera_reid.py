#!/usr/bin/env python3
"""
Cross-Camera Re-Identification Module
Matches persons across multiple camera views using ReID embeddings
"""

import pickle
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import AgglomerativeClustering
import argparse
import os


def load_camera_tracks(output_dir):
    """
    Load tracking data from all cameras
    
    Args:
        output_dir: Directory containing camera_X subdirectories
    
    Returns:
        dict: {camera_id: tracking_data}
    """
    
    print("Loading camera tracking data...")
    
    camera_tracks = {}
    
    for item in sorted(os.listdir(output_dir)):
        if item.startswith('camera_'):
            try:
                camera_id = int(item.split('_')[1])
                tracking_file = f"{output_dir}/{item}/tracking_data.pkl"
                
                if os.path.exists(tracking_file):
                    with open(tracking_file, 'rb') as f:
                        camera_tracks[camera_id] = pickle.load(f)
                    print(f"Camera {camera_id}: {len(camera_tracks[camera_id]['tracks'])} tracks")
                else:
                    print(f"Tracking file not found for Camera {camera_id}")
            except (ValueError, IndexError):
                print(f"Skipping invalid directory: {item}")
    
    if not camera_tracks:
        raise ValueError(f"No camera tracking data found in {output_dir}")
    
    print(f"Loaded {len(camera_tracks)} cameras")
    
    return camera_tracks


def perform_cross_camera_reid(camera_tracks, similarity_threshold=0.6, 
                               temporal_tolerance=300, use_temporal=True):
    """
    Match tracks across cameras using ReID embeddings
    
    Args:
        camera_tracks: Dict of tracking data per camera
        similarity_threshold: Minimum similarity for matching (0-1)
        temporal_tolerance: Frame tolerance for temporal overlap
        use_temporal: Whether to use temporal constraints
    
    Returns:
        dict: Mapping (camera_id, local_track_id) -> global_id
    """
    
    print(f"\nPerforming Cross-Camera Re-Identification...")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Temporal tolerance: {temporal_tolerance} frames")
    print(f"Use temporal constraints: {use_temporal}")
    
    # Build track database
    track_database = []
    track_keys = []
    
    for camera_id, camera_data in camera_tracks.items():
        for local_track_id, track_info in camera_data['tracks'].items():
            
            # Skip tracks without valid signatures
            if 'signature' not in track_info:
                continue
            
            signature = track_info['signature']
            
            # Skip zero signatures
            if np.allclose(signature, 0):
                continue
            
            track_database.append({
                'camera_id': camera_id,
                'local_track_id': local_track_id,
                'signature': signature,
                'temporal_range': track_info.get('temporal_range', (0, 0)),
                'num_detections': track_info.get('num_detections', 0)
            })
            track_keys.append((camera_id, local_track_id))
    
    n_tracks = len(track_database)
    print(f"Total valid tracks: {n_tracks}")
    
    if n_tracks == 0:
        raise ValueError("No valid tracks found for ReID")
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = np.zeros((n_tracks, n_tracks))
    
    for i in range(n_tracks):
        for j in range(i+1, n_tracks):
            track_i = track_database[i]
            track_j = track_database[j]
            
            # Same camera = no match
            if track_i['camera_id'] == track_j['camera_id']:
                similarity_matrix[i, j] = 0
                similarity_matrix[j, i] = 0
                continue
            
            # Temporal overlap check
            if use_temporal:
                start_i, end_i = track_i['temporal_range']
                start_j, end_j = track_j['temporal_range']
                
                # Can't be same person if tracks overlap in time
                if not (end_i + temporal_tolerance < start_j or 
                        end_j + temporal_tolerance < start_i):
                    similarity_matrix[i, j] = 0
                    similarity_matrix[j, i] = 0
                    continue
            
            # Compute cosine similarity
            try:
                sim = 1 - cosine(track_i['signature'], track_j['signature'])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim
            except Exception:
                similarity_matrix[i, j] = 0
                similarity_matrix[j, i] = 0
    
    print(f"Similarity matrix computed")
    
    # Cluster to assign global IDs
    print("Clustering tracks...")
    
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=0.0)
    
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - similarity_threshold,
        metric='precomputed',
        linkage='complete'
    )
    
    global_ids = clustering.fit_predict(distance_matrix)
    
    # Create mapping
    global_id_mapping = {}
    for idx, (camera_id, local_track_id) in enumerate(track_keys):
        global_id_mapping[(camera_id, local_track_id)] = int(global_ids[idx])
    
    # Statistics
    n_global_persons = len(set(global_ids))
    print(f"\nCross-camera ReID complete!")
    print(f"Unique persons identified: {n_global_persons}")
    
    # Multi-camera statistics
    global_id_counts = {}
    for gid in global_ids:
        global_id_counts[gid] = global_id_counts.get(gid, 0) + 1
    
    multi_camera = sum(1 for count in global_id_counts.values() if count > 1)
    print(f"Persons in multiple cameras: {multi_camera}")
    
    return global_id_mapping


def main():
    """Standalone execution"""
    parser = argparse.ArgumentParser(description='Cross-camera person re-identification')
    parser.add_argument('--input_dir', required=True, 
                       help='Directory containing camera tracking results')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Similarity threshold (0-1)')
    parser.add_argument('--temporal_tolerance', type=int, default=300,
                       help='Frame tolerance for temporal overlap')
    parser.add_argument('--no_temporal', action='store_true',
                       help='Disable temporal constraints')
    
    args = parser.parse_args()
    
    # Load camera tracks
    camera_tracks = load_camera_tracks(args.input_dir)
    
    # Perform ReID
    global_mapping = perform_cross_camera_reid(
        camera_tracks,
        similarity_threshold=args.threshold,
        temporal_tolerance=args.temporal_tolerance,
        use_temporal=not args.no_temporal
    )
    
    # Save results (will be handled by save module, but can save here too)
    output_file = f"{args.input_dir}/global_id_mapping.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(global_mapping, f)
    
    print(f"\n Global mapping saved to {output_file}")
    print("Cross-camera ReID complete!")


if __name__ == '__main__':
    main()
