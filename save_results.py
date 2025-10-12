#!/usr/bin/env python3
"""
Save Results Module
Exports tracking and ReID results in multiple formats
"""

import pickle
import os
import json
import csv
import argparse
from collections import defaultdict
from datetime import datetime


def load_all_data(output_dir):
    """
    Load all tracking and ReID data
    
    Args:
        output_dir: Directory containing results
    
    Returns:
        tuple: (camera_tracks, global_id_mapping)
    """
    
    print("Loading all data...")
    
    # Load camera tracks
    camera_tracks = {}
    for item in sorted(os.listdir(output_dir)):
        if item.startswith('camera_'):
            camera_id = int(item.split('_')[1])
            tracking_file = f"{output_dir}/{item}/tracking_data.pkl"
            
            if os.path.exists(tracking_file):
                with open(tracking_file, 'rb') as f:
                    camera_tracks[camera_id] = pickle.load(f)
                print(f"Loaded Camera {camera_id}")
    
    # Load global mapping
    mapping_file = f"{output_dir}/global_id_mapping.pkl"
    if os.path.exists(mapping_file):
        with open(mapping_file, 'rb') as f:
            global_id_mapping = pickle.load(f)
        print(f"Loaded global ID mapping")
    else:
        print("No global ID mapping found")
        global_id_mapping = None
    
    return camera_tracks, global_id_mapping


def save_global_mapping_csv(global_id_mapping, camera_tracks, output_dir):
    """Save global ID mapping as CSV"""
    
    print("Saving global ID mapping CSV...")
    
    csv_path = f"{output_dir}/global_id_mapping.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['camera_id', 'local_track_id', 'global_id', 'num_detections',
                        'first_frame', 'last_frame', 'duration_frames'])
        
        for (camera_id, local_track_id), global_id in sorted(global_id_mapping.items()):
            track_info = camera_tracks[camera_id]['tracks'][local_track_id]
            num_detections = track_info.get('num_detections', 0)
            first_frame = track_info.get('first_frame', 0)
            last_frame = track_info.get('last_frame', 0)
            duration = last_frame - first_frame + 1
            
            writer.writerow([camera_id, local_track_id, global_id, num_detections,
                           first_frame, last_frame, duration])
    
    print(f"Saved: {csv_path}")


def save_reid_summary_json(global_id_mapping, camera_tracks, output_dir):
    """Save detailed ReID summary as JSON"""
    
    print("Saving ReID summary JSON...")
    
    # Build global person statistics
    global_person_stats = defaultdict(lambda: {
        'appearances': [],
        'total_detections': 0,
        'cameras': set(),
        'total_duration': 0
    })
    
    for (camera_id, local_track_id), global_id in global_id_mapping.items():
        track_info = camera_tracks[camera_id]['tracks'][local_track_id]
        
        num_detections = track_info.get('num_detections', 0)
        first_frame = track_info.get('first_frame', 0)
        last_frame = track_info.get('last_frame', 0)
        duration = last_frame - first_frame + 1
        
        global_person_stats[global_id]['appearances'].append({
            'camera_id': camera_id,
            'local_track_id': local_track_id,
            'num_detections': num_detections,
            'first_frame': first_frame,
            'last_frame': last_frame,
            'duration': duration
        })
        
        global_person_stats[global_id]['total_detections'] += num_detections
        global_person_stats[global_id]['cameras'].add(camera_id)
        global_person_stats[global_id]['total_duration'] += duration
    
    # Convert sets to lists for JSON
    for gid in global_person_stats:
        global_person_stats[gid]['cameras'] = sorted(list(global_person_stats[gid]['cameras']))
        global_person_stats[gid]['num_cameras'] = len(global_person_stats[gid]['cameras'])
    
    summary = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_cameras': len(camera_tracks),
            'total_local_tracks': len(global_id_mapping),
            'unique_global_persons': len(set(global_id_mapping.values()))
        },
        'global_persons': dict(global_person_stats)
    }
    
    json_path = f"{output_dir}/reid_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved: {json_path}")


def save_reid_summary_txt(global_id_mapping, camera_tracks, output_dir):
    """Save human-readable text summary"""
    
    print("Saving ReID summary TXT...")
    
    # Build statistics
    global_person_stats = defaultdict(lambda: {
        'appearances': [],
        'total_detections': 0
    })
    
    for (camera_id, local_track_id), global_id in global_id_mapping.items():
        track_info = camera_tracks[camera_id]['tracks'][local_track_id]
        
        global_person_stats[global_id]['appearances'].append({
            'camera_id': camera_id,
            'local_track_id': local_track_id,
            'num_detections': track_info.get('num_detections', 0),
            'first_frame': track_info.get('first_frame', 0),
            'last_frame': track_info.get('last_frame', 0)
        })
        global_person_stats[global_id]['total_detections'] += track_info.get('num_detections', 0)
    
    txt_path = f"{output_dir}/reid_summary.txt"
    
    with open(txt_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MULTI-CAMERA PERSON TRACKING & RE-IDENTIFICATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Number of cameras: {len(camera_tracks)}\n")
        f.write(f"Total local tracks: {len(global_id_mapping)}\n")
        f.write(f"Unique global persons: {len(set(global_id_mapping.values()))}\n")
        
        # Multi-camera statistics
        multi_camera_count = sum(1 for stats in global_person_stats.values() 
                                if len(stats['appearances']) > 1)
        f.write(f"Persons in multiple cameras: {multi_camera_count}\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("GLOBAL PERSON DETAILS\n")
        f.write("=" * 70 + "\n\n")
        
        for global_id in sorted(global_person_stats.keys()):
            stats = global_person_stats[global_id]
            f.write(f"Global ID {global_id}:\n")
            f.write(f"  Total detections: {stats['total_detections']}\n")
            f.write(f"  Appears in {len(stats['appearances'])} camera(s):\n")
            
            for app in stats['appearances']:
                f.write(f"    • Camera {app['camera_id']}, "
                       f"Local Track {app['local_track_id']}: "
                       f"{app['num_detections']} detections "
                       f"(frames {app['first_frame']}-{app['last_frame']})\n")
            f.write("\n")
    
    print(f"Saved: {txt_path}")


def save_per_camera_summaries(camera_tracks, output_dir):
    """Save individual summaries for each camera"""
    
    print("Saving per-camera summaries...")
    
    for camera_id, camera_data in camera_tracks.items():
        camera_dir = f"{output_dir}/camera_{camera_id}"
        
        # CSV summary
        csv_path = f"{camera_dir}/track_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['track_id', 'num_detections', 'first_frame', 'last_frame', 
                           'duration', 'num_crops'])
            
            for track_id, track_info in camera_data['tracks'].items():
                writer.writerow([
                    track_id,
                    track_info.get('num_detections', 0),
                    track_info.get('first_frame', 0),
                    track_info.get('last_frame', 0),
                    track_info.get('last_frame', 0) - track_info.get('first_frame', 0) + 1,
                    len(track_info.get('crop_paths', []))
                ])
        
        print(f"Camera {camera_id}: {csv_path}")


def save_crop_index(camera_tracks, global_id_mapping, output_dir):
    """Save index of all person crops organized by global ID"""
    
    print("Creating crop index...")
    
    crop_index = defaultdict(list)
    
    for (camera_id, local_track_id), global_id in global_id_mapping.items():
        track_info = camera_tracks[camera_id]['tracks'][local_track_id]
        crop_paths = track_info.get('crop_paths', [])
        
        for crop_path in crop_paths:
            crop_index[global_id].append({
                'path': crop_path,
                'camera_id': camera_id,
                'local_track_id': local_track_id
            })
    
    # Save as JSON
    json_path = f"{output_dir}/crop_index.json"
    with open(json_path, 'w') as f:
        json.dump({str(k): v for k, v in crop_index.items()}, f, indent=2)
    
    print(f"Saved crop index: {json_path}")
    
    return dict(crop_index)


def generate_statistics(camera_tracks, global_id_mapping):
    """Generate and display statistics"""
    
    print("\n" + "=" * 70)
    print("PIPELINE STATISTICS")
    print("=" * 70)
    
    # Camera statistics
    print(f"\n📹 Cameras: {len(camera_tracks)}")
    total_frames = sum(cam['total_frames'] for cam in camera_tracks.values())
    total_local_tracks = sum(len(cam['tracks']) for cam in camera_tracks.values())
    print(f"Total frames processed: {total_frames}")
    print(f"Total local tracks: {total_local_tracks}")
    
    # Global statistics
    if global_id_mapping:
        n_global = len(set(global_id_mapping.values()))
        print(f"👥 Unique global persons: {n_global}")
        
        # Distribution
        global_counts = defaultdict(int)
        for gid in global_id_mapping.values():
            global_counts[gid] += 1
        
        multi_camera = sum(1 for count in global_counts.values() if count > 1)
        print(f"Persons in multiple cameras: {multi_camera}")
    
    print("=" * 70 + "\n")


def save_all_results(output_dir):
    """
    Main function to save all results in various formats
    
    Args:
        output_dir: Directory containing pipeline results
    """
    
    print("=" * 70)
    print("SAVING ALL RESULTS")
    print("=" * 70)
    print()
    
    # Load all data
    camera_tracks, global_id_mapping = load_all_data(output_dir)
    
    if not camera_tracks:
        print("No camera tracking data found")
        return
    
    # Save per-camera summaries
    save_per_camera_summaries(camera_tracks, output_dir)
    
    if global_id_mapping:
        # Save global mapping in multiple formats
        save_global_mapping_csv(global_id_mapping, camera_tracks, output_dir)
        save_reid_summary_json(global_id_mapping, camera_tracks, output_dir)
        save_reid_summary_txt(global_id_mapping, camera_tracks, output_dir)
        save_crop_index(camera_tracks, global_id_mapping, output_dir)
    else:
        print("No global ID mapping found, skipping ReID summaries")
    
    # Generate statistics
    generate_statistics(camera_tracks, global_id_mapping)
    
    print("=" * 70)
    print("ALL RESULTS SAVED")
    print("=" * 70)
    print(f"\n Output directory: {output_dir}/")
    print("\n Generated files:")
    print("  • camera_X/tracking_data.pkl - Tracking data per camera")
    print("  • camera_X/tracked_video.mp4 - Annotated videos")
    print("  • camera_X/track_summary.csv - Per-camera summaries")
    print("  • camera_X/crops/track_Y/*.jpg - Person crops")
    if global_id_mapping:
        print("  • global_id_mapping.pkl - Global ID mapping (pickle)")
        print("  • global_id_mapping.csv - Global ID mapping (CSV)")
        print("  • reid_summary.json - Detailed ReID statistics")
        print("  • reid_summary.txt - Human-readable summary")
        print("  • crop_index.json - Index of all crops by global ID")
    print()


def main():
    """Standalone execution"""
    parser = argparse.ArgumentParser(description='Save and export pipeline results')
    parser.add_argument('--output_dir', required=True, 
                       help='Directory containing pipeline results')
    
    args = parser.parse_args()
    
    save_all_results(args.output_dir)
    
    print("Results export complete!")


if __name__ == '__main__':
    main()
