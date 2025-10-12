# CVSearch - A Multi-Person Tracking and Query System
A complete end-end pipeline for detecting, tracking and searching people across multiple camera feeds using computer vision, and vision-language models

## Features
- Multi-camera tracking using YOLOv11
- Cross camera re identification using ReID embeddings
- Interactive and batch search options
- Comprehensive visualization and export options

<img width="907" height="297" alt="image" src="https://github.com/user-attachments/assets/94fef7a4-cdb6-4774-a191-9fd15888e59d" />
<img width="747" height="546" alt="image" src="https://github.com/user-attachments/assets/8ea23722-85e2-4332-bdf6-5683655b8bf5" />



## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Phase 1: Tracking](#phase-1-tracking)
- [Phase 2: Search](#phase-2-search)
- [Configuration](#configuration)
- [Technical details](#technical-details)
## Installation
### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM (16GB+ for multiple cameras)

### Clone repository
git clone https://github.com/VamseeNY/CVSearch.git 

### Install required packages
cd CVSearch 
pip install -r requirements.txt

## Quick Start
### Data Preparation 
Organize input videos:
```
videos/
│   camera1.mp4
│   camera2.mp4
```
### Create Configuration file
Create config.yaml
```
output_dir: "./results"

cameras:
  - id: 0
    video: "videos/camera1.mp4"
  - id: 1
    video: "videos/camera2.mp4"

tracking:
  confidence_threshold: 0.75
  max_age: 50
  n_init: 3
  max_cosine_distance: 0.3

reid:
  similarity_threshold: 0.6
  temporal_tolerance: 300
  use_temporal: true
```

### Run Phase 1 (Tracking)
python pipeline.py --config config.yaml

### Run Phase 2 (Search)
```
# Interactive search
python phase2_query/interactive_search.py --results_dir ./results

# Or quick search
python phase2_query/interactive_search.py --results_dir ./results \
    --query "person wearing red shirt"
```

## Pipeline Overview
<img width="732" height="728" alt="image" src="https://github.com/user-attachments/assets/9c6c3a3f-ac23-4019-93cb-a9aa3a2f7d41" />

## Phase 1: Tracking
Phase 1 consists of modular scripts that can be run independently or together via the orchestrator.

### Directory Structure
```
CVSearch/
│   detect.py             #YOLO person detection
│   track.py              #DeepSORT + TorchReID Tracking
|   cross_camera_reid.py  #Cross camera re-identification
|   save_results.py       #Export results in multiple formats
|   pipeline.py           #Main pipeline
```

### Running Individual modules
1. Detection 
```
python detect.py --video camera1.mp4 --camera_id 0 --output_dir ./results
```
2. Tracking (requires detection results)
```
python track.py --video camera1.mp4 --camera_id 0 --detections ./results/camera_0/detections/detections.pkl --output_dir ./results
```
3. Cross-camera ReID
```
python cross_camera_reid.py --input_dir ./results --threshold 0.6
```
4. Export results
```
python save_results.py --output_dir ./results
```

### Running complete pipeline
```
# Sequential processing
python pipeline.py --config config.yaml

# Parallel processing (faster, more memory)
python pipeline.py --config config.yaml --parallel
```

### Phase 1 output structure
```
results/
├── camera_0/
│   ├── detections/
│   │   └── detections.pkl           # Bounding boxes and metadata
│   ├── crops/
│   │   ├── track_1/                 # Person crops for each track
│   │   │   ├── frame_000045.jpg
│   │   │   └── ...
│   │   └── track_2/
│   ├── tracking_data.pkl            # Complete tracking data
│   ├── tracked_video.mp4            # Annotated video
│   └── track_summary.csv            # Per-track statistics
├── camera_1/
│   └── ...
├── global_id_mapping.pkl            # Cross-camera person mapping
├── global_id_mapping.csv            # Human-readable mapping
├── reid_summary.json                # Detailed ReID statistics
├── reid_summary.txt                 # Human-readable summary
└── crop_index.json                  # Index of all crops by global ID
```

## Phase 2: Search

### Directory Structure
```
CVSearch/
│   search_siglip.py             #SigLIP search engine
│   interactive_search.py        #Interactive interface
|   visualize.py                 #Result visualization
```

### Running Search functions 
```
# With visualization
python interactive_search.py \
    --results_dir ./results \
    --query "person in red shirt"

# Text-only results
python interactive_search.py \
    --results_dir ./results \
    --query "man with backpack" \
    --no_images
```

### Phase 2 Outputs

```
results/
├── search_person_wearing_red_shirt.png
├── search_man_with_backpack.png
├── search_history.json
└── batch_search_summary.json
```

## Configuration 
```
# Multi-Camera Person Tracking Pipeline Configuration

output_dir: "./multi_camera_results"

# Camera definitions
cameras:
  - id: 0
    video: "videos/camera1.mp4"
    location: "entrance"        # Optional description
  
  - id: 1
    video: "videos/camera2.mp4"
    location: "corridor"
  
  - id: 2
    video: "videos/camera3.mp4"
    location: "exit"

# Detection parameters
detection:
  model: "yolov8n.pt"           # yolov8n.pt, yolov8s.pt, yolo11n.pt
  confidence_threshold: 0.75     # Detection confidence (0-1)

# Tracking parameters
tracking:
  confidence_threshold: 0.75
  max_age: 50                    # Frames to keep track alive
  n_init: 3                      # Detections before confirmation
  max_cosine_distance: 0.3       # Feature matching threshold
  nn_budget: 100                 # Feature gallery size
  reid_model: "osnet_x1_0"       # ReID model name

# Cross-camera re-identification
reid:
  similarity_threshold: 0.6      # Matching threshold (0-1)
  temporal_tolerance: 300        # Frame overlap tolerance
  use_temporal: true             # Enable temporal constraints

# Processing options
processing:
  parallel: false                # Parallel camera processing
  save_visualizations: true
  save_crops: true

# Advanced options
advanced:
  frame_skip: 1                  # Process every Nth frame
  output_video_codec: "mp4v"
  output_video_quality: 30
```

### Parameter Tuning Guide
#### Detection Confidence (0.5-0.9)
- **0.5-0.6**: More detections, more false positives
- **0.75**: Balanced (recommended)
- **0.8-0.9**: Fewer false positives, may miss some persons

#### Tracking Parameters

**max_age** (30-100 frames)
- Lower: Fewer ID switches, tracks lost in occlusions
- Higher: Persistent tracking, more ID switches

**n_init** (3-5 frames)
- Lower: Faster ID assignment, more false tracks
- Higher: Fewer false tracks, slower assignment

**max_cosine_distance** (0.2-0.4)
- Lower: Stricter matching, more ID switches
- Higher: Lenient matching, fewer switches but more errors

#### ReID Similarity Threshold (0.5-0.8)
- **0.5**: Very lenient, may merge different persons
- **0.6**: Balanced (recommended)
- **0.7**: Strict, fewer false matches
- **0.8**: Very strict, may miss correct matches

#### Temporal Tolerance
Depends on camera layout and FPS:
- **150 frames** = 5 seconds @ 30fps (nearby cameras)
- **300 frames** = 10 seconds @ 30fps (moderate distance)
- **600 frames** = 20 seconds @ 30fps (distant cameras)

## Technical Details

### Models Used

1. **YOLO (YOLOv11)** - Person detection
   - Fast and accurate object detection
   - Class 0 = person
   
2. **Torchreid (OSNet)** - Person re-identification
   - 512-dimensional embeddings
   - Trained on person ReID datasets
   
3. **DeepSORT** - Multi-object tracking
   - Combines detection and ReID features
   - Handles occlusions and ID persistence
   
4. **SigLIP** - Vision-language search
   - Google's state-of-the-art model
   - Sigmoid-based similarity (better than CLIP)


## Areas for improvement
- Additional search models
- Real time streaming support
- Web Interface
- Database integration
