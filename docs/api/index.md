# API Reference

Welcome to the Nellie API documentation. The Nellie package provides a complete pipeline for automated organelle segmentation, tracking, and hierarchical feature extraction in 2D/3D live-cell microscopy.

## Main Components

- **[Main Interface](run.md)** - Entry point for running the Nellie pipeline
- **[Segmentation](segmentation.md)** - Image filtering, labeling, and network analysis
- **[Tracking](tracking.md)** - Object tracking across timepoints
- **[Feature Extraction](feature_extraction.md)** - Hierarchical feature computation
- **[Image Info](im_info.md)** - File handling and metadata management
- **[Utilities](utils.md)** - Helper functions for logging and GPU operations

## Quick Start

```python
from nellie import run
from nellie.im_info import FileInfo

# Load your microscopy image
file_info = FileInfo("path/to/image.tif")

# Run the complete Nellie pipeline
im_info = run(file_info)
```

## Pipeline Overview

The Nellie pipeline consists of several stages:

1. **Filtering**: Multi-scale Frangi filtering for vesselness detection
2. **Segmentation**: Threshold-based instance segmentation
3. **Network Analysis**: Skeletonization and topological analysis
4. **Marker Detection**: Motion capture marker generation
5. **Tracking**: Temporal tracking using Hu moments and flow interpolation
6. **Voxel Reassignment**: Frame-to-frame voxel tracking
7. **Feature Extraction**: Multi-level hierarchical features
