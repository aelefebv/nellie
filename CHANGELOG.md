# Changelog
All notable changes to this project will be documented in this file.
The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]
### Added
### Changed
### Fixed
### Removed

## [1.0.0] - date tbd
### Added
- A changelog
- Major speed ups across the board for segmentation and tracking:
    - **GPU (tested with CUDA / Windows)**
        - Filter: 5.19× faster (+419%)
        - Label: 3.28× faster (+228%)
        - Network: 19.47× faster (+1,847%)
        - Markers: 7.58× faster (+658%)
        - HuMomentTracking: 7.40× faster (+640%)
        - VoxelReassigner: 2.03× faster (+103%)
        - Hierarchy: no change
        - Total: 3.25× faster (+225%)
    - **CPU (tested with Mac)**
        - Filter: 1.47× faster (+47%)
        - Label: no change
        - Network: 8.57× faster (+757%)
        - Markers: no change
        - HuMomentTracking: 1.52× faster (+52%)
        - VoxelReassigner: 2.22× faster (+122%)
        - Hierarchy: no change
        - Total: 1.58× faster (+58%)
- UV compatibility
- Added a dropdown for which stat to visualize for the feature of interest in the analysis plugin tab
### Changed
- Docs to mkdocs format
- Defaulted logger to INFO level
### Fixed
- Lots of little things in the napari GUI (e.g. buttons turning on and off, things like that)
- Removed some annoying test strings
- Track visualization to only tracks that end up at a valid mask pixel
- Version verification
