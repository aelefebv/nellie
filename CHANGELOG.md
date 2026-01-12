# Changelog
All notable changes to this project will be documented in this file.
The format is based on Keep a Changelog and this project adheres to Semantic Versioning.

## [Unreleased]
### Added
### Changed
### Fixed
### Removed

## [1.0.0] - 01/12/2026
### Added
- A changelog
- Major speed ups across the board for segmentation and tracking (all tests performed on CUDA / Windows):
    - **GPU**
        - Filter: 9x faster
        - Label: 7x faster
        - Network: 14x faster
        - Markers: 10x faster 
        - HuMomentTracking: 6x faster 
        - VoxelReassigner: 2x faster 
        - Hierarchy: no change
        - Total: 3.5x faster 
    - **CPU**
        - Filter: 1.5x faster 
        - Label: 1.5x faster
        - Network: 11x faster
        - Markers: no change
        - HuMomentTracking: no change
        - VoxelReassigner: 1.5x faster
        - Hierarchy: no change
        - Total: 1.5x faster 
- UV compatibility
- Added a dropdown for which stat to visualize for the feature of interest in the analysis plugin tab
- Added a tab for advanced settings to tweak specific parameters for each step of the pipeline.
### Changed
- Docs to mkdocs format
- Defaulted logger to INFO level
### Fixed
- Lots of little things in the napari GUI (e.g. buttons turning on and off, things like that)
- Removed some annoying test strings
- Track visualization to only tracks that end up at a valid mask pixel
- Version verification
- Discover nellie entrypoint plugins when importlib is outdated.
