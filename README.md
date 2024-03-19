# Nellie
Automated organelle segmentation, tracking, and hierarchical feature extraction in 2D/3D live-cell microscopy


https://github.com/aelefebv/nellie/assets/26515909/1574ad99-a694-42bf-94c2-15619383422c


## Installation

**Notes:**
- It is recommended (but usually not required) to [create a new environment](https://docs.python.org/3/library/venv.html) for Nellie to avoid conflicts with other packages.
- May take several minutes to install.
- Choose one of the following methods, and only one!
### Via Napari plugin manager:
If not already installed, install Napari: https://napari.org/stable/tutorials/fundamentals/installation
1. Open Napari
2. Go to ```Plugins > Install/Uninstall Plugins...```
3. Search for Nellie and click ```Install```
### Via PIP:
```bash
pip install git+https://github.com/aelefebv/nellie.git
```
### Installation NVIDIA GPU (Windows, Linux):
Follow one of the above methods, then run the following:
```bash
pip install cupy-cudaXXx
```
- replace ```cupy-cudaXXx``` with the [appropriate version](https://docs.cupy.dev/en/stable/install.html#installing-cupy) for your CUDA version.
  - i.e. ```cupy-cuda11x``` for CUDA 11.x or ```cupy-cuda12x``` for CUDA 12.x
- if you don't have CUDA installed, [go here](https://docs.cupy.dev/en/stable/install.html).
- Mac Metal GPU-acceleration coming... eventually.

## Usage
https://github.com/aelefebv/nellie/assets/26515909/6a2ea765-1df3-4210-9f1c-60ee8e3d0228
### Running Nellie's processing pipeline
1. Start Napari (open a Terminal and type napari)
    ```bash
    napari
    ```
2. Go to 
```Plugins > Nellie (nellie)``` then to the ```File select``` tab.
3. Click ```Select File``` of ```Select Folder``` to select your image(s).
   - If the metadata boxes do not fill in automatically and turn red, this means Nellie did not detect that metadata portion from your image, and you must manually enter it or reformat your image and try again.
     - The metadata slot will appear green if it is in the correct format.
   - *Note, if you are batch processing, the metadata must be the same for all images if any of them are in an incorrect format (this will be fixed eventually). If they are different, but all pass validation, then it will process fine.
   - You can preview 2 time points of your image via the ```Open preview``` button once the metadata is filled in to ensure it looks correct.
4. Click the ```Process``` tab.
   - If you have multiple fluorescence channels, select the channel you want to process/analyze.
   - If you only want to analyze up to a certain timepoint, you can set this in slider. By default it will run all timepoints.
   - If you have odd noise on the edges of your image, check the ```Remove image edges``` checkbox.
5. You can run the full pipeline with ```Run Nellie```, or run individual steps below.
    - Steps can only be run once its previous step has been run.
    - Likewise, visualizations in the ```Visualization``` tab can only be opened once its respective step has been run.
6. All intermediate files and output csvs will be saved to ```[image_directory]/nellie_output/```.
   - A separate .csv is created for each level of the organellar hierarchy.
7. Once features have been exported, Nellie will automatically detect this, and allow analysis via the ```Analyze``` tab.
   - Analysis at this point is optional, but can be helpful for visualizing, and selectively exporting data.

### Using Nellie's visualization plugin
1. Follow the previous processing steps, you only need to do this once per file as long as you don't move or delete the files.
2. Open the ```Visualization``` tab
3. Select a visualization from the list.
   1. ```Raw```: Visualize the raw data for the processed channel.
   2. ```Preprocessed```: Visualize the contrast-enhanced data.
   3. ```Segmentation```: Visualize the organelle and branch instance segmentation masks.
   4. ```Mocap Markers```: Visualize the mocap markers used for waypoints.
   5. ```Reassigned Labels```: Visualize the organelle and branch instance segmentation masks where voxels are reassigned based on the first timepoint.
4. To visualize tracks, open and select one of the segmentation layers.
5. ```Alt+Click``` on a label to visualize the track of that selected organelle/branch across all timepoints.
   - If the segmentation labels are selected, it will generate tracks for all voxels in the selected timepoint only.
   - If the reassigned labels are selected, it will generate tracks for all voxels across all timepoints.
   - *Note: If you have a 3D image, toggle to 2D mode via the ```Toggle 2D/3D view``` at the bottom left before ```Alt+Click```ing (eventually I'll get it to work while in 3D mode).

### Using Nellie's analysis plugin


https://github.com/aelefebv/nellie/assets/26515909/b93f90b3-53bd-40da-a8bf-a9a5c9c1cf4a


1. Follow the previous processing steps, you only need to do this once per file as long as you don't move or delete the files.
2. Open the ```Analyze``` tab, select the hierarchy level you want to visualize from the dropdown.
3. Select the level-specific feature you want to visualize from the new dropdown.
4. A histogram of all the data will be displayed.
   - This histogram can be directly exported via the ```Save graph``` button. A .png will be saved to ```[image_directory]/nellie_output/graphs/``` with the current datetime.
   - The values of the histogram can be exported via the ```Export graph data``` button. A .csv will be saved to ```[image_directory]/nellie_output/graphs/``` with the current datetime.
   - The histogram's x-axis can be viewed in log10 scale via the ```Log scale``` checkbox.
   - By default, the histogram shows lines at the mean +/- 1 standard deviation. This can instead be switched to median and quartiles via the ```Median view``` checkbox.
5. Press the ```Overlay mask``` button to colormap the organelle mask based on your selected feature.
   - Once overlaid, toggle the ```Timepoint data``` checkbox to allow you to select a specific timepoint to visualize via the slider.

## Other features
- Nellie's plugin offers an ```Easy screenshot``` feature:
  - Press the button under ```Easy screenshot``` or hit Ctrl/Cmd-Shift-E after clicking your image.
  - The .png will be saved to ```[image_directory]/nellie_output/screenshots/``` with the current datetime.
