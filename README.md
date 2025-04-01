# Nellie
## Automated organelle segmentation, tracking, and hierarchical feature extraction in 2D/3D live-cell microscopy

<img src="https://github.com/aelefebv/nellie/assets/26515909/96b7a113-be60-4028-bcd9-b444bdb943f6" width="200px" align="left" /> *Nature Methods (2025)* 

  [Article Link](https://www.nature.com/articles/s41592-025-02612-7) | [Cite](#reference)

**Abstract:** Cellular organelles undergo constant morphological changes and dynamic interactions that are fundamental to cell homeostasis, stress responses and disease progression. Despite their importance, quantifying organelle morphology and motility remains challenging due to their complex architectures, rapid movements and the technical limitations of existing analysis tools. Here we introduce Nellie, an automated and unbiased pipeline for segmentation, tracking and feature extraction of diverse intracellular structures. Nellie adapts to image metadata and employs hierarchical segmentation to resolve sub-organellar regions, while its radius-adaptive pattern matching enables precise motion tracking. Through a user-friendly Napari-based interface, Nellie enables comprehensive organelle analysis without coding expertise. We demonstrate Nellie’s versatility by unmixing multiple organelles from single-channel data, quantifying mitochondrial responses to ionomycin via graph autoencoders and characterizing endoplasmic reticulum networks across cell types and time points. This tool addresses a critical need in cell biology by providing accessible, automated analysis of organelle dynamics. 

**Nellie's pipeline and Napari plugin are both very much in early stages,** therefore [I highly encourage any and all feedback](#getting-help).

## Example output intermediates

https://github.com/aelefebv/nellie/assets/26515909/1df8bf1b-7116-4d19-b5fb-9658f744675b

## Installation (~ 1 minute)

**Notes:** 
- It is recommended (but usually not required) to create a new environment [via venv](https://docs.python.org/3/library/venv.html) or [via conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#creating-environments) before installing Nellie to avoid conflicts with other packages.
- May take several minutes to install.
- Choose one of the following methods, and only one!

### Step 0: If you do not already have Python 3.10 or higher installed, download it via the [python website](https://www.python.org/downloads/):

https://github.com/user-attachments/assets/50b1cd4b-6df7-4f19-8db3-4dcc03388513

### Option 1. If you already have Napari installed:
1. Open Napari
2. Go to ```Plugins > Install/Uninstall Plugins...```
3. Search for Nellie and click ```Install```
4. Make sure Nellie is updated to the latest version.
5. Restart Napari.


https://github.com/user-attachments/assets/0d44abe5-f575-4bd4-962a-2c102faf737c


### Option 2. If you don't have Nellie installed or Option 1 didn't work:
1. Open up Terminal (or Powershell on Windows)
- (optional but recommended) Create and activate a new [Python](https://docs.python.org/3/library/venv.html) or [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#creating-environments) environment.
2. pip install nellie:
```bash
python3 -m pip install nellie
```


https://github.com/user-attachments/assets/b63df093-e3e1-49cb-925b-7efce36b9015


#### Option 2a for NVIDIA GPU acceleration, optional (Windows, Linux):
To use GPU acceleration via NVIDIA GPUs, you also need to install cupy:
```bash
python3 -m pip install cupy-cudaXXx
```
- replace ```cupy-cudaXXx``` with the [appropriate version](https://docs.cupy.dev/en/stable/install.html#installing-cupy) for your CUDA version.
  - i.e. ```cupy-cuda11x``` for CUDA 11.x or ```cupy-cuda12x``` for CUDA 12.x
- if you don't have CUDA installed, [go here](https://docs.cupy.dev/en/stable/install.html).
- Mac Metal GPU-acceleration coming... eventually. Let me know if this is important to you!

## Usage
The sample dataset shown below is in the repo if you want to play around without, and can be downloaded [here](https://github.com/aelefebv/nellie/tree/main/sample_data).

### General data preparation
- It is strongly recommended to have your data in a parsable format, such as .ome.tif, .nd2, or other raw data files from microscopes.
  - Importing into ImageJ/FIJI and saving via BioFormats with the proper image dimensions should do the trick.
  - If the metadata cannot be parsed, you will have to manually enter it.
- It is also recommended to crop your image as much as possible to reduce processing time and memory usage. But really, unless you have massive lightsheet data, it should be pretty fast (seconds to minutes on a typical modern desktop computer).

### 3D + Timeseries dataset

https://github.com/user-attachments/assets/531f76ee-f58e-4058-b5dc-4fdf09af3660

### 3D (no Timeseries) dataset

https://github.com/user-attachments/assets/30d55bfa-bade-4987-88f0-255bb36cb7e8

### 2D + Timeseries dataset

https://github.com/user-attachments/assets/d534c6e1-df31-4964-9c12-edff56228be3

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
   - You can preview your image via the ```Open preview``` button once the metadata is filled in to ensure it looks correct.
   - From this tab, you can also choose what time points and channel you want to analyze, if your file contains more than one slice in those dimensions.
4. Click the ```Process``` tab.
5. You can run the full pipeline with ```Run Nellie```, or run individual steps below.
    - Steps can only be run once its previous step has been run.
    - Likewise, visualizations in the ```Visualization``` tab can only be opened once its respective step has been run.
6. All intermediate files and output csvs will be saved to ```[image_directory]/nellie_output/```, which can be accessed via the ```Open output directory``` button.
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
5. To visualize all tracks of all organelles/branches, click the ```Visualize all frame labels' tracks``` button.
6. To visualize all tracks of a specific organelle/branch:
   1. Click on the layer, and use the eyedropper tool at the top to select an organelle/branch to track.
   2. Click the ```Visualize selected label's tracks```.

### Using Nellie's analysis plugin
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

## Feedback / Getting Help
A few options are available for providing feedback or getting help with Nellie:

[Github Issues](https://github.com/aelefebv/nellie/issues/new) | [email](mailto:austin.e.lefebvre+nellie@gmail.com) | [X](https://twitter.com/Austin_Lefebvre) | wherever else you can find me!

To avoid any unnecessary back-and-forth, please include any/all (if possible) of the following information in your bug report:
- What kind of computer do you have, and what are its specs?
- Send me screenshots of what is not working.
- Send me any error logs in your terminal.
- Send me the file you ran (if possible).
- Any other information that might be helpful

## Other Info
For a 16bit dataset, the output:input ratio is ~15x. There is an option in the GUI to automatically delete intermediates after processing, keeping only the CSV files containing the extracted features.

## Requirements
Nellie has been tested on the following configurations:
- Mac, Linux, and Windows operating systems
- Python >= 3.10

## License
Nellie © 2024 by [Austin E. Y. T. Lefebvre](https://github.com/aelefebv) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Reference
If you used Nellie or found this work useful in your own research, please cite our [Nature Methods Paper](https://www.nature.com/articles/s41592-025-02612-7):

```
Lefebvre, A.E.Y.T., Sturm, G., Lin, TY. et al. Nellie: automated organelle segmentation, tracking and hierarchical feature extraction in 2D/3D live-cell microscopy. Nat Methods (2025). https://doi.org/10.1038/s41592-025-02612-7
```

## More fun examples
### Microtubule growing ends:

https://github.com/aelefebv/nellie/assets/26515909/88578dc9-f5c5-4188-a0e2-4e37037a44a9

### Endoplasmic reticulum:

https://github.com/aelefebv/nellie/assets/26515909/db76d388-a9cc-4650-b93d-69d357ace418

### Peroxisomes:

https://github.com/aelefebv/nellie/assets/26515909/58bda3cb-6489-4620-8584-a3728cd6b2ec

## Code contents:
Full documentation can be found within the code, and compiled by Sphinx in the file docs/_build/html/index.html

### Nellie pipeline
All the Nellie pipeline code is found within the nellie folder
- File and metadata loading, and file preparation is found at nellie/im_info/verifier.py
- Preprocessing is found at nellie/segmentation/filtering.py
- Segmentation of organelles is found at nellie/segmentation/labelling.py
- Skeletonization and segmentation of branches is found at nellie/segmentation/networking.py
- Mocap marker detection is found at nellie/segmentation/mocap_marking.py
- Mocap marker tracking is found at nellie/tracking/hu_tracking.py
- Voxel reassignment via flow interpolation is found at nellie/tracking/voxel_reassignment.py
- Hierarchical feature extraction is found at nellie/feature_extraction/hierarchical.py

### Nellie Napari plugin
All the Napari plugin code is found with the nellie_napari folder
- The home tab is found at nellie_napari/nellie_home.py
- The file selection tab is found at nellie_napari/nellie_fileselect.py
- The processing tab is found at nellie_napari/nellie_processor.py
- The visualization tab is found at nellie_napari/nellie_visualizer.py
- The analysis tab is found at nellie_napari/nellie_analysis.py
- The settings tab is found at nellie_napari/nellie_settings.py
