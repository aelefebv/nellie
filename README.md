# Nellie
## Automated organelle segmentation, tracking, and hierarchical feature extraction in 2D/3D live-cell microscopy

<img src="https://github.com/aelefebv/nellie/assets/26515909/96b7a113-be60-4028-bcd9-b444bdb943f6" width="200px" align="left" /> *arXiv* 

  [Preprint Link](https://arxiv.org/abs/2403.13214) | [Cite](#reference)

**Abstract:** The analysis of dynamic organelles remains a formidable challenge, though key to understanding biological processes. We introduce Nellie, an automated and unbiased pipeline for segmentation, tracking, and feature extraction of diverse intracellular structures. Nellie adapts to image metadata, eliminating user input. Nellie’s preprocessing pipeline enhances structural contrast on multiple intracellular scales allowing for robust hierarchical segmentation of sub-organellar regions. Internal motion capture markers are generated and tracked via a radius-adaptive pattern matching scheme, and used as guides for sub-voxel flow interpolation. Nellie extracts a plethora of features at multiple hierarchical levels for deep and customizable analysis. Nellie features a Napari-based GUI that allows for code-free operation and visualization, while its modular open-source codebase invites customization by experienced users. 

**Nellie's pipeline and Napari plugin are both very much in early stages,** therefore [I highly encourage any and all feedback](#getting-help).

## Example output intermediates

https://github.com/aelefebv/nellie/assets/26515909/1df8bf1b-7116-4d19-b5fb-9658f744675b

## Installation

**Notes:** 
- It is recommended (but usually not required) to [create a new environment](https://docs.python.org/3/library/venv.html) for Nellie to avoid conflicts with other packages.
- May take several minutes to install.
- Choose one of the following methods, and only one!
### Option 1. Via Napari plugin manager:
If not already installed, install Napari: https://napari.org/stable/tutorials/fundamentals/installation
1. Open Napari
2. Go to ```Plugins > Install/Uninstall Plugins...```
3. Search for Nellie and click ```Install```
### Option 2. Via PIP:
```bash
pip install nellie
```
#### Option 2a for NVIDIA GPU acceleration, optional (Windows, Linux):
To use GPU acceleration via NVIDIA GPUs, you also need to install cupy:
```bash
pip install cupy-cudaXXx
```
- replace ```cupy-cudaXXx``` with the [appropriate version](https://docs.cupy.dev/en/stable/install.html#installing-cupy) for your CUDA version.
  - i.e. ```cupy-cuda11x``` for CUDA 11.x or ```cupy-cuda12x``` for CUDA 12.x
- if you don't have CUDA installed, [go here](https://docs.cupy.dev/en/stable/install.html).
- Mac Metal GPU-acceleration coming... eventually.

## Usage
The sample dataset shown below is in the repo if you want to play around without, and can be downloaded [here](https://github.com/aelefebv/nellie/tree/main/sample_data).

https://github.com/aelefebv/nellie/assets/26515909/05199fed-ed8c-4237-b3ba-0a3f4cdcb337

### General data preparation
- It is strongly recommended to have your data in a parsable format, such as .ome.tif, .nd2, or other raw data files from microscopes.
  - Importing into ImageJ/FIJI and saving via BioFormats with the proper image dimensions should do the trick.
  - If the metadata cannot be parsed, you will have to manually enter it.
- It is also recommended to crop your image as much as possible to reduce processing time and memory usage. But really, unless you have massive lightsheet data, it should be pretty fast.

https://github.com/aelefebv/nellie/assets/26515909/372d07a8-15a0-4926-8594-108dd4b97280

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
   - If the reassigned labels are selected, you can choose to generate tracks for all voxels across all timepoints.
   - You can skip voxels to track so that the area is not too crowded by tracks.
   - *Note: If you have a 3D image, toggle to 2D mode via the ```Toggle 2D/3D view``` at the bottom left before ```Alt+Click```ing (eventually I'll get it to work while in 3D mode).

### Using Nellie's analysis plugin

https://github.com/aelefebv/nellie/assets/26515909/7f4f09a4-3687-4635-988d-e1d16ad2a4af

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

## License
Nellie © 2024 by [Austin E. Y. T. Lefebvre](https://github.com/aelefebv) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

## Reference
If you used Nelly or found this work useful in your own research, please cite our [arXiv preprint](https://arxiv.org/abs/2403.13214):

Lefebvre, A. E. Y. T., Sturm, G., et. al. Nellie: Automated organelle segmentation, tracking, and hierarchical feature extraction in 2D/3D live-cell microscopy, arXiv, 2024, https://arxiv.org/abs/2403.13214

```
@misc{lefebvre2024nellie,
      title={Nellie: Automated organelle segmentation, tracking, and hierarchical feature extraction in 2D/3D live-cell microscopy}, 
      author={Austin E. Y. T. Lefebvre and Gabriel Sturm and Ting-Yu Lin and Emily Stoops and Magdalena Preciado Lopez and Benjamin Kaufmann-Malaga and Kayley Hake},
      year={2024},
      eprint={2403.13214},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## More fun examples
### Microtubule growing ends:

https://github.com/aelefebv/nellie/assets/26515909/88578dc9-f5c5-4188-a0e2-4e37037a44a9

### Endoplasmic reticulum:

https://github.com/aelefebv/nellie/assets/26515909/db76d388-a9cc-4650-b93d-69d357ace418

### Peroxisomes:

https://github.com/aelefebv/nellie/assets/26515909/58bda3cb-6489-4620-8584-a3728cd6b2ec

