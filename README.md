# Nellie
## Installation
*Note: May take several minutes*

### Installation CPU (Mac, Windows, Linux):
```bash
pip install git+https://github.com/aelefebv/nellie.git
```
### Installation NVIDIA GPU (Windows, Linux):
- replace cupy-cuda12x with the appropriate version for your CUDA version (https://docs.cupy.dev/en/stable/install.html)
```bash
pip install cupy-cuda12x git+https://github.com/aelefebv/nellie.git 	
```
Mac Metal GPU-acceleration coming... eventually.

## Usage
### Running Nellie's processing pipeline
1. Start Napari (open a Terminal and type napari)
    ```bash
    napari
    ```
2. Go to 
```Plugins > Nellie (nellie)```
3. Select a file to process and/or analyze or a folder to batch process.
   - if you have multiple fluorescence channels, select the channel you want to process/analyze.
   - if the metadata does not fill in, it did not detect any from your image, and you must manually enter it or reformat your image and try again.
4. Click ```Open Nellie Processor``` to open the processor window.
5. You can run the full pipeline with ```Run Nellie```, or run individual steps below.
    - Steps can only be run once its previous step has been run.
    - Likewise, visualizations can only be opened once its respective step has been run.
6. Once features have been exported, Nellie will automatically detect this, and allow analysis to be run via:
   - ```Open Nellie Analyzer``` from the original or processor window.
   - Analysis at this point is optionally, but can be helpful for visualizing, and selectively exporting data.

### Using Nellie's analysis plugin
1. Follow the previous processing steps, you only need to do this once per file.
2. Once the ```Nellie Analyzer``` window is open, select the hierarchy level you want to visualize from the dropdown.
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
  - The screenshot will be saved to ```[image_directory]/nellie_output/screenshots/``` with the current datetime.
