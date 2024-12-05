# phase_STEM
phase_STEM is a Python's package for reconstructing phase image from datasets captured by segmented detector or pixelated detector using scanning transmission electron microscopy.

Dr. Xia Yu developed this open-source Python package for processing and analyzing STEM (scanning transmission electron microscopy) images, with a particular focus on reconstructing phase-contrast images through DPC (differiential phase contrast) images. This project was completed during Dr. Xia’s tenure at Dr. Willhammar’s research group at the Stockholm University.

Currently, this module has been tested using the datasets acquired by machine FEI Titan Themis S/TEM series , e.g. Z and 80-300.

The datasets with .emd format is used as the input, which is loaded by "rosettasciio" or "hyperSpy.api".

The CUDA is employed to accelerate the calculation, which the cupy and numba are required.

### Dependencies

* numpy
* cupy
* cupyx.scipy
* scipy
* h5py
* matplotlib.pyplot
* matplotlib_scalebar
* numba
* math
* cmath
* hyperspy
* scikit-learn
* matplotlib.colors.hsv_to_rgb
* ipywidgets
* tqdm
* pillow
* panda




