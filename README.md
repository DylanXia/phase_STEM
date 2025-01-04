Dr. Yu Xia developed this open-source Python package for processing and analyzing STEM (scanning transmission electron microscopy) images, with a particular focus on reconstructing phase-contrast images through DPC (differential phase contrast) images. This project was completed during Dr. Yu Xia’s tenure at Prof. Tom Willhammar’s research group at the Stockholm University.

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
### Updates:
v0.0.1: 1. the reconstruction procedure is included
        2. integrating HR(S)TEM filter,
           # HR(S)TEM filter
           ## Introduction
`hrtem_filter` provides a set of python functions to denoise HR(S)TEM images. A Wiener filter and an average background subtraction filter were designed based on __R. Kilaas J. Microscopy, 1998, 190, 45-51__. Some steps are adopted from D. R. G. Mitchell's script for GMS. A nonlinear filter was adopted by the "non-linear filter plugin from GMS", originally developed by Dr. Hongchu Du. Refer to the original paper: __Hongchu Du, A nonlinear filtering algorithm for denoising HR(S)TEM micrographs, Ultramicroscopy 2015, 151, 62-67__


v.0.0.2: 1. adding a new submodule "io" for loading and saving data;
v.0.0.3: adding "denoising" 
v.0.0.4: rewrite the core algorithms and extending the functions to fit the pixelated detector(camera)



