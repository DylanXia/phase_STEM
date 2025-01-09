Dr. Yu Xia developed this open-source Python package for processing and analyzing STEM (scanning transmission electron microscopy) images, with a particular focus on reconstructing phase-contrast images through DPC (differential phase contrast) images. This project was completed during Dr. Yu Xia’s tenure at Prof. Tom Willhammar’s research group at the Stockholm University.

Currently, this module has been tested using the datasets acquired by machine FEI Titan Themis S/TEM series , e.g. Z and 80-300.

The datasets with .emd format is used as the input, which is loaded by "rosettasciio" or "hyperSpy.api".

The CUDA is employed to accelerate the calculation, which the cupy and numba are required.
