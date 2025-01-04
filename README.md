Dr. Yu Xia developed this open-source Python package, with the funding support by Carl Trygger Foundation, for processing and analyzing STEM (scanning transmission electron microscopy) images, with a particular focus on reconstructing phase-contrast images through DPC (differential phase contrast) images. This package also can reconstruct phase-contrast images through 4D-STEM datasets.

This project was completed at Prof. Tom Willhammar’s research group at the Stockholm University.

Currently, this module has been tested using the datasets acquired by machine FEI Titan Themis S/TEM series , e.g. Z and 80-300.

The datasets with .emd format is used as the input, which is loaded by "rosettasciio" or "hyperSpy.api".

The GUI of phase_STEM has been developed using PyQt5.

The CUDA is employed to accelerate the calculation, which the cupy and numba are required.

If phase_STEM helps you in your research, please cite it (a paper is preparing).
