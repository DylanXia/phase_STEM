Dr. Yu Xia created this open-source Python package for the processing and analysis of Scanning Transmission Electron Microscopy (STEM) images. A key feature is the reconstruction of phase-contrast images from STEM dataset. This work was conducted while Dr. Yu Xia worked in Prof. Tom Willhammar’s research group at Stockholm University.

We encourage you to cite our paper, "Introducing the phase_STEM package: A Comparative Analysis of iDPC, First-Moment, and OBF Techniques", if this package proves useful in your work.

Datasets for our paper are available on Zenodo at https://zenodo.org/records/15182064

The package has been validated using datasets acquired from FEI Titan Themis S/TEM instruments, including the Z and 80-300 models. It accepts .emd format files as input, which are loaded using the "rosettasciio" library. For accelerated computation, CUDA is utilized, requiring the installation of cupy and numba.







