
from setuptools import setup, find_packages

from distutils.util import convert_path


with open("README.md","r") as f:
    long_description = f.read()
version_ns = {}
vpath = convert_path('phase_STEM/_version.py')
with open(vpath) as version_file:
    exec(version_file.read(), version_ns)
setup(
    name='phase_STEM',
    version=version_ns['__version__'],
    packages= find_packages(),
    description='An open source python package for processing and analysis of 4D-STEM datasets, especially for reconstructing phase contrast images. It was completed when Dr. Xia worked in Prof. Tom Willhammar group at the Stockholm University',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=' ',
    author='Yu Xia',
    author_email='yu.xia1989@outlook.com',
    license='GNU GPLv3',
    keywords="4D-STEM, phase-contrast image reconstruction, image post-processings",
    python_requires='>=3.7',
    install_requires=[
        'numpy >= 1.19',
        'numba >= 0.57.1',
        'h5py >= 3.9.0',
        'cupy >= 12.2.0',
        'matplotlib >= 3.8.0',
        'hyperspy >= 1.7.5',
        'ipywidgets >= 8.0.4',
        'PyQt5 >= 5.10',
        'pyqtgraph >= 0.11',
        'tqdm >= 4.44.1',
        'pillow >= 9.4.0',
        'scipy >= 1.10.1',
        'matplotlib_scalebar>=0.8.1'
        ],
    
)
