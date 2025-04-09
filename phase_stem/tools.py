"""
It is the sub-module of phase-STEM for loading raw datasets, displaying results as figures, and saving results.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib_scalebar.scalebar import ScaleBar
import cupy as cp
import numpy as np
#import hyperspy.api as hs #load dataset 
from rsciio.emd import file_reader as emd_reader
from rsciio.digitalmicrograph import file_reader as dm_reader
from rsciio.mrc import file_reader as mrc_reader
from rsciio.tia import file_reader as tia_reader

from PIL import Image
from skimage.segmentation import watershed
from ipywidgets import interact, widgets
import h5py
import cv2
from skimage.feature import peak_local_max, blob_log
from skimage import exposure , measure
from skimage.registration import phase_cross_correlation
from tqdm import tqdm
from scipy import ndimage
import cupyx.scipy.ndimage as cpndimage
from typing import Optional, Tuple, List
import pyfftw
from phase_STEM import analysis, EMFilters, io

class ImageProcessor:
    """
    A class to process images by loading data from various file formats, applying filters,
    plotting images, and saving them to HDF5 files.
    This class only can process one file at a time.

    Example: 
    if __name__ == "__main__":
        path = "path/image.emd"  # Replace with actual file path
        processor = ImageProcessor(path)
        processor.plot_images(colormap='viridis', saving=os.path.splitext(path)[0])
        processor.save_to_h5(saving=os.path.splitext(path)[0])
    """
    
    def __init__(self, file_path, resolution=None, unit=None):
        """
        Initialize the ImageProcessor with a file path and load the data.

        Args:
            file_path (str): Path to the data file.
            resolution (float): Resolution of the images.
            unit (str): Unit of the resolution.
        """
        self.file_path = file_path
        self.images = {}  # Dictionary to store image data
        self.resolution = resolution  # Resolution of the images
        self.unit = unit  # Unit of the resolution
        self.mode = 'Probe'
        self.load_data()

    def load_data(self):
        """
        Load data from the file and populate images, resolution, and unit attributes.
        
        Raises:
            ValueError: If the file format is not supported.
        """
        file_extension = os.path.splitext(self.file_path)[1][1:].lower()
        self.mode = image_mode(self.file_path)
        if file_extension == 'emd':
            data = emd_reader(self.file_path)
            if self.resolution is None:
                self.resolution = data[0]['axes'][0]['scale']
            if self.unit is None:
                self.unit = data[0]['axes'][0]['units']
            self.images = {item['metadata']['General']['title']: item['data'] for item in data}
        
        elif file_extension in ('dm3', 'dm4'):
            data = dm_reader(self.file_path)
            if self.resolution is None:
                self.resolution = data[0]['axes'][0]['scale']
            if self.unit is None:
                self.unit = data[0]['axes'][0]['units']
            self.images = {item['metadata']['General']['title']: item['data'] for item in data}
        
        elif file_extension == 'mrc':
            data = mrc_reader(self.file_path)
            if self.resolution is None:
                self.resolution = 1
            if self.unit is None:
                self.unit = 'px'
            self.images = {item['metadata']['General']['title']: item['data'] for item in data}
        
        elif file_extension in ('ser', 'emi'):
            data = tia_reader(self.file_path)
            if self.resolution is None:
                self.resolution = data[0]['axes'][0]['scale']
            if self.unit is None:
                self.unit = data[0]['axes'][0]['units']
            self.images = {item['metadata']['General']['title']: item['data'] for item in data}
        
        elif file_extension in ('h5', 'hdf5'):
            self.images = io.h5_reader(self.file_path)
            if self.resolution is None:
                self.resolution = 1
            if self.unit is None:
                self.unit = 'px'
        
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")


    def plot_images(self, colormap='grey', saving=None):
        """
        Plot images from the loaded data with optional saving.

        Args:
            colormap (str): Colormap for plotting (default: 'grey').
            saving (str or None): Directory to save images; if None, images are not saved.
        """
        if saving and not os.path.exists(saving):
            os.makedirs(saving)

        excluded = {'dDPC', 'DF4_A', 'DF4_B', 'DF4_C', 'DF4_D', 'A-C', 'B-D'}
        i = 0
        if self.unit == "Å":
            self.unit = "nm"
            self.resolution *= 0.1
        if self.mode == 'DIFFRACTION' :
            unit_type = 'si-length-reciprocal' 
            self.unit = f"1/{self.unit}"
        else:
            unit_type ='si-length'
        for title, image in self.images.items():
            if i >= 50:
                break
            if title == 'iDPC' or title not in excluded:
                if title == 'iDPC':
                    image = EMFilters.gaussian_bandpass(image, space='real', highpass=True, cutoff_ratio=0.01)
                
                fig, ax = plt.subplots(figsize=(6, 6))
                if self.mode != 'DIFFRACTION':
                    ax.imshow(image, cmap=colormap)
                else:
                    ax.imshow(np.log(np.abs(image)+1), cmap=colormap)
                scale_bar = ScaleBar(
                    self.resolution,
                    units = self.unit,
                    dimension = unit_type,
                    length_fraction = 0.2,
                    location ='lower left',
                    scale_loc ='top'
                )
                ax.add_artist(scale_bar)
                ax.set_title(title)
                ax.axis('off')
                plt.tight_layout()
                
                if saving:
                    plt.savefig(os.path.join(saving, f"{title}.png"), format='png', dpi=600)
                plt.show()
                i += 1

    def save_to_h5(self, saving='path'):
        """
        Save images to an HDF5 file, applying a filter to 'iDPC' images.

        Args:
            saving (str): Base path for the HDF5 file (without extension).
        """
        with h5py.File(f"{saving}.h5", 'w') as data:
            group_name = f"{self.resolution}{self.unit}"
            reconstruct = data.create_group(group_name)
            for title, image in self.images.items():
                if title == 'iDPC':
                    filtered_image = EMFilters.gaussian_bandpass(image, space='real', highpass=True, cutoff_ratio=0.01)
                    reconstruct.create_dataset("High-pass filtered iDPC", data=filtered_image)
                else:
                    reconstruct.create_dataset(title, data=image)


def diagonal_split(img):
    '''
    This function takes an input image and splits it diagonally into four sub-regions.
    The input image must have dimensions that are divisible by 4.
    The resolution of the splitted image becomes sqrt(2) time of original image.
    '''
    # Get the shape of the input image
    h, w = img.shape
    # Check that the image has dimensions divisible by 4
    if (h % 4 != 0) or (w % 4 != 0):
        raise ValueError('Input image must have dimensions divisible by 4')
    # Crop the image to make sure the dimensions are divisible by 4
    img = img[:h//4*4, :w//4*4]
    h, w = img.shape
    # Create indices for the rows and columns
    row_indices = np.arange(h)
    col_indices = np.arange(w)

    # Split the indices into two groups, one for each diagonal split
    row_split_u = row_indices[::2]
    row_split_d = row_indices[1::2]

    col_split_l = col_indices[::2]
    col_split_r = col_indices[1::2]

    # Split the image into four sub-regions using advanced indexing
    split_images = np.zeros((4, h//2, w//2))
    sub_a1 = img[np.ix_(row_split_u, col_split_l)]
    sub_a2 = img[np.ix_(row_split_d, col_split_r)]
    sub_b1 = img[np.ix_(row_split_d, col_split_l)]
    sub_b2 = img[np.ix_(row_split_u, col_split_r)]
    split_images[0] = sub_a1
    split_images[1] = sub_a2
    split_images[2] = sub_b1
    split_images[3] = sub_b2
    # Return the four sub-regions
    return split_images



class FourierRingCorrelation(object):
    """
    A class for calculating 2D Fourier ring correlation. Calculates a 2D polar coordinate
    centered at the geometric center of the data shape, and contains methods to calculate
    the FRC as well as to plot the results.
    """

    def __init__(self, image1, image2, d_bin):
        """
        :param image1: First 2D image for FRC calculation
        :param image2: Second 2D image for FRC calculation
        :param d_bin: Thickness of the ring in pixels
        """
        if image1.shape != image2.shape:
            raise ValueError("The image dimensions do not match")
        if image1.ndim != 2:
            raise ValueError("Fourier ring correlation requires 2D images.")
        
        self.image_shape = image1.shape
        self.d_bin = d_bin

        # Initialize FourierRingIterator variables
        self._nbins = int(np.floor(self.image_shape[0] / (2 * self.d_bin)))
        self.freq_nyq = int(np.floor(self.image_shape[0] / 2.0))
        self._radii = np.arange(0, self.freq_nyq, self.d_bin)

        # Create meshgrid and radius matrix
        axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in self.image_shape)
        y, x = np.meshgrid(*axes)
        self.r = np.sqrt(x ** 2 + y ** 2)

        # Initialize FRC variables
        self.fft_image1 = np.fft.fftshift(np.fft.fft2(image1))
        self.fft_image2 = np.fft.fftshift(np.fft.fft2(image2))

    @property
    def radii(self):
        return self._radii

    @property
    def nbins(self):
        return self._nbins

    def get_points_on_ring(self, ring_start, ring_stop):
        return (self.r >= ring_start) & (self.r < ring_stop)

    def execute(self):
        """
        Calculate the FRC
        :return: Returns the FRC results.
        """
        radii = self.radii
        nbins = self.nbins
        c1 = np.zeros(nbins, dtype=np.float32)
        c2 = np.zeros(nbins, dtype=np.float32)
        c3 = np.zeros(nbins, dtype=np.float32)
        points = np.zeros(nbins, dtype=np.float32)

        for idx in range(nbins):
            ring = self.get_points_on_ring(idx * self.d_bin, (idx + 1) * self.d_bin)
            subset1 = self.fft_image1[ring]
            subset2 = self.fft_image2[ring]
            c1[idx] = np.sum(subset1 * np.conjugate(subset2)).real
            c2[idx] = np.sum(np.abs(subset1) ** 2)
            c3[idx] = np.sum(np.abs(subset2) ** 2)
            points[idx] = subset1.size

        # Calculate spatial frequencies and FRC
        spatial_freq = radii.astype(np.float32) / self.freq_nyq
        with np.errstate(divide="ignore", invalid="ignore"):
            frc = np.abs(c1) / np.sqrt(c2 * c3)
            frc[~np.isfinite(frc)] = 0.0  # Replace inf and NaN with 0.0

        data_set = {
            "correlation": frc,
            "frequency": spatial_freq,
            "points-x-bin": points
        }
        return data_set



def align_images(images, space='real', mask_center=(300, 1450), mask_size=(10, 256), upsampling = 1):
    """
    Perform image registration using skimage.registration.phase_cross_correlation.
    
    Args:
        images: np.ndarray, with shape (num, pixel_x, pixel_y)
        space: string, if 'images' are in real space, use 'real'; 
               if they are Fourier transformed, use 'Fourier'.
        mask_center: tuple, the coordinate of the mask center in pixels, like (300, 1450)
        mask_size: tuple, the height and width of the mask, like (256, 256)
    
    Returns:
        aligned_images: np.ndarray of aligned images.
    """
    if images.ndim != 3:
        raise TypeError('Image registration needs an image stack')

    num, px, py = images.shape
    values = np.zeros(num)
    
    # Create the mask
    mask = np.zeros((px, py), dtype=np.uint8)
    half_mask_size = (mask_size[0] // 2, mask_size[1] // 2)
    
    row1 = max(0, mask_center[0] - half_mask_size[0])
    row2 = min(px, mask_center[0] + half_mask_size[0])
    col1 = max(0, mask_center[1] - half_mask_size[1])
    col2 = min(py, mask_center[1] + half_mask_size[1])
    
    if max(mask_center) >= max(px, py) or row2 <= row1 or col2 <= col1:
        raise ValueError("Invalid mask parameters. Check mask center or mask size.")
    
    mask[row1:row2, col1:col2] = 1
    
    # Calculate the standard deviation of the real part for each image
    for i in range(num):
        values[i] = np.std(images[i].real)
    
    # Choose the image with the highest variance as the reference
    index = np.argmax(values)
    aligned_images = []
    
    # Real-space alignment
    if space == 'real':
        reference = images[index]
        aligned_images.append(reference)
        
        for i in range(num):
            if i != index:
                shift_vector, _, _ = phase_cross_correlation(reference * mask, images[i] * mask, upsample_factor=upsampling, space = "real")
                aligned_image = ndimage.shift(images[i], shift_vector)
                aligned_images.append(aligned_image)
    
    # Fourier-space alignment
    elif space in ['Fourier', 'Fourier space', 'Reciprocal space', 'Reciprocal']:
        reference = pyfftw.interfaces.numpy_fft.ifft2(pyfftw.interfaces.numpy_fft.ifftshift(images[index]))
        aligned_images.append(reference.real)
        for i in range(num):
            if i != index:
                compare = pyfftw.interfaces.numpy_fft.ifft2(pyfftw.interfaces.numpy_fft.ifftshift(images[i]))
                shift_vector, _, _ = phase_cross_correlation(reference * mask, compare * mask, upsample_factor=upsampling, space = "fourier")
                aligned_image = ndimage.shift(compare, shift_vector)
                aligned_images.append(aligned_image)
    
    else:
        raise ValueError("Please choose the correct space: 'real' or 'Fourier'.")
    
    return np.array(aligned_images)


class TemplateMatcher:
    """
    It provides an approach to use template matching for extracting the unit cell image of a crystalline sample 
    with high contrast.

    Usage example:

    matcher = ps.tools.TemplateMatcher(image)

    matcher.select_template(template_shape={"top_left":[678,1750], "height":256, "width":256})

    # Search for the template in the image
    matcher.search_template()

    # Get the matches with a correlation threshold of 0.8
    matches = matcher.get_matches(threshold=0.8)
    I = 1
    for pos, value, matched_area in matches:
        print(f"{I} Match at position: {pos}, Correlation value: {value}")
        I+=1

    stacks = matcher.stack_matches()

    matcher.display()

    plt.imshow(np.sum(stacks, axis =0))
    plt.axis('off')
    """
    def __init__(self, image):
        self.image = image.astype(np.float32)
        self.template = None
        self.correlation_map = None

    def select_template(self, template_shape):
        """
        Selects a template from the image based on the given top-left corner and dimensions.
        The size of template is designed in a dict with keys {'top_left': [10,10], 'height':10, 'width':10}
        """
        top_left = template_shape['top_left'].T
        height = template_shape['height']
        width = template_shape['width']
        
        self.template = self.image[top_left[0]:top_left[0] + height, top_left[1]:top_left[1] + width]

    def search_template(self, method = cv2.TM_CCOEFF_NORMED):
        """
        Searches for the selected template in the image and computes the result.
        There are four methods provided by cv2:
        methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
                  'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
        Reference:
        https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
        
        """
        self.method = method
        if self.template is None:
            raise ValueError("Template has not been selected.")
        self.result = cv2.matchTemplate(self.image, self.template, method)

    def get_matches(self, threshold=0.8):
        """Finds all positions with result values above the given threshold."""
        if self.result is None:
            raise ValueError("Result has not been computed.")
        locations = np.where(self.result >= threshold)
        
        matches = []
        h, w = self.template.shape
        for pt in zip(*locations[::-1]):    
            match_position = (pt[1], pt[0])            
            matched_area = self.image[match_position[0]:match_position[0] + h, match_position[1]:match_position[1] + w]
            correlation_value = self.result[pt[1], pt[0]]
            matches.append((match_position, correlation_value, matched_area))
        # Sort matches by correlation value in descending order
        matches.sort(key=lambda x: x[1], reverse=True)
        self.match = matches
        return matches
    def stack_matches(self):
        """
        return all matched sub_images in a np.ndarray with a shape of (N, x, y)
        """
        cropped_series = []
        for n in range (len(self.match)-1):
            cropped_series.append(self.match[n][2])
        return np.stack(cropped_series, axis =0)
    def display(self):
        """Displays the original image, template, and correlation map."""
        if self.template is None or self.result is None:
            raise ValueError("Template and correlation map must be computed before displaying results.")
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(self.image, cmap='gray')
        plt.title('Original Image')

        plt.subplot(1, 3, 2)
        plt.imshow(self.template, cmap='gray')
        plt.title('Template')

        plt.subplot(1, 3, 3)
        plt.imshow(self.result, cmap='hot')
        plt.title('Correlation Map')
        plt.colorbar()
        
        plt.show()


def contrast_ratio(image):
    if isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)
    sizeX, sizeY = image.shape
    max_intensity = np.max(image)
    min_intensity = np.min(image)
    mean_intensity = np.average(image)
    ratio = max_intensity / min_intensity if min_intensity !=0 else max_intensity
    if min_intensity <0:
        corr_image = image - min_intensity
        max_int = np.max(corr_image)
        mean_int = np.mean(corr_image)
        luminance_contrast = max_int/mean_int
    else: 
        luminance_contrast = (max_intensity - min_intensity)/mean_intensity 
    weber_contrast = (mean_intensity - min_intensity)/min_intensity if min_intensity !=0 else (mean_intensity - min_intensity)
    michelson_contrast = max_intensity /(max_intensity + 2*min_intensity)

    PSNR_contrast = 10*np.log10(max_intensity**2/(np.mean(np.square(image))))
    
    print(f"The minimum intensity is : {np.round(min_intensity, 2)}")
    print(f"The maximum intensity is : {np.round(max_intensity, 2)}")
    print(f"The mean intensity is : {np.round(mean_intensity, 2)}")
    print("The contrast of image is evaluated by: ")
    print(f"--> Max./Min. ratio: {np.round(ratio, 2)}")
    print(f"--> Luminance contrast: {np.round(luminance_contrast, 2)}")
    print(f"--> Weber contrast: {np.round(weber_contrast, 2)}")
    print(f"--> Michelson contrast: {np.round(michelson_contrast, 4)}")
    print(f"--> Peak SNR: {np.round(PSNR_contrast, 4)} dB")

def get_mssim(img1, img2, astype = np.float32, sigma=2, radius=1, C1 = 6.5025, C2 = 58.5225):
    
    """
    This code calculates the Mean Structural Similarity Index (MSSIM) between two images.
    """
    

    # Convert images to float32
    I1 = img1.astype(astype)
    I2 = img2.astype(astype)

    # Precompute squares and products
    I2_2 = I2 * I2
    I1_2 = I1 * I1
    I1_I2 = I1 * I2

    # Apply Gaussian blur
    mu1 = ndimage.gaussian_filter(I1, (sigma, sigma), radius)
    mu2 = ndimage.gaussian_filter(I2, (sigma, sigma), radius)

    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_2 = ndimage.gaussian_filter(I1_2, (sigma, sigma), radius) - mu1_2
    sigma2_2 = ndimage.gaussian_filter(I2_2, (sigma, sigma), radius) - mu2_2
    sigma12 = ndimage.gaussian_filter(I1_I2, (sigma, sigma), radius) - mu1_mu2

    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2

    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2

    ssim_map = t3 / t1

    # Compute mean of SSIM map
    mssim = np.mean(ssim_map)

    return mssim

def calculate_psnr(reference, experimental, data_range=None):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Parameters:
        reference (numpy.ndarray): The reference image.
        experimental (numpy.ndarray): The experimental (possibly distorted) image.
        data_range (float, optional): The data range of the input image (e.g., 255 for 8-bit images).
                                  If None, it's computed from the reference image.

    Returns:
        float: The PSNR value in decibels (dB).
    """
    if reference.shape != experimental.shape:
        raise ValueError("Input images must have the same dimensions.")

    if data_range is None:
        data_range = np.max(reference) - np.min(reference)

    mse = np.mean((reference - experimental) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    return psnr

def plot_img_and_hist(image, bins=256):
    """Plot an image along with its histogram and cumulative histogram."""
   
    if isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)
    
    fig, axes = plt.subplots(1, 2, figsize =(10, 5))
    ax_cdf = axes[1].twinx()

    # Display image
    axes[0].imshow(image, cmap=plt.cm.gray)
    axes[0].set_axis_off()

    # Display histogram
    axes[1].hist(image.ravel(), bins=bins, histtype='step', color='black')
    axes[1].ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    axes[1].set_xlabel('Pixel intensity')
    #axes[1].set_xlim(0, 1)
    axes[1].set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

def image_mode(file_path):
    """
    Determines the imaging mode of the dataset.
    
    Args:
        file_path (str): Path to the dataset file.
    
    Returns:
        str: The imaging mode.
    """
    file_extension = os.path.splitext(file_path)[1][1:].lower()
    
    if file_extension in ('dm3', 'dm4'):
        dm_data = dm_reader(file_path)
        return dm_data[0]['original_metadata']['ImageList']['TagGroup0']['ImageTags']['Microscope Info']['Operation Mode']
    elif file_extension == 'emd':
        emd_data = emd_reader(file_path)
        return emd_data[0]['original_metadata']['Optics']['IlluminationMode']
    
    return None
    

def show_FFT(img, resolution, unit="nm"):
    """
    It shows the Fourier transform of an image.
    If it returned, it ouputs the log(amplitute) of FFT.

    It requires the "%matplotlib inline" in jupytr notebook.
    """
    if isinstance(img, cp.ndarray):
        image = cp.asnumpy(img)
    else: image = img
        
    def show(image, sigma, alpha, zoom_in, draw_circle, output):
        sx, sy = image.shape
        bck_fft, image_fft = analysis.periodic_DFT(image, inverse_dft=False)
        img_fft = np.fft.fftshift(cp.asnumpy(image_fft - bck_fft))
        magnitude_fft =np.log(.1+np.abs(img_fft))
        magnitude_fft0 = ndimage.gaussian_filter(magnitude_fft, sigma=(sigma, sigma), order=0)
        magnitude_fft1 = ndimage.gaussian_filter(magnitude_fft0, sigma=(1,1), order=0)
        sharpened = magnitude_fft1 + alpha *(magnitude_fft0 - magnitude_fft1)
        display = linscale(crop_matrix(sharpened, (0,1),  [int(sx/2), int(sy/2)],[int(sx/zoom_in), int(sy/zoom_in)]))
        dx, dy = display.shape
        #pixe size in reciprocal space
        rec_resolution = 1/(resolution*sx)
        centre = np.array([int(dx/2), int(dy/2)])
        extend_edge = rec_resolution*dx/2
        extend = [-extend_edge, extend_edge, -extend_edge, extend_edge]
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(display, cmap=plt.cm.gray, extent=extend)
        radius = draw_circle*rec_resolution
        patch = plt.Circle((0,0), radius, color='yellow', linewidth=2, fill=False)
        ax.add_patch(patch)
        plt.title(f"Information resolution: {round(1/radius, 2)} {unit}")
        plt.axis('on')
        plt.ylabel(f"Field of view (1/{unit})")
        plt.tight_layout()
        plt.show()
        if output:
            return sharpened
    interact(show, 
            draw_circle = widgets.FloatSlider(min = 4, max = 1024, value = 512, layout=widgets.Layout(width='80%')),
            image=widgets.fixed(image),                        
            sigma = widgets.FloatSlider(min = 3, max = 10, value = 5),
            alpha=widgets.IntSlider(min = 10, max = 150, value = 50),
            zoom_in = widgets.FloatSlider(min = 1, max = 8, value = 2),
            output=widgets.Dropdown(options=[False, True]))    


def plot_vector_image(dataset, title, imgsize, storing = [False, "path"]):
    """
    Plot the vector images and optionally save them to the given path
    
    Args:
    dataset: vector image, consisting of real component and imaginary component
             dataset or the elements in the dataset should be numpy.array or cupy. array
             
    title: a list storing the name of images in dataset, which the first name is the whole dataset
           e.g. title = ['built frequency filter', 'filter_1', 'filter_2']
           The 'built frequency filter' is the name of dataset, while the 'filter_1' and 'filter_2' are the elements of data
           
    storing: a list [bool, path] where the first element indicates whether to save the plot and the second is the path
    """
    # Calculate the number of filters
    if not isinstance(dataset, np.ndarray):
        dataset = cp.asnumpy(dataset)
    if len(dataset.shape)==2:
        num_imgs = 1
    elif len(dataset.shape)==3:
        num_imgs = len(dataset)
    else: print("The size of the dataset is incorrect!")
    # Create a grid of subplots
    if num_imgs ==1:
        rows, cols = 1, 1
    else:
        rows, cols = 2, int((num_imgs+1)/2)
        
    fig, axes = plt.subplots(rows, cols, figsize=(imgsize, imgsize))
    if num_imgs == 1:
        axes = np.array([[axes]])
    # Iterate over the images and plot them in the grid
    for i in range(rows):
        for j in range (cols):
            ax = axes[i,j]
            if i*cols+j < num_imgs:
                if num_imgs ==1:
                    image = dataset
                else: image=dataset[i * cols + j]
                EMag = np.abs(image)
                EMagScale = EMag / np.amax(EMag)        
        # Calculate hue and value channels for the HSV representation
                hue = np.angle(image) / (2 * np.pi) % 1
                saturation = np.ones_like(hue)
                hsv_image = np.stack((hue, saturation, EMagScale), axis=-1)
                EDir = hsv_to_rgb(hsv_image)        
             
                LegPix = 360
                LegRad = 1
                x, y = np.meshgrid(np.linspace(-1, 1, LegPix, endpoint=True), np.linspace(-1, 1, LegPix, endpoint=True))
                X, Y = x * (x ** 2 + y ** 2 < LegRad ** 2), y * (x ** 2 + y ** 2 < LegRad ** 2)       
        # Calculate hue and value channels for the legend
                hue_leg = np.angle(X + 1j * Y) / (2 * np.pi) % 1
                saturation_leg = np.ones_like(hue_leg)
                RI = np.sqrt(X ** 2 + Y ** 2) / np.amax(np.sqrt(X ** 2 + Y ** 2))
                hsv_legend = np.stack((hue_leg, saturation_leg, RI), axis=-1)
        
                EDirLeg = hsv_to_rgb(hsv_legend)
        
        # Show the legend
                ax.imshow(EDir)
                ax.set_title(title[i * cols + j])
                ax.set_ylabel(' Pixels')
                ax.axis('off')
        # Create an inset axes for 'EDirLeg'
                inset_ax = ax.inset_axes([0.8, 0.0, 0.2, 0.2])
            # [x0, y0, width, height],Adjust the inset position and size as needed
                inset_ax.imshow(EDirLeg)
        # Add annotations to the inset image
                an1=inset_ax.annotate("", xy=(1, 0.5), xycoords='axes fraction',
                          xytext=(0.5, 0.5), textcoords ='axes fraction', 
                          ha="center", va="center",
                          arrowprops=dict(arrowstyle="-|>", color="w"),
                          )
                an2=inset_ax.annotate("Amp", xy=(0.5, 0.5), xycoords=an1.get_window_extent,
                             xytext=(2, 2), textcoords="offset points", color="white", size=15*(imgsize/10),
                             ha="center", va="bottom")        
                an3 = inset_ax.annotate("Phase",
                  xy=(0.2, 0.2), xycoords='axes fraction',
                  xytext=(0.3, 0.8), textcoords='axes fraction', color="white", size=15*(imgsize/10),
                   va="center", ha="center",
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3,rad=0.6", color="w"))
                inset_ax.axis('off')
            else:
                ax.axis('off')  # Turn off the axis for empty plots
     # Adjust layout and display/save the plot
    plt.tight_layout()
    if storing[0]:
        plt.savefig(f'{storing[1]} .tiff', dpi=600)
    plt.show()


def browse_images(dataset, properties={
        'resolution': 0.0131411635412124,
        'magnification': '3.6 Mx',
        'unit': 'nm',
        'figsize': 8,
        'cmap': 'viridis'
            }):
    
    """
    This code requires the "%matplotlib inline".

    Browse the images stored in the dataset.
    This function can denoise images through making bandpass filters on Fourier transform spots, 
    which can be achieved by setting 'ration =1 and inverse = True'.
    The present filtered image and the corresponding filter will be return.
    
    default_properties = {
        'resolution': 0.0131411635412124,
        'magnification': '3.6 Mx',
        'unit': 'nm',
        'figsize': 8,
        'cmap': 'viridis'
       }
       
    The items included in the above dictionary can be omitted.
    
    Args:
    
    dataset: list, storing the data with matrix. It is loaded using 'hs.load()'
    
    """

    default_properties = {
        'resolution': 0.0131411635412124,
        'magnification': '3.6 Mx',
        'unit': 'nm',
        'figsize': 8,
        'cmap': 'viridis'
           }
    
    for key, value in default_properties.items():
        if key in properties.keys():
            if properties[key] is None or properties[key] == ' ' or properties[key] ==str(''):
                properties[key] = value
        else:
            properties[key] = value
     
    unit = properties['unit']    
    if isinstance(dataset, np.ndarray):
        shape = dataset.shape
        n = shape[0] if len(shape) == 3 else 1 if len(shape) == 2 else None
    elif isinstance(dataset, list) or isinstance(dataset, dict):
        n = len(dataset)
    else:
        print("Error: The input dataset seems not correct in format!")
        return

    outputImage = []
    outputMask =[]
    def view_img(i, ratio, sigma_max, background, FFTspots, inverse, disk_size):
        if n > 0:
            if isinstance(dataset, dict):
                title = list(dataset.keys())[i]
                img = dataset[title]
            else:
                img = dataset[i]
                title = ''
            if not isinstance(img, np.ndarray):
                img = np.asarray(img)
                
            FOV = properties['resolution']*img.shape[0]
            length = [-FOV/2, FOV/2, -FOV/2, FOV/2]
           
            img_fft, bck_fft = analysis.periodic_DFT(img, inverse_dft=False) #np.fft.fft2(img) 
           
            sx, sy = img_fft.shape
            magnitude_fft = np.array(np.log(1+np.abs(np.fft.fftshift(cp.asnumpy(img_fft - bck_fft)))))
            magnitude_fft0 = ndimage.gaussian_filter(magnitude_fft, sigma=(5,5), order=0)
            magnitude_fft1 = ndimage.gaussian_filter(magnitude_fft0, sigma=(1,1), order=0)
            magnitude_fft2 = magnitude_fft1 + 50 *(magnitude_fft0 - magnitude_fft1)
            fft_crop = crop_matrix(magnitude_fft2, (0,1), [int(sx/2), int(sy/2)],[int(sx/ratio), int(sy/ratio)])
            power_spec = linscale(fft_crop) #make the intensity linear

            #pixe size in reciprocal space
            rec_resolution = 1/(properties['resolution']*power_spec.shape[0])

            pixels = (np.linspace(0,power_spec.shape[0]-1,power_spec.shape[0])-power_spec.shape[0]/2)* rec_resolution
            x,y = np.meshgrid(pixels,pixels)
            mask = np.zeros(power_spec.shape)

            mask_spot = x**2+y**2 > 2**2 
            mask = mask + mask_spot
            mask_spot = x**2+y**2 < 5**2 
            mask = mask + mask_spot

            mask[np.where(mask==1)]=0 # just in case of overlapping disks

            minimum_intensity = power_spec[np.where(mask==2)].min()*0.85

            maximum_intensity = power_spec[np.where(mask==2)].max()*1.25
            rec_scale = np.array([rec_resolution, rec_resolution,1])
            center = np.array([int(power_spec.shape[0]/2), int(power_spec.shape[1]/2),1] )
     
            extend_edge = rec_resolution*power_spec.shape[0]/2
            extend = [-extend_edge, extend_edge, -extend_edge, extend_edge]
            
            if inverse and ratio==1:
                fig, axs = plt.subplots(1,4 , figsize=(properties['figsize']*4, properties['figsize']))
            else:
                fig, axs = plt.subplots(1,2 , figsize=(properties['figsize']*2, properties['figsize']))
                
            axs[0].imshow(img, cmap = properties['cmap'], extent = length)            
            axs[0].set_title(f'Image {title} by '+properties['magnification'])
            axs[0].set_xlabel(f'Field of view ({unit})')
            axs[1].imshow(power_spec.T, cmap = 'gray', extent = extend,  vmin = minimum_intensity, vmax= maximum_intensity)
            zero_feq = plt.Circle((0,0), 5*rec_resolution, color='yellow', linewidth=2, fill=True)
            axs[1].add_patch(zero_feq)
            if FFTspots:    
                detected_spots = blob_log(power_spec, max_sigma=sigma_max, threshold=background/10)
                for spot in detected_spots:  
                    spot_y, spot_x, spot_r = spot
                    c = plt.Circle(((spot_y-power_spec.shape[0]/2)*rec_resolution, -(spot_x-power_spec.shape[1]/2)*rec_resolution), spot_r*2*rec_resolution, color='red', linewidth=2, fill=False)
                    axs[1].add_patch(c)
                
            axs[1].set_title(f"Fast Fourier Transform of {title}")
            axs[1].set_xlabel(f'Reciprocal distance (1/{unit})')
            if inverse and ratio ==1:
                mask = np.zeros(power_spec.shape)
                for spot in detected_spots:  
                    spot_y, spot_x, spot_r = spot
                    coords = (spot_y, spot_x)
                    mask_spot = circle_mask(power_spec.shape, coords, (0, disk_size))
                    mask += mask_spot
                fft_filtered = np.fft.fftshift(cp.asnumpy(img_fft - bck_fft))*mask
                filtered = np.fft.ifft2(np.fft.ifftshift(fft_filtered))
                axs[2].imshow(filtered.real,extent=(0, properties['resolution']*power_spec.shape[0],0, properties['resolution']*power_spec.shape[1]), origin = 'upper')
                axs[2].set_title(f"Bandpass filtered image of {title}")
                axs[2].set_xlabel(f'Length ({unit})')
                axs[3].imshow(mask)
                axs[3].set_title(f"Bandpass filter of {title}")
                if len(outputImage) ==0:
                    outputImage.append(filtered.real)
                    outputMask.append(mask)
                else: 
                    outputImage[0] = filtered.real
                    outputMask[0] = mask
            plt.tight_layout()
            plt.show()            
    interact(view_img, 
             i=widgets.IntSlider(min = 0, max = n-1, value = 0),
            ratio = widgets.IntSlider(min = 1, max = 8, value = 2),
            FFTspots=widgets.Dropdown(options=[True, False]),
            inverse=widgets.Dropdown(options=[False, True]),
            sigma_max = widgets.FloatSlider(min = 3, max = 10, value = 5),
            background=widgets.FloatSlider(min = 0.1, max = 1, value = 0.5),
            disk_size=widgets.IntSlider(min = 10, max = 100, value = 20, step = 1)
            ) 
    return outputImage, outputMask

def browse_images_in_folder(folder_path, colormap='viridis'):
    """
    Browse images from a folder and display them with line profile analysis.

    Args:
    - folder_path: str, path to the folder containing images.
    - colormap: str, the colormap to use for displaying the images.
    
    Note: Requires "%matplotlib inline" in Jupyter notebook.
    """  
    matches = []
    file_names = []
    allowed_formats = ('.png', '.jpeg', '.jpg', '.tiff', '.gif')
    
    for path, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            if filename.lower().endswith(allowed_formats):
                file_path = os.path.join(path, filename)
                file_names.append(filename)
                matches.append(file_path)    
    num_images = len(matches)
    
    def show_image(n, dy, width, brightness, mode):
        """
        Displays the selected image with its line profile.

        Args:
            n (int): Index of the image to display.
            dy (int): The y-coordinate for the line profile.
            width (int): The width of the line profile.
            brightness (int): Amplifier for scaling the image intensity in the colormap.
        """
        with Image.open(matches[n]) as img:
            if img.mode == "RGBA" or img.mode == "RGB":
                img = img.convert('L')  # Ensure the image is in grayscale
            image_array = np.array(img, dtype=np.float32)
            nx, ny = img.size

        if mode == "log":
            img_array = np.log(image_array+.1)
        elif mode == "inverse":
            img_array = 1/(image_array+.1)
        elif mode =="linear" : img_array = image_array
        else: img_array = 1/np.log(image_array+.1)
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        img_display = axes[0].imshow(img_array, cmap=colormap, vmin=np.min(img_array), vmax=np.max(img_array) / brightness)
        norm = Normalize(vmin=np.min(img_array), vmax=np.max(img_array))
        fig.colorbar(cm.ScalarMappable(norm=norm), ax=axes[0], location='right', shrink=1)
        
        axes[0].set_title(f"{file_names[n]}")
        find_centre = image_array.copy()
        find_centre[find_centre<np.max(find_centre)/2]=0
        centre = ndimage.center_of_mass(find_centre)
        position = [round(centre[1]), round(centre[0])]
        axes[0].scatter(centre[1], centre[0], color='r')
        axes[0].text(.05,.95,  'Centre: '+str(position),
                    horizontalalignment='left',
                    verticalalignment='top',
                    color = 'white')
        line = measure.profile_line(image_array, (dy, 0), (dy, nx - 1), linewidth=width, reduce_func=np.sum)
        rectangle = patches.Rectangle(xy=(0, dy), width=ny - 1, height=width, color='green', linewidth=2, alpha=0.7)
        axes[0].add_patch(rectangle)
        axes[0].set_xlabel("Pixels")
        
        axes[1].plot(np.arange(nx), line) 
        axes[1].text(.05,.95,  'Total Int.: '+str(np.sum(image_array))+' counts',
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=axes[1].transAxes)
        axes[1].set_title("Line profile")
        axes[1].set_xlabel("Pixels")
        axes[1].set_ylabel("Integrated Intensity of line (counts)")
        plt.tight_layout()
        plt.show()

    interact(show_image,                        
             n=widgets.IntSlider(description='Sequence', min=0, max=num_images - 1, value=0, step=1, layout=widgets.Layout(width='100%')),
             dy=widgets.IntSlider(min=0, max=512, value=256, step=1, layout=widgets.Layout(width='70%')),
             width=widgets.IntSlider(min=1, max=150, value=5, step=1, layout=widgets.Layout(width='60%')),
             brightness=widgets.IntSlider(min=1, max=50, value=1, step=1, layout=widgets.Layout(width='50%')),
             mode = widgets.Dropdown(options = ["log", "inverse", "linear", "inverse_log"], value='log',
                                     description='Mode:', disabled=False)
            )

def extract_segmented_image(file_path, order_map):
    """
    Searching the segmented images from '.emd' dataset to store in a list with their names
    
    Args:
    file_path: the directory of the dataset, loading data using rsciio.emd.file_reader(file_path)
    order_map: dic, listing the names of each segments, e.g.
        {
          'DF4-A': 'A',
          'DF4-B': 'B',
          'DF4-C': 'C',
          'DF4-D': 'D'
        }

    Return:
    DPCs: a list, storing the segmented images extracted from the dataset
                     The segmented image is stored with 'np.ndarray'
              
    titles: a list storing the corresponding names of images
    
    """
    emd_data = emd_reader(file_path)
    DPC_imgs = []
    titles = []
    if order_map is None:
        order_map = {
        'DF4-A': 'A',
        'DF4-B': 'B',
        'DF4-C': 'C',
        'DF4-D': 'D'
        }
    num = len(emd_data)
    if num > 1:
        for i in range(num):
            title = emd_data[i]['metadata']['General']['title']
            if title in order_map:
                DPC_imgs.append(emd_data[i]['data'])
                titles.append(title)
    
        # Sort the images based on the predefined order
        sorted_indices = sorted(range(len(titles)), key=lambda k: order_map[titles[k]])
        DPC_imgs = [DPC_imgs[i] for i in sorted_indices]
        titles = [titles[i] for i in sorted_indices]

    else:
        print(f'DPC segmented image is lacking in raw data')
  

    return DPC_imgs, titles


def get_info(file_path, all_info = False):
    """
    It is to extract the information of the dataset, including the resolution, magnification, unit, and so on.
    It supports .emd, .dm3, .dm4, .ser, and .emi files.
    """
    file_extension = os.path.splitext(file_path)[1][1:].lower()  # Get extension without dot
    library = {}  # Initialize empty dictionary

    if file_extension == 'emd':
        emd_data = emd_reader(file_path)
        if all_info:
            print(dict_tree(emd_data[0]))
        library['title'] = emd_data[0]['metadata']['General']['original_filename']
        library['resolution'] = emd_data[0]['axes'][0]['scale']
        library['unit'] = emd_data[0]['axes'][0]['units']
        library['pixel_size'] = emd_data[0]['axes'][0]['size']
        library['Acquisition Date'] = emd_data[0]['metadata']['General']['date']
        library['Acquisition Time'] = emd_data[0]['metadata']['General']['time']
        library['Stage Position'] = emd_data[0]['original_metadata']['Stage']['Position']
        library['Dwell Time (s)'] = emd_data[0]['original_metadata']['Scan']['DwellTime']
        library['Acc. voltage (V)'] = emd_data[0]['original_metadata']['Optics']['AccelerationVoltage']
        library['Illumination mode'] = emd_data[0]['original_metadata']['Optics']['IlluminationMode']
        if library['Illumination mode'] == 'Probe':
            library['Probe semi_convergence angle (rad)'] = emd_data[0]['original_metadata']['Optics']['BeamConvergence']
        library['LastMeasuredScreenCurrent (pA)'] = float(emd_data[0]['original_metadata']['Optics']['LastMeasuredScreenCurrent'])*10**12
    elif file_extension in ('dm3', 'dm4'):
        dm_data = dm_reader(file_path)
        if all_info:
            print(dict_tree(dm_data[0]))
        library['title'] = dm_data[0]['metadata']['General']['title']
        library['resolution'] = dm_data[0]['axes'][0]['scale']
        library['unit'] = dm_data[0]['axes'][0]['units']
        library['pixel_size'] = dm_data[0]['axes'][0]['size']
        library['Exposure Time (s)'] = dm_data[0]['original_metadata']['ImageList']['TagGroup0']['ImageTags']['DataBar']['Exposure Time (s)']
        library['Acquisition Date'] = dm_data[0]['original_metadata']['ImageList']['TagGroup0']['ImageTags']['DataBar']['Acquisition Date']
        library['Acquisition Time'] = dm_data[0]['original_metadata']['ImageList']['TagGroup0']['ImageTags']['DataBar']['Acquisition Time']
        library['Stage Position'] = dm_data[0]['original_metadata']['ImageList']['TagGroup0']['ImageTags']['Microscope Info']['Stage Position']
        library['Acc. voltage (V)'] = dm_data[0]['original_metadata']['ImageList']['TagGroup0']['ImageTags']['Microscope Info']['Voltage']
        library['Illumination mode'] = dm_data[0]['original_metadata']['ImageList']['TagGroup0']['ImageTags']['Microscope Info']['Operation Mode']
    elif file_extension in ('ser', 'emi'):
        tia_data = tia_reader(file_path)
        if all_info:
            print(dict_tree(tia_data[0]))
        library['title'] = tia_data[0]['metadata']['General']['title']
        library['resolution'] = tia_data[0]['axes'][0]['scale']
        library['unit'] = tia_data[0]['axes'][0]['units']
        library['pixel_size'] = tia_data[0]['axes'][0]['size']

    return library



def dict_tree(data, indent=""):
  """Recursively prints nested dictionary keys and values with indentation."""

  for key, value in data.items():
    print(f"{indent}{key}: ")
    if isinstance(value, dict):
        dict_tree(value, indent + "-> ")
    else:
        print(f"{indent}{value}")
    print()  # Add a newline after each top-level dictionary

def data_tree(file):
    """
    It searches the groups, subgroups, and the datasets stored in the loaded h5py file.
    This code can give the names and the corresponding values stored in the datasets.

    """
    def head_tree(node, indent=0):
        if isinstance(node, h5py.Group):
            print("||>--"+"--" * indent + "\033[91m"+f" Group: {node.name}"+"\033[0m")
            for key in node.keys():
                head_tree(node[key], indent+1)
        elif isinstance(node, h5py.Dataset):
            print("  |>"+"--" * (indent+1) + "\033[92m"+f"Dataset: {node.name}"+"\033[0m")
            values = node[()]            
            if isinstance(values, np.ndarray):
                size = values.shape
                if size[0]>10:
                    print("  |>"+"---"*(indent+1)+f"Here is an array with a shape of {size}")
                else:                     
                    if len(values)>2:
                        for v in values:                            
                            print("    |>"+"---"*(indent+1)+ "\033[96m"+f"value: {v}."+"\033[0m")
                    else:print("    |"+"---"*(indent+1)+ "\033[96m"+f"value: {values}. "+"\033[0m")
            else: 
                print("    |>"+"---"*(indent+1) + "\033[96m"+f"value: {values}"+"\033[0m")            


    for key in file.keys():
        head_tree(file[key])



def crop_matrix(original_matrix, axis=(0, 1), centre=None, crop_size=None):
    """
    Crops a submatrix from the original matrix around a specified centre.

    Args:
        original_matrix (np.ndarray or cp.ndarray): The input matrix to be cropped.
        axis (tuple, optional): The axes along which cropping is performed. Default is (0,1).
        centre (tuple, optional): Coordinates of the centre of the cropped matrix.
                                  Defaults to the centre of the original matrix.
        crop_size (tuple, optional): Size of the cropped matrix (height, width).
                                     Defaults to half of the original size.

    Returns:
        np.ndarray or cp.ndarray: Cropped submatrix.

    Raises:
        ValueError: If the crop size is invalid or exceeds matrix bounds.
    """

    shape = original_matrix.shape
    ndim = len(shape)

    # Validate axis
    if len(axis) != 2 or any(a >= ndim for a in axis):
        raise ValueError(f"Invalid axis {axis} for a {ndim}-dimensional array.")

    # Extract the relevant dimensions
    px, py = shape[axis[0]], shape[axis[1]]

    # Default crop size (half of the original matrix)
    if crop_size is None:
        crop_size = (px // 2, py // 2)

    # Default centre (middle of the matrix)
    if centre is None:
        centre = (px // 2, py // 2)

    # Compute cropping indices
    start_row, end_row = max(0, centre[0] - crop_size[0] // 2), min(px, centre[0] + crop_size[0] // 2)
    start_col, end_col = max(0, centre[1] - crop_size[1] // 2), min(py, centre[1] + crop_size[1] // 2)

    # Crop the matrix along the specified axes
    slicing = [slice(None)] * ndim  # Create a full slicing list
    slicing[axis[1]] = slice(start_row, end_row)
    slicing[axis[0]] = slice(start_col, end_col)

    return original_matrix[tuple(slicing)]


def expand_matrix(matrix, mode='x'):
    """
        Expand the matrix by combining it with its flipped versions.
    
        Parameters:
        - matrix: The input 2D matrix.
        - mode: The type of expansion ('x' or 'y'), which 'x' represnts the matrix is expanded along the x-axis, and 'y' represents the matrix is expanded along the y-axis.
    
        Returns:
        - A new matrix expanded based on the specified mode.
    """
    y_flipped = np.flip(matrix, axis=1)
    x_flipped = np.flip(matrix, axis=0)
    both_flipped = np.flip(matrix, axis=(0, 1))
    
    if mode == 'x':
        top_row = np.hstack((-both_flipped, -x_flipped))
        bottom_row = np.hstack((y_flipped, matrix))
    elif mode == 'y':
        top_row = np.hstack((-both_flipped, x_flipped))
        bottom_row = np.hstack((-y_flipped, matrix))
    else:
        raise ValueError("Invalid mode. Use 'x' or 'y'.")
    return np.vstack((top_row, bottom_row))

def normalize_array(array):
    """
    Normalization of an array, ensuring the sum of the square of the array equal to 1.
    """
    if isinstance(array, np.ndarray):
        array = cp.array(array)
    if cp.iscomplexobj(array):
        squared_array = cp.square(cp.abs(array))
        result = array/cp.sqrt(cp.sum(squared_array))
    else: result = array/cp.linalg.norm(array)
    return result

def dpc_normalization(dpc_imgs):
    '''
    Normalize the raw DPC measurements by dividing and subtracting out the mean intensity.
    '''
    for img in dpc_imgs:
        img          /= ndimage.uniform_filter(img, size=img.shape[0]//2)
        meanIntensity = img.mean()
        img          /= meanIntensity        # normalize intensity with DC term
        img          -= 1.0                  # subtract the DC term
    return dpc_imgs

def linscale(image, nmin=0, nmax=1, min = None, max = None):
    """
    Calculating the contrast of image and rescale image intensities.

    Rescale the image intensities to a new scale. The value
    min and everything below gets mapped to nmin, max and everything
    above gets mapped to nmax. By default the minimum and maximum
    get mapped to 0 and 1.

    Args:
       arr : 2D np.ndarray object, the image to normalize

       nmin : float, optional
          The new minimum of the image. Defaults to 0.
       nmax : float, optional
          The new maximum of the image. Defaults to 1. For 8-bit images use 255.
          For 16-bit images use 65535.
    
    Returns
    -------
       result : array
         Intensity-rescaled image

    Notes
    -----
    The type recasting happens in an 'unsafe' manner. That is, if elements
    have float values like 0.99, recasting to np.uint8 will turn this into 0
    and not 1.
   
    """
    if not isinstance(image, np.ndarray):
        copied_image = cp.asnumpy(image.copy())
    else: copied_image = image.copy()
    if min is None:
        min = copied_image.min()
    else:
        copied_image[copied_image < min] = min
    if max is None:
        max = copied_image.max()
    else:
        copied_image[copied_image > max] = max
    
    a = (nmax-nmin)/(max-min)
    linear = (copied_image-min)*a
    rescaled = (linear+nmin)

    return rescaled



def plot_image(DPC_imgs, properties=None):
    """
    It is used to present images.
    The input datasets are np.ndarray or a list with the element of 2D np.ndarray.
    The complex data would be abandoned its imaginary component.

    Args:

    DPC_imgs: the data for plotting, it can be a single numpy.ndarray or a list of numpy.ndarray.

    properties: a dictionary, items which can be omitted, definning the properties of plotted images, the default values as shown below:

    properties = {
        'resolution': 1,
        'unit': '',
        'bar location': 'lower left',
        'image titles': [],
        'annotates': annotate,
        'figsize': 8,
        'interpolation': 'gaussian',
        'cmap': 'viridis',
        'dpi': 100,
        'image format': '.jpeg',
        'showing titles': True,
        'cropping image': [False, [512, 512], [512, 512]], #details explained in "crop_matrix"
        'saving image': False, #(or True for saving the plotted image)
        'saving path': ''  #if you expect to save the plotted image, put the saving path at here
    }
    """
    annotate=['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)',
              '(l)', '(m)', '(n)', '(o)', '(p)', '(q)', '(r)', '(s)', '(t)', '(z)', '(u)']
    
    default_properties = {
        'resolution': 1,
        'unit': 'nm',
        'mode': None,
        'bar location': 'lower left',
        'image titles': [None]*len(DPC_imgs),
        'annotates': annotate,
        'figsize': 8,
        'interpolation': 'gaussian',
        'cmap': 'hot',
        'dpi': 600,
        'image format': '.jpeg',
        'showing titles': False,
        'cropping image': [False, [512, 512], [512, 512]],
        'saving image': False,
        'saving path': ''
    }
    if properties['showing titles']:
        a = properties['figsize']
    else: a = 0  
    for key, value in default_properties.items():
        if key in properties.keys():
            if properties[key] is None or properties[key] == str('') or properties[key] ==' ':
                properties[key] = value
        else:
            properties[key] = value
            
    if properties['saving image']:
        if properties['saving path'] is None or properties['saving path']=='':
            print("The saving path should be pointed!!")
            
    image = []
    img_name = []
    
    if properties['mode'] is None:
        mode = "Imaging"
            
    if isinstance(DPC_imgs, list) and len(DPC_imgs)>1: 
        if isinstance(DPC_imgs[0], list):
            print(f'The format of input data is incorrect!!!')
        else:
            try:
                label = DPC_imgs[0]['axes'][0]['units']
                for j in range (len(DPC_imgs)):
                    img_name.append(DPC_imgs[j]['metadata']['General']['title']) #read the title of images
            except:
                label = properties['unit']
                for j in range (len(properties['image titles'])):
                    img_name.append(properties['image titles'][j])
                
        for i in range (len(DPC_imgs)):            
            if not isinstance(DPC_imgs[i], np.ndarray):
                image.append(cp.asnumpy(DPC_imgs[i]))
            else: image.append(DPC_imgs[i]) 
                
            if properties['cropping image'][0] == True:
                image[i] = crop_matrix(image[i], (0, 1), properties['cropping image'][1], properties['cropping image'][2])

        row = int(np.sqrt(len(image)))       
        column = len(image)//row
        if row*column < len(image):
            column += 1
        fig, axs =  plt.subplots(row, column, 
                                 figsize=(properties['figsize']*column, a + properties['figsize']*row)) 
        
        for i, ax in enumerate(axs.flat):
            if mode =="DIFFRACTION": # The constant '80' can be adjusted for the diffraction displaying
                if np.issubdtype(image[i].dtype, np.complex128):
                    f_image = image[i].real
                else: f_image = image[i]
                im = np.interp(f_image, (0, 80*np.log(f_image.max())), (0, 65535)).astype(np.uint16) 
                unit_style = 'si-length-reciprocal'
            else:
                if np.issubdtype(image[i].dtype, np.complex128):
                    f_image = image[i].real
                else: f_image = image[i]
                im = np.interp(f_image, (f_image.min(), f_image.max()), (0, 65535)).astype(np.uint16)
                unit_style = 'si-length'
            if properties['resolution'] is None:
                try:
                    resolution = DPC_imgs[0]['axes'][0]['scale']
                except:
                    resolution =1
            else: resolution = properties['resolution']
            
            ax.imshow(im, cmap = properties['cmap'], interpolation = properties['interpolation'])
            if properties['showing titles']:
                ax.set_title(img_name[i])
                ax.annotate(
                          properties['annotates'][i],
                          xy=(0, 1), xycoords='axes fraction',
                          xytext=(+0.5, -0.5), textcoords='offset fontsize',
                          fontsize='large', verticalalignment='top', fontfamily='arial',
                         bbox=dict(facecolor='0.7', edgecolor='none', pad=3.0))
                
            scale_bar = ScaleBar(resolution, units=label, dimension = unit_style, length_fraction = 0.2, 
                                 location = properties['bar location'],scale_loc = 'top')
            ax.add_artist(scale_bar)
            #ax.set_ylabel(label)
            ax.axis("off")
            
        plt.tight_layout()            
        if properties['saving image']:
            plt.savefig(fname=properties['saving path']+properties['image format'], dpi=properties['dpi'], transparent=True)
        plt.show()
            
    else: 
        if isinstance(DPC_imgs, list):
            print(f'The input data is incorrect!!!')
        else:
            try:
                label = DPC_imgs[0]['axes'][0]['units']
                img_name.append(DPC_imgs[0]['metadata']['General']['title'])
            except:
                label = properties['unit']
                img_name.append(properties['image titles'])  
                
        if not isinstance(DPC_imgs, np.ndarray):
            image = cp.asnumpy(DPC_imgs)
        else: image = DPC_imgs
                
        if properties['cropping image'][0]:
            image = crop_matrix(image,(0,1), properties['cropping image'][1], properties['cropping image'][2])
                
        if mode =="DIFFRACTION":
            if np.issubdtype(image.dtype, np.complex128):
                f_image = image.real
            else: f_image = image
            im = np.interp(f_image, (0, 80*np.log(f_image.max())), (0, 65535)).astype(np.uint16)
            unit_style = 'si-length-reciprocal'
        else:
            if np.issubdtype(image.dtype, np.complex128):
                f_image = image.real
            else: f_image = image
            im = np.interp(f_image, (f_image.min(), f_image.max()), (0, 65535)).astype(np.uint16)
            unit_style = 'si-length'
            
        if properties['resolution'] is None:  
            try:
                resolution = DPC_imgs[0]['axes'][0]['scale']
            except: resolution = 1
        else: resolution = properties['resolution']
        
        fig, ax = plt.subplots(figsize = (properties['figsize'], properties['figsize']))
        plt.imshow(im, cmap = properties['cmap'], interpolation = properties['interpolation'])
        scale_bar = ScaleBar(resolution, units=label, dimension = unit_style, length_fraction = 0.2, location = properties['bar location'], scale_loc = 'top')
        ax.add_artist(scale_bar)
        if properties['showing titles']:
                ax.set_title(img_name)

        ax.axis("off")
        plt.tight_layout()
        
        if properties['saving image']:
            plt.savefig(fname=properties['saving path']+properties['image format'], dpi=properties['dpi'], transparent=True)
        plt.show()



def plot_fft(images, log=True, names=None):
    """
    Displays each image in a list alongside its Fourier Transform (FFT) magnitude spectrum.
    You also can try the "show_fft" function.
    Args:
        images (list or np.ndarray): A single 2D image or a list of 2D images.
        log (bool): If True, displays the logarithmic scale of the FFT magnitude for better visualization.
    """
    # Ensure `images` is a list
    if not isinstance(images, list):
        images = [images]
        
    if names is not None and len(names) == len(images):
        name = 1
    else: name = 0
        
    for idx, image in enumerate(images):
        # Compute the FFT and shift zero-frequency components to the center
        ft = np.fft.fftshift(np.fft.fft2(image))
        
        # Compute the magnitude spectrum
        magnitude = np.abs(ft)
        
        # Apply log scaling if specified
        if log:
            magnitude = np.log1p(magnitude)  # log(1 + abs(ft)) to avoid log(0)
        
        # Create the figure and axes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Display the original image
        p1 = ax1.imshow(image, cmap='gray')
        if name ==1:
            ax1.set_title(f"Image: {names[idx]}")
        else: ax1.set_title(f"Image")
        fig.colorbar(p1, ax=ax1, shrink=0.8, extend='both')
        
        # Display the FFT magnitude spectrum
        p2 = ax2.imshow(magnitude, cmap='viridis')
        if name ==1:
            ax2.set_title(f"FFT: {names[idx]}")
        else: ax2.set_title(f"FFT")
        fig.colorbar(p2, ax=ax2, shrink=0.8, extend='both')
        
        # Adjust layout for better readability
        plt.tight_layout()
        plt.show()


def Bragg_points(points, given_point, threshold=0.05, pixel_size = 1):
    """
    Compute distances from each point in `points` to `given_point`, 
    filter out points whose distance difference (relative to the previous accepted point)
    is less than `threshold` (5% by default), and return the filtered points and distances.
    
    Parameters:
        points (list or array): List of points (each a tuple or list of coordinates).
        given_point (tuple or list): The reference point.
        threshold (float): The minimum relative difference required between distances.
        pixel_size: float, the real size of one pixel.
    Returns:
        filtered_points (list): The points that pass the filtering.
        filtered_distances (list): Their corresponding distances.
    """
    # Convert input lists to NumPy arrays for vectorized operations.
    points_arr = np.array(points)
    pts_arr = points_arr * pixel_size
    given_point_arr = np.array(given_point)
    given_pt = given_point_arr * pixel_size
    # Compute Euclidean distances from each point to the given point.
    distances = np.linalg.norm(pts_arr - given_pt, axis=1)
    distances = np.reciprocal(distances, where = (distances != 0))
    # Sort points by distance.
    sorted_indices = np.argsort(distances)
    sorted_points = points_arr[sorted_indices]
    sorted_distances = distances[sorted_indices]
    
    # Initialize lists to store the filtered points and distances.
    filtered_points = []
    filtered_distances = []
    
    # Iterate over sorted points and filter based on the relative difference.
    for pt, d in zip(sorted_points, sorted_distances):
        if not filtered_points:
            # Always add the first point.
            filtered_points.append(pt)
            filtered_distances.append(d)
        else:
            last_d = filtered_distances[-1]
            # If the last accepted distance is zero (i.e., the given point was in the list),
            # then only add the new point if its distance is non-zero.
            if last_d == 0:
                if d != 0:
                    filtered_points.append(pt)
                    filtered_distances.append(d)
            else:
                # Only add the point if the distance difference is at least `threshold` times the last distance.
                if (d - last_d) / last_d >= threshold:
                    filtered_points.append(pt)
                    filtered_distances.append(d)
                    
    print("Bragg points and their d-spacings:")
    for pt, d in zip(filtered_points, filtered_distances):
        print(np.round(pt), np.round(d,3))
        
    return filtered_points, filtered_distances

def circle_mask(mask_size=(2048, 2048), center=(1024, 1024), radius=(10,20)):
    """
    Creates a circular mask.

    Parameters:
        size (tuple): Size of the mask (height, width).
        center (tuple): Center of the circle (x, y).
        radius (float): Radius of the circle (inner-circle, outer-circle).

    Returns:
        numpy.ndarray: Circular mask.
    """
    mask = np.zeros(mask_size)
    y, x = np.ogrid[:mask_size[0], :mask_size[1]]
    distance_from_center = (y - center[1])**2 + (x - center[0])**2
    mask[(distance_from_center >= radius[0]**2)&(distance_from_center <= radius[1]**2)] = 1
    return mask

def find_radius(image, center=None, mask_size= None):
    """
    Plot an image with a circle centered at a specified point using an interactive slider.
    It is designed to find the center and radius of CBED.
    
    Parameters:
        image: np.ndarray, The image to be displayed.
        center (tuple): Coordinates of the center of the circle (center_x, center_y).
            If center is None, the code will search the center automatically.
        mask_size: int, for finding the center of pattern.
    Return:
        radius (float): Radius of the circle.
    """
    px, py = image.shape
    
    if isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)
    if center is not None:
        if isinstance(center, list) or isinstance(center, tuple):
            cc = []
            for e in center:
                if isinstance( e, cp.ndarray):
                    cc.append(cp.asnumpy(e))
                else: cc.append(e)
            cc = np.array(cc)
        elif isinstance(center, cp.ndarray):
            cc = cp.asnumpy(center)
        else: cc = center    
    else:
        cc = np.zeros(2)
        coarse_c = ndimage.center_of_mass(image)
        rmax = min( coarse_c[0], coarse_c[1], abs(py - coarse_c[0]), abs(px - coarse_c[1]))
        r = mask_size if mask_size is not None else rmax*0.8         
        mask = np.zeros((px, py))
        x, y = np.ogrid[:px, :py]
        distance_from_center = (x - coarse_c[1])**2 + (y - coarse_c[0])**2
        mask[(distance_from_center <= r**2)] = 1
        cc[1], cc[0] = ndimage.center_of_mass(image * mask)

    max_r = min( abs(py - cc[1]), abs(px - cc[0]) , cc[0], cc[1])

    def draw_circles(image, cx, cy, radius, zoom):
        cc = (cx, cy)
        start_row = int(cc[1]) - zoom // 2
        end_row = start_row + zoom
        start_col = int(cc[0]) - zoom // 2
        end_col = start_col + zoom
        if start_row >= 0 and start_col >= 0:
            cropped_image = image[start_row:end_row, start_col:end_col]
            
        fig, ax = plt.subplots(1,2, figsize=(16,8))

        dx = cc[0] - int(cc[0])
        dy = cc[1] - int(cc[1])
        nx, ny = cropped_image.shape
        radius = min(radius, nx/2-dx, ny/2-dy)
        cm1 = ax[0].imshow(np.log(cropped_image + 1), cmap='hot')
        fig.colorbar(cm1, ax=ax[0], shrink=0.9, label='Log (intensity)')
        ax[0].set_title(f'Radius = {round(radius, 2)} pixels')
        ax[0].scatter(nx/2 + dx, ny/2 +dy, color='k', label='Center')
        circle1 = plt.Circle((nx/2 +dx, ny/2 +dy), radius=radius, 
                        fill=False, color='k', linestyle='--', linewidth=2)
        ax[0].add_patch(circle1)
        ax[0].legend()
        cm2 = ax[1].imshow(cropped_image, cmap='hot')
        fig.colorbar(cm2, ax=ax[1], shrink=0.9, label='Linear intensity')
        ax[1].set_title(f'Radius = {round(radius, 2)} pixels')
        ax[1].scatter(nx/2 + dx, ny/2 +dy, color='k', label='Center')
        circle2 = plt.Circle((nx/2 +dx, ny/2 +dy), radius=radius, 
                        fill=False, color='c', linestyle='--', linewidth=2)
        ax[1].add_patch(circle2)
        ax[1].legend()
        fig.suptitle(f"Center = {round(cc[0], 2), round(cc[1], 2)} pixels")
        plt.show()
        return radius
    interact(draw_circles, 
             image=widgets.fixed(image),
             cx=widgets.FloatText(value=round(cc[0],2), step = 0.1, description='center X:'),
             cy=widgets.FloatText(value=round(cc[1],2), step = 0.1,description='center Y:'),
             radius=widgets.FloatText(value=round(max_r/2, 2), step = 0.1,description='Radius:'),
            zoom = widgets.IntSlider(min=1, max=int(max_r*2), value=int(max_r)), description='Zoom in:')


def segmented_circular_masks(mask_size, nbins_radial, nbins_azimuthal, center, inner_radius, outer_radius, rotation=np.pi/4):
    """
    Creates multiple binary masks with ones within the specified circular segments.
    The rotation should be specified in radians.
    The mask will be displayed in count counter-clockwise.
    Args:
        mask_size (tuple): (height, width) of the mask.
        nbins_radial (int): the number of segments divided along radial direction.
        nbins_azimuthal (int): the number of segments divided in the azimuthal plane.
        center (tuple): (center_x, center_y) coordinates of the circle's center.
        inner_radius (float): Inner radius of the circular segment.
        outer_radius (float): Outer radius of the circular segment.
        angle_ranges (list of tuples): List of angle ranges in degrees, e.g., [(-45, 45), (45, 135)].
    
    Returns:
        np.ndarray: An array of binary masks with shape (n, height, width).
    """
    width, height = mask_size
    # Determine the center
    if center is not None:
        center_x, center_y = center
    else:
        # Note: center_x corresponds to width and center_y to height.
        center_x, center_y = width // 2, height // 2

    # Create grid coordinates
    y, x = np.ogrid[:height, :width]
    # Adjust y-coordinates so that increasing y goes upward relative to the center
    flipped_y = center_y - y

    # Compute distance from the center for each pixel
    dist_from_center = np.sqrt((x - center_x)**2 + flipped_y**2)

    # Compute the angle for each pixel (in radians, normalized to [0, 2π])
    theta = np.arctan2(flipped_y, x - center_x)  # returns angle in [-π, π]
    theta = (theta + 2 * np.pi) % (2 * np.pi)       # normalize to [0, 2π]

    # ---------------------------
    # Create azimuthal (angle) bins
    # ---------------------------
    segment_angle = 2 * np.pi / nbins_azimuthal
    # Preallocate an array to store the angle ranges for each azimuthal bin
    angle_ranges_rad = np.empty((nbins_azimuthal, 2))
    for n in range(nbins_azimuthal):
        start_angle = (rotation + n * segment_angle) % (2 * np.pi)
        end_angle = (rotation + (n + 1) * segment_angle) % (2 * np.pi)
        angle_ranges_rad[n] = [start_angle, end_angle]

    # ---------------------------
    # Create radial bins
    # ---------------------------
    radial_step = (outer_radius - inner_radius) / nbins_radial
    # Preallocate an array to store the radial ranges for each bin
    radial_bins = np.empty((nbins_radial, 2))
    for i in range(nbins_radial):
        r_min = inner_radius + i * radial_step
        r_max = inner_radius + (i + 1) * radial_step
        radial_bins[i] = [r_min, r_max]

    # ---------------------------
    # Create masks for each polar bin (combination of radial and azimuthal)
    # ---------------------------
    # Total number of regions = nbins_radial * nbins_azimuthal
    masks = np.zeros((nbins_radial * nbins_azimuthal, height, width), dtype=bool)

    region = 0
    for i in range(nbins_radial):
        # Create radial mask for the i-th radial bin
        radial_mask = (dist_from_center >= radial_bins[i, 0]) & (dist_from_center < radial_bins[i, 1])
        for j in range(nbins_azimuthal):
            # Create azimuthal mask for the j-th angular bin
            start_angle, end_angle = angle_ranges_rad[j]
            if start_angle < end_angle:
                azimuthal_mask = (theta >= start_angle) & (theta < end_angle)
            else:
                # Handle the wrap-around (e.g., when the bin spans the 2π -> 0 boundary)
                azimuthal_mask = (theta >= start_angle) | (theta < end_angle)
        
            # Combine the radial and azimuthal masks for the current region
            masks[region] = radial_mask & azimuthal_mask
            region += 1
            
    region_map = np.full((height, width), -1, dtype=int)
    for idx in range(masks.shape[0]):
        region_map[masks[idx]] = idx

    # ---------------------------
    # Plot the composite region map using a discrete colormap
    # ---------------------------
    plt.figure(figsize=(8, 8))
    # Choose a colormap: 'tab10' works well for up to 10 regions; 'tab20' for more
    if masks.shape[0] <= 10:
        cmap = plt.get_cmap('tab10', masks.shape[0])
    else:
        cmap = plt.get_cmap('tab20', masks.shape[0])

    # Display the region map; note vmin and vmax are set to include the background (-1)
    im = plt.imshow(region_map, cmap=cmap, vmin=-1, vmax=masks.shape[0]-1)
    cbar = plt.colorbar(im, ticks=range(-1, masks.shape[0]))
    cbar.ax.set_yticklabels(['Background'] + [f"Region {i}" for i in range(masks.shape[0])])
    plt.title("Polar Bin Regions")
    plt.axis('off')
    plt.show()
    return masks

def radialAverage(IMG, cx, cy, w):
        """
        computes the radial average of the image IMG around the [cx, cy] point
        w is the size of vector of radii starting from zero
        """
        if isinstance(IMG, cp.ndarray):
            print("The input array image should have a format of np.ndarray!")
            print("You should use np.ndarray(IMG) to convert the array with a cupy.array")
        a, b = IMG.shape
        Y, X = np.meshgrid(np.arange(1, a + 1) - cx, np.arange(1, b + 1) - cy)
        R = np.sqrt(X**2 + Y**2)
        profile = []
        for i in range(w): #radius of the circle
            mask = (i < R) & (R < (i + 1)) #smooth 1 px around the radius
            values = (1 - np.abs(R[mask] - i)) * IMG[mask] #smooth based on distance to ring
            profile.append(np.mean(values))
        return profile

def coordinates_segment_in_DPC(collection, kBF, segment, wavelength, center, N=(128,128)):    

    # Convert rad to 1/nm using theta/lambda
    kBF /= wavelength
    k_DPC_min = collection[0]
    k_DPC_max = collection[1]
    k_DPC_min *= 0.5 / wavelength
    k_DPC_max *= 0.5 / wavelength
    
    VNx = np.linspace(-N[0]//2, N[0]//2, N[0])
    VNy = np.linspace(-N[1]//2, N[1]//2, N[1])
    Ncx = N[0] // 2
    Ncy = N[1] // 2
    dkx = 2 * kBF / Ncx
    dky = 2 * kBF / Ncy
    ekx = VNx * dkx
    eky = VNy * dky
    ky, kx = np.meshgrid(eky, ekx)
    ksquare = kx ** 2 + ky ** 2
    knorm = np.sqrt(ksquare)
   
    condition = segmented_circular_masks(N, center, k_DPC_min, k_DPC_max, angle_range=segment)

    return kx*condition[0], ky*condition[0]


class COMProcessor:
    """
    A class for integrated center of mass (iCOM) image reconstruction.

    The intensities of segments are extracted by simulating a segmented detector.
    The integrated intensity is calculated as:
        I = sum( I(k, r) * D(k) ),
        where D(k) is the detector response function.

    The CoM (center of mass) in the diffraction pattern is:
        K_CoM = sum( ki * Ii ) / sum(Ii),
        where ki = (ki_x, ki_y) is the pixel position and Ii is the intensity at that pixel.

    Usage example:
        COM = COMProcessor(array_data, center=None, nbins_radial=4, nbins_azimuthal=4, device='cpu')
        segs, coms = COM.segment_intensities(inner=10, outer=50)
        # <segs> stores intensities of virtual segmented images, shape (N, nx, ny)
        # <coms> stores CoMx, CoMy, shape (2, nx, ny)
        COM.visualize(plot='segments')
        # Options: plot='segments', 'center', or 'masks'
    """
    def __init__(self, datacube, center, nbins_radial, nbins_azimuthal, rotation=np.pi/4, device="cpu", return_cpu=True):
        """
        Initialize the COMProcessor.

        Args:
            datacube (ndarray): 4D array of shape (nx, ny, px, py) containing diffraction patterns.
            center (tuple or None): (x, y) coordinates of the center; if None, computed automatically.
            nbins_radial (int): Number of radial bins.
            nbins_azimuthal (int): Number of azimuthal bins.
            rotation (float): Rotation offset for azimuthal bins in radians (default: pi/4).
            device (str): 'cpu' or 'gpu' for computation device (default: 'cpu').
            return_cpu (bool): If True, return NumPy arrays; if False, return CuPy arrays when device='gpu' (default: True).
        """
        if datacube.ndim != 4:
            raise ValueError("Datacube must be a 4D array with shape (nx, ny, px, py)")
        self.nx, self.ny, self.px, self.py = datacube.shape
        self.device = device.lower()
        if self.device not in ["cpu", "gpu"]:
            raise ValueError("Device must be either 'cpu' or 'gpu'")
        self.center = center
        self.nbins_radial = nbins_radial
        self.nbins_azimuthal = nbins_azimuthal
        self.rotation = rotation
        self.return_cpu = return_cpu
        self.xp = np if self.device == "cpu" else cp
        self.datacube = self.xp.array(datacube)
        self.average = self.xp.average(self.datacube, axis=(0, 1))

    def _return_array(self, array):
        """
        Return array in the desired format based on return_cpu.

        Args:
            array: Input array (NumPy or CuPy).

        Returns:
            ndarray: NumPy array if return_cpu=True, otherwise CuPy array if device='gpu'.
        """
        if self.return_cpu:
            return array.get() if isinstance(array, cp.ndarray) else array
        return array

    def find_center(self, image, mask_size=None):
        """
        Find the refined center of mass of an image.

        Args:
            image (ndarray): 2D diffraction pattern.
            mask_size (float or None): Radius of the circular mask; if None, defaults to 80% of min distance to edge.

        Returns:
            tuple: (x, y) coordinates of the refined center.
        """
        if image.ndim != 2:
            raise ValueError("Input image must be a 2-dimensional array")
        x, y = image.shape

        # Initial CoM
        if self.device == 'cpu':
            center = ndimage.center_of_mass(image)
        else:
            center = cpndimage.center_of_mass(image)

        rmax = min(center[0], center[1], abs(y - center[0]), abs(x - center[1]))
        r = mask_size if mask_size is not None else rmax * 0.8

        # Create circular mask
        mask = self.xp.zeros((self.px, self.py))
        y_grid, x_grid = self.xp.ogrid[:self.py, :self.px]
        distance_from_center = (x_grid - center[1])**2 + (y_grid - center[0])**2
        mask[distance_from_center <= r**2] = 1

        # Refine CoM
        if self.device == 'cpu':
            refined_center = ndimage.center_of_mass(image * mask)
        else:
            refined_center = cpndimage.center_of_mass(image * mask)

        return refined_center[1], refined_center[0]

    def segmented_circular_masks(self, inner_radius, outer_radius):
        """
        Create segmented circular masks for radial and azimuthal bins.

        Args:
            inner_radius (float): Inner radius of the annular region.
            outer_radius (float): Outer radius of the annular region.

        Returns:
            ndarray: Masks of shape (nbins_radial * nbins_azimuthal, px, py).
        """
        height, width = self.px, self.py
        if self.center is None:
            self.center = self.find_center(self.average)
        center_x, center_y = self.center

        y, x = self.xp.ogrid[:height, :width]
        flipped_y = center_y - y  # Cartesian convention: y increases upwards
        dist_from_center = self.xp.sqrt((x - center_x)**2 + flipped_y**2)
        theta = self.xp.arctan2(flipped_y, x - center_x)
        theta = (theta + 2 * self.xp.pi) % (2 * self.xp.pi)  # Normalize to [0, 2π]

        segment_angle = 2 * self.xp.pi / self.nbins_azimuthal
        angle_ranges_rad = self.xp.empty((self.nbins_azimuthal, 2))
        for n in range(self.nbins_azimuthal):
            start_angle = (self.rotation + n * segment_angle) % (2 * self.xp.pi)
            end_angle = (self.rotation + (n + 1) * segment_angle) % (2 * self.xp.pi)
            angle_ranges_rad[n] = [start_angle, end_angle]

        radial_step = (outer_radius - inner_radius) / self.nbins_radial
        radial_bins = self.xp.empty((self.nbins_radial, 2))
        for i in range(self.nbins_radial):
            r_min = inner_radius + i * radial_step
            r_max = inner_radius + (i + 1) * radial_step
            radial_bins[i] = [r_min, r_max]

        masks = self.xp.zeros((self.nbins_radial * self.nbins_azimuthal, height, width), dtype=bool)
        region = 0
        for i in range(self.nbins_radial):
            radial_mask = (dist_from_center >= radial_bins[i, 0]) & (dist_from_center < radial_bins[i, 1])
            for j in range(self.nbins_azimuthal):
                start_angle, end_angle = angle_ranges_rad[j]
                if start_angle < end_angle:
                    azimuthal_mask = (theta >= start_angle) & (theta < end_angle)
                else:
                    azimuthal_mask = (theta >= start_angle) | (theta < end_angle)
                masks[region] = radial_mask & azimuthal_mask
                region += 1

        return masks

    def merge_ronchgrams(self, crop_size=None):
        """
        Merge all diffraction patterns into a single array for plotting.

        Args:
            crop_size (int or tuple or None): Size to crop each pattern (width, height); if None, uses full size.

        Returns:
            ndarray: Merged array of shape (nx * crop_x, ny * crop_y).

        Note:
            Original code used an undefined 'crop_matrix'. Here, basic cropping is implemented.
            Replace with custom crop_matrix if intended.
        """
        rows, cols = self.nx, self.ny
        if crop_size is None:
            crop_x, crop_y = self.px, self.py
        else:
            crop_x, crop_y = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size

        cube = np.zeros((rows * crop_x, cols * crop_y))
        center_x, center_y = self.center if self.center else (self.px // 2, self.py // 2)
        half_x, half_y = crop_x // 2, crop_y // 2

        for i in range(rows):
            for j in range(cols):
                cube[i * crop_x:(i + 1) * crop_x, j * crop_y:(j + 1) * crop_y] = crop_matrix(self.datacube[i, j],(0,1), self.center, [crop_x, crop_y])

        return cube

    def segment_intensities(self, inner_radius, outer_radius):
        """
        Extract segment intensities and compute CoM for each diffraction pattern.

        Args:
            inner_radius (float): Inner radius of the annular region.
            outer_radius (float): Outer radius of the annular region.

        Returns:
            tuple: (segments, coms) where segments is shape (N, nx, ny) and coms is shape (2, nx, ny).
        """
        self.mask = self.segmented_circular_masks(inner_radius, outer_radius)
        num = self.nbins_radial * self.nbins_azimuthal
        qx, qy = self.xp.meshgrid(self.xp.arange(self.px), self.xp.arange(self.py))  # Note: qx=x, qy=y
        self.segments = self.xp.zeros((num, self.nx, self.ny))

        if self.nbins_azimuthal >1 :
            self.COMs = self.xp.zeros((2, self.nx, self.ny))
            for i in tqdm(range(self.nx), desc="Extracting intensities", unit="row"):
               for j in range(self.ny):
                    DP = self.datacube[i, j, :, :]
                    total_intensity = self.xp.sum(DP)
                    if total_intensity > 0:
                        self.COMs[0, i, j] = self.xp.sum(DP * qx) / total_intensity  # CoMx
                        self.COMs[1, i, j] = self.xp.sum(DP * qy) / total_intensity  # CoMy
                    else:
                        self.COMs[0, i, j] = 0
                        self.COMs[1, i, j] = 0
                    for n in range(num):
                        self.segments[n, i, j] = self.xp.sum(DP * self.mask[n])

            return self._return_array(self.segments), self._return_array(self.COMs)
        else: # when the nbins_azimuthal =1, representing the non-segmented mask
            for i in tqdm(range(self.nx), desc="Extracting intensities", unit="row"):
                for j in range(self.ny):
                    DP = self.datacube[i, j, :, :]
                    for n in range(num):
                        self.segments[n, i, j] = self.xp.sum(DP * self.mask[n])
            return self._return_array(self.segments), None

    def visualize(self, plot='segments'):
        """
        Visualize segments, center, or masks.

        Args:
            plot (str): 'segments', 'center', or 'masks' (default: 'segments').
        """
        if plot == 'segments':
            segments = cp.asnumpy(self.segments)
            num = segments.shape[0]
            if num > 1:
                m = int(np.sqrt(num))
                n = m + 1 if m**2 < num else m
                fig, axes = plt.subplots(m, n, sharex=True, sharey=True)
                for i, ax in enumerate(axes.ravel()):
                    if i < num:
                        pcm = ax.imshow(segments[i], cmap='viridis', interpolation='gaussian')
                        ax.set_title(f'Virtual image {i+1}')
                        fig.colorbar(pcm, ax=ax, shrink=0.9)
                        ax.axis('off')
            else:
                plt.imshow(segments[0], cmap='viridis', interpolation='gaussian')
                plt.title('Virtual image')
                plt.axis('off')
            plt.tight_layout()
            plt.show()
        elif plot == 'center':
            center_x, center_y = self.center if self.center else self.find_center(self.average)
            center_x, center_y = cp.asnumpy(center_x), cp.asnumpy(center_y)
            fig, ax = plt.subplots()
            ax.imshow(cp.asnumpy(self.average), cmap='gray')
            ax.scatter(center_x, center_y, color='red', label='Refined Center')
            circle = plt.Circle((center_x, center_y), radius=self.px * 0.4, fill=False, color='red', linestyle='--')
            ax.add_patch(circle)
            plt.legend()
            plt.show()
        else:  # 'masks'
            seg_mask = cp.asnumpy(self.mask)
            num = seg_mask.shape[0]
            if num > 1:
                m = int(np.sqrt(num))
                n = m + 1 if m**2 < num else m
                fig, axes = plt.subplots(m, n, sharex=True, sharey=True)
                for i, ax in enumerate(axes.ravel()):
                    if i < num:
                        pcm = ax.imshow(seg_mask[i])
                        ax.set_title(f'Mask {i+1}')
                        fig.colorbar(pcm, ax=ax, shrink=0.9)
                        ax.axis('off')
            else:
                plt.imshow(seg_mask[0])
                plt.title('Virtual detector')
                plt.axis('off')
            plt.tight_layout()
            plt.show()


def DFT_analysis(image, properties=None):
    """
    Return:
    d_sapcing
    filtered_image
    """
    if properties is None:
        properties = {
            'resolution': 0.0131411635412124,
            'unit': 'nm',
            'figsize': 8,
            'cmap': 'viridis',
            'max_sigma': 5,
            'threshold_abs': 0.2,
            'Zoom_in': 2,
            'mask size': 20,
            'inverse FFT': True,
            'print d-spacings': False
        }

    default_properties = {
        'resolution': 0.0131411635412124,
        'unit': 'nm',
        'figsize': 8,
        'cmap': 'viridis',
        'sigma_max': 5,
        'threshold_abs': 0.2,
        'Zoom_in': 2,
        'mask size': 10,
        'inverse FFT': True,
        'print d-spacings': False
    }

    for key, value in default_properties.items():
        properties.setdefault(key, value)

    color = properties['cmap']
    unit = properties['unit']
    inverse = properties['inverse FFT']
    print_d = properties['print d-spacings']
    height, width = image.shape
    Zoom_in = properties['Zoom_in']
    FOV_inverse = 1 / (height * properties['resolution'])

    FFT = np.fft.fftshift(np.fft.fft2(image))  # Compute FFT

    gaussian_1 = ndimage.gaussian_filter(np.log(np.abs(FFT) + 1), sigma=4)
    gaussian_2 = ndimage.gaussian_filter(gaussian_1, sigma=1)
    diffraction_image = gaussian_2 + 80*(gaussian_1 - gaussian_2)
    diffraction_image = (diffraction_image - np.min(diffraction_image)) / (
            np.max(diffraction_image) - np.min(diffraction_image))  # Normalize
    coordinates = blob_log(diffraction_image, max_sigma=properties['max_sigma'], threshold=properties['threshold_abs']/10)
    spots_coor = coordinates[:, :2]
    
    if inverse:
        filters = np.zeros((height, width))
        for blob in coordinates:
            by, bx, br = blob
            built_filter = circle_mask((height, width), (by, bx), (0, properties['mask size']))
            filters += built_filter

        filtered_image = FFT * filters
        filtered_result = np.fft.ifft2(np.fft.ifftshift(filtered_image))

    intensities = [diffraction_image[int(y), int(x)] for y, x in spots_coor]
    Direct_beam_coordinate = [height / 2, width / 2]
    spots = [[-(coor[1] - width / 2), (coor[0] - height / 2)] for coor in spots_coor]

    fig, axs = plt.subplots(1, 3 if inverse else 2, figsize=(properties['figsize'] * (3 if inverse else 2),
                                                           properties['figsize']))

    fft_for_show = crop_matrix(diffraction_image*filters,(0,1), [int(height /2), int(width /2)],
                               [height // Zoom_in, width // Zoom_in])
    X_crop, Y_crop = fft_for_show.shape
    axs[0].imshow(fft_for_show, cmap=color)
    scale_bar1 = ScaleBar(properties['resolution'], units='1/' + unit, dimension='si-length-reciprocal',
                          length_fraction=0.2, location='lower left', scale_loc='top')
    axs[0].add_artist(scale_bar1)
    axs[0].set_title('DFT image')
    axs[0].axis('off')

    axs[1].imshow(fft_for_show, cmap=color)
    axs[1].scatter(np.array(spots_coor)[:, 1] - width / 2 + X_crop / 2, np.array(spots_coor)[:, 0] - height / 2 + Y_crop / 2, c='red', alpha=0.3,
                   label='spots')
    j = 0
    d_spacing = []
    for i, spot in enumerate(spots):
        if spot[1] > 0:
            j += 1
            axs[1].annotate(str(j), (coordinates[i][1] - width // 2 + X_crop // 2, coordinates[i][0] - height // 2 + Y_crop // 2))
            d = np.sqrt((spot[0]) ** 2 + (spot[1]) ** 2) * FOV_inverse
            d_spacing.append([j, 1 / d])
            if print_d:
                print(f'The spot {j} has a d_spacing of {(1 / d) * 10:.2f} Å')

    axs[1].scatter(Direct_beam_coordinate[1] - width // 2 + X_crop // 2, Direct_beam_coordinate[0] - height // 2 + Y_crop // 2, c='yellow')
    axs[1].set_title('Filters for inverse DFT')
    axs[1].axis('off')

    if inverse:
        axs[2].imshow(filtered_result.real, cmap=color)
        scale_bar2 = ScaleBar(properties['resolution'], units=unit, dimension='si-length', length_fraction=0.2,
                              location='lower left', scale_loc='top')
        axs[2].add_artist(scale_bar2)
        axs[2].set_title('Inverse DFT image')
        axs[2].axis('off')

    plt.tight_layout()
    plt.show()

    if inverse:
        return d_spacing, filtered_result.real
    else:
        return d_spacing, None


def find_local_max(image, points, neighbor_size, threshold=1):
    """
    Function is utilized to locate the point with the maximum intensity.
    Finding the maximum intensity within a given area, which the assumed center of this area is provided by 'points',
    The size of the area is determined by neighbor_size.
    If the searched point is dim, then it would be abandoned using the ratio.
    Args:

    image: np.ndarray

    points: np.ndarray, or list with np.ndarray elements

    neighbor_size: a list or a tuple, containing two figures, which determing the size of searching area

    ratio: a float, the intensity of the found point to the average intensity of the searched area
    """
    sizex, sizey = image.shape
    coordinates = []
    if isinstance(points, list):
        if len(points)>2:
            for point in points:
                if isinstance(point, np.ndarray):
                    if point.shape ==(2,):
                        x, y = point[0], point[1]
                    elif point.shape==(2,1):
                        x, y = point[0][0], point[0][1]
                else:
                    x, y = point[0], point[1]

                start_x = max(0, round(x - neighbor_size[0]))
                end_x = min(sizex-1, round(x + neighbor_size[0]))
                start_y = max(0, round(y - neighbor_size[1]))
                end_y = min(sizey-1, round(y + neighbor_size[1]))

                grid_x = np.linspace(start_x, end_x - 1, end_x - start_x, dtype=int)
                grid_y = np.linspace(start_y, end_y - 1, end_y - start_y, dtype=int)

                if len(grid_x) * len(grid_y) == 0:
                    coordinates.append(np.array([x, y]))
                else:
                    sub_image = np.zeros((len(grid_x), len(grid_y)))
                    for i, gx in enumerate(grid_x):
                        for j, gy in enumerate(grid_y):
                            if ((gx - x) ** 2 + (gy - y) ** 2) < neighbor_size[0] ** 2:
                                sub_image[i, j] = image[gy, gx]

                max_coords = peak_local_max(sub_image, min_distance=1, threshold_abs=0, num_peaks=1)
            
                if isinstance(max_coords, np.ndarray) and max_coords.shape == (1, 2):
                    max_point = np.array([max_coords[0][1], max_coords[0][0]])  
                else:
                    max_point = (max_coords[1], max_coords[0])                          

                if sub_image[max_point[0], max_point[1]] > threshold*np.mean(crop_matrix(image,(0,1), [int(sizex/4), int(sizey/4)], [int(sizex/2), int(sizey/2)])):
                    coordinates.append(np.array([max_point[0] + start_x, max_point[1] + start_y]))
                else: coordinates.append(np.array([x, y]))  

    return coordinates

def excluding_spots(given_spot, points, k1, k2, delta=0.7):

    """
    It is used to exclude spots on diffraction patterns or FT domains.

    Args:

    given_spot: np.ndarray(), recorded the coordinate of given spot in pixel

    points: an array list, with a type of np.ndarray, whose shape is (m, 2) with m number of members.

    k1, k2: two basic vectors of the cell structure

    delta: a float, determining the minimum distance for excluding the spot
    """
    
    def find_nearest(given_spot, points):
        distances = np.linalg.norm(points - given_spot, axis=1)
        nearest_indices = np.argsort(distances.flatten())[:3]
        # Retrieve the two nearest points
        nearest_points = points[nearest_indices]
        return [nearest_points[1], nearest_points[2]]
    
    given_points = find_nearest(given_spot, points)
    L1 = np.linalg.norm(given_points[0] - given_spot)
    L2 = np.linalg.norm(given_points[1] - given_spot)

    d1 = np.linalg.norm(k1)
    d2 = np.linalg.norm(k2)
    d3 = np.linalg.norm(k1+k2)

    if min(L1, L2) > delta*min(d1, d2, d3):
        return given_spot


def line_intensity_profile(image, names, resolution, width=5, length_unit='nm', zoom_in = False):
    """
    Drawing the line profiles of images

    Return:
          (x_coordinates, intensities)

    """

    if isinstance(image, list) and len(image)>=1:
        plt.imshow(image[0])
    else: plt.imshow(image)
        
    if zoom_in:
        num =3
    else: num =2
        
    points = np.asarray(plt.ginput(num, timeout=-1))
    plt.show()
    plt.close()
    # Generate the coordinates of the line using Bresenham's algorithm
    start_point = points[-2]
    end_point = points[-1]
    x0, y0 = int(start_point[0]), int(start_point[1])
    x1, y1 = int(end_point[0]), int(end_point[1])
    line_length = end_point-start_point
    length = np.sqrt(line_length[0]**2+line_length[1]**2)
    num = max(abs(x1 - x0), abs(y1 - y0))
    rows = np.linspace(x0, x1, num)
    columns = np.linspace(y0, y1, num)
    line_coordinates = np.column_stack((rows, columns))                                        
    # Create a mask for the line width
    mask = np.ones(width)
    # Extract pixel values along the line
    profiles_list = []
    if isinstance(image, list) and len(image)>=1:        
        for n in range(len(image)):
            line_values = linscale(image[n], nmin=0, nmax=1)[np.round(line_coordinates[:, 1]).astype(int), np.round(line_coordinates[:, 0]).astype(int)]
            # Compute the average intensity profile across the width
            intensity_profile = np.convolve(line_values, mask, mode='same') / width
            x_coor = np.linspace(0,length*resolution, len(intensity_profile))
            profiles_list.append((x_coor, intensity_profile))
    else: 
        line_values = image[np.round(line_coordinates[:, 1]).astype(int), np.round(line_coordinates[:, 0]).astype(int)]
        intensity_profile = np.convolve(line_values, mask, mode='same') / width
        x_coor = np.linspace(0,length*resolution, len(intensity_profile))
        profiles_list.append((x_coor, intensity_profile))  
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    
    for pair in range(len(profiles_list)):
        x_values = profiles_list[pair][0]
        y_values = profiles_list[pair][1]/(np.max(profiles_list[pair][1])-np.min(profiles_list[pair][1]))
        ax1.plot(x_values, pair +y_values, label=f'{names[pair]}')
    ax1.set_title('Intensity Profile of the Line')
    ax1.set_xlabel('length (nm)')
    ax1.set_ylabel('Intensity')
    ax1.legend()
    ax2.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red', linewidth=width)

    if isinstance(image, list) and len(image)>=1:   
        ax2.imshow(image[0])
    else: ax2.imshow(image) 
    ax2.axis('off')
    scale_bar = ScaleBar(resolution, units= length_unit, dimension = 'si-length', length_fraction = 0.2, \
                         location = 'lower left',scale_loc = 'top')
    ax2.add_artist(scale_bar)
    fig.tight_layout()
    plt.show()
    return profiles_list


def rotating_image(image, angle, centerX, centerY, cropped_size):
    """
    Rotates and crops an image interactively.
    
    Parameters:
    - image: The input image (2D numpy array)
    - angle: The angle of rotation (degrees)
    - centerX, centerY: The center of cropping
    - cropped_size: turple, Size of the cropped image
    - show_plot: Whether to display the plot (default: True)
    
    Returns:
    - The cropped rotated image
    
    """
    if isinstance(image, cp.ndarray):
        image = cp.asnumpy(image)
        
    X, Y = image.shape
    rotated_img = ndimage.rotate(image, angle, mode = 'constant')
    NX, NY = rotated_img.shape
    if cropped_size[0] < NX and cropped_size[1] < NY:
        sizeX, sizeY = cropped_size
    else: 
        sizeX = int(min(centerX, cropped_size[0]))
        sizeY = int(min(centerY, cropped_size[1]))
        
    length = (centerX - X//2)**2 + (centerY - Y//2)**2
    if centerX != X//2 and centerY != Y//2:
        kx = (centerX-X//2)
        ky = (centerY-Y//2)
        #slope = ky/kx
        #Sangle = np.arctan(slope)
        Sangle = (np.arctan2(np.abs(ky), np.abs(kx)) * (-2 * (kx < 0) + 1) + np.pi * (kx < 0)) * (-2 * (ky < 0) + 1)
        Nslope = np.tan(-np.radians(angle)+Sangle)
        if kx<0:
            newX = NX//2 - int(np.sqrt(length/(1+Nslope**2)))
        elif kx>0:
            newX = NX//2 + int(np.sqrt(length/(1+Nslope**2)))
        if ky <0:
            newY = NY//2 - int(np.sqrt(length/(1+(1/Nslope)**2)))
        elif ky>0:
            newY = NY//2 + int(np.sqrt(length/(1+(1/Nslope)**2)))
          
    else: 
        newX = NX//2
        newY = NY//2
    
    cropped = crop_matrix(rotated_img,(0,1), [int(newX), int(newY)], [sizeX, sizeY])
    croppedX, croppedY = cropped.shape

    fig, axes =  plt.subplots(1,2, figsize=(16, 8))
    axes[0].imshow(rotated_img)
    axes[0].plot(newX, newY, marker='+', markersize=20, color='r')
    rect = patches.Rectangle((int(np.abs(newX-sizeX/2)), int(np.abs(newY-sizeY/2))), sizeX, sizeY, linewidth=2, edgecolor='r', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].set_title(f'Rotating {round(angle,1)} degrees')
    axes[1].imshow(cropped)
    axes[1].plot(croppedX/2, croppedY/2, marker='+', markersize=20, color='r')
    axes[1].set_title(f'Zoom in {int(X/sizeX)} times')
    plt.show()
    return cropped

def interactive_rotating_image(image):
    """
    Interactive Widget for Rotating Image
    """
    def wrapper(angle, centerX, centerY, cropX, cropY):
        cropped_size = (cropX, cropY)
        return rotating_image(image, angle, centerX, centerY, cropped_size)    
    interact(wrapper, 
             angle=widgets.FloatSlider(min = -45, max = 45, value = 0, layout=widgets.Layout(width='80%')),
             centerX = widgets.IntSlider(min = 0, max = image.shape[0], value = image.shape[0]//2, layout=widgets.Layout(width='80%')),
             centerY = widgets.IntSlider(min = 0, max = image.shape[1], value = image.shape[1]//2, layout=widgets.Layout(width='80%')),
             cropX = widgets.IntSlider(min = 16, max = image.shape[0], value = image.shape[0]//8, layout=widgets.Layout(width='80%')),
             cropY = widgets.IntSlider(min = 16, max = image.shape[1], value = image.shape[1]//8, layout=widgets.Layout(width='80%'))
            )
    
def plot_fields_map(CoMx, CoMy, resolution, arrow_size=200, threshold=0.8, unit="nm", cmap='red'):
    """
    It plots the electric fields of DPC using arrows.
    
    Args:
        CoMx: the component of DPC in x-axis 
        CoMy: the component of DPC in y-axis 
        resolution: float, length in pixel
        arrow_size: float, the size of arrows
        unit: string, the unit of the resolution
        threshold: float, plotting arraows based on the intensities
        cmap: string, setting the colour of arrows
    """
    if isinstance(CoMx, cp.ndarray):
        CoMx = linscale(cp.asnumpy(CoMx))
    else: CoMx = linscale(CoMx)
    if isinstance(CoMx, cp.ndarray):
        CoMx = linscale(cp.asnumpy(CoMx))
    else: CoMx = linscale(CoMx)
        
    ex = np.arange(CoMx.shape[0])  
    ey = np.arange(CoMx.shape[1])
    x, y = np.meshgrid(ex*resolution, ey*resolution)     

    dpc = ex[:, None] *CoMx + ey[None, :] *CoMy
    magnitude = np.linalg.norm(np.array([CoMx, CoMy]), axis=0)
    mask = magnitude > threshold
    inverted_magnitude = np.reciprocal(magnitude, where=magnitude != 0)
    inverted_magnitude[magnitude==0]=0

    fig, axs = plt.subplots(1, 2, figsize=(16, 8), sharex=True, layout='constrained')
    lx, ly = CoMx.shape
    extend = [0, lx*resolution, ly*resolution, 0]
    vm = axs[0].imshow(dpc, cmap='viridis', extent=extend)
    im = axs[1].imshow(inverted_magnitude, cmap='viridis', extent=extend)
    #scale_bar = ScaleBar(resolution, units=label, dimension = 'si-length', length_fraction = 0.2, location = 'lower left', scale_loc = 'top')
    #axs[0].add_artist(scale_bar)
    #axs[1].add_artist(scale_bar)
    axs[0].quiver(x[mask], y[mask], CoMx[mask], CoMy[mask],  color=cmap, pivot='middle', scale=arrow_size)
    axs[1].quiver(x[mask], y[mask], CoMx[mask], CoMy[mask],  color=cmap, pivot='middle', scale=arrow_size)

    colorbar_vm = fig.colorbar(vm, ax=axs[0], location='right', shrink=0.5, label='Magnitude')
    colorbar_vm.minorticks_on()
    colorbar_im = fig.colorbar(im, ax=axs[1], location='right', shrink=0.5, label='Inverted Magnitude')  
    colorbar_im.minorticks_on()
    axs[0].set_xlabel(f"Length ({unit})")
    axs[0].set_title('DPC image')
    axs[1].set_xlabel(f"Length ({unit})")
    axs[1].set_title('DPC vector fields map')
    plt.show()
    print("The high magnitude arrows in Inverted Magnitude are interpreted as electric fields toward accumulated negative charge.")
    return dpc

class PoreAnalyzer:
    """
    # Example usage:
    # img = cv2.imread('image_path', cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread('image_path', cv2.IMREAD_GRAYSCALE)
    # analyzer = PoreAnalyzer(img, img2)
    # analyzer.analyze_pores()
    # pore_sizes = analyzer.get_pore_sizes()
    # pore_intensities = analyzer.get_pore_intensities()
    # mask = analyzer.get_mask()
    """
    def __init__(self, img, img2, mask=None):
        self.mask = mask
        self.pore_sizes = []
        self.pore_intensities = []
        self.hist_pore_sizes = None
        self.hist_intensities = None
        self.img_1 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        self.img_2 = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX)

        # Binary Image
        self.img_uint8 = self.img_1.astype('uint8')
    def updating_parameters(self, blur_size, min_gap, min_pore_size, bins, mask_shrink):
        """
        Perform statistics on the pore size distribution in images.
        """
        
        if blur_size % 2 == 0:
            blurred = cv2.GaussianBlur(self.img_uint8, (blur_size - 1, blur_size - 1), 0)
        elif blur_size == 1:
            blurred = self.img_uint8
        else:
            blurred = cv2.GaussianBlur(self.img_uint8, (blur_size, blur_size), 0)

        binary = cv2.adaptiveThreshold(self.img_uint8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        self.binary = binary

        # Distance Transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
        self.dist_transform = dist_transform

        # Find local maxima
        coordinates = peak_local_max(dist_transform, min_distance=min_gap, labels=binary)
        local_max = np.zeros_like(dist_transform, dtype=bool)
        local_max[tuple(coordinates.T)] = True

        # Watershed
        markers, _ = ndimage.label(local_max)
        labels = watershed(-dist_transform, markers, mask=binary)

        # Create mask overlay
        if self.mask is None:
            self.mask = np.zeros_like(self.img_1, dtype=np.uint8)
            for label in range(1, labels.max() + 1):
                pore_mask = labels == label
                if np.sum(pore_mask) > min_pore_size:
                    self.mask[pore_mask] = 255

        # Shrink the mask if required
        if mask_shrink > 0:
            kernel = np.ones((3, 3), np.uint8)
            self.shrinked_mask = cv2.erode(self.mask, kernel, iterations=mask_shrink)
        else: 
            self.shrinked_mask = self.mask
        # Overlay mask on original image
        overlay = cv2.cvtColor(self.img_2.astype('uint8'), cv2.COLOR_GRAY2RGB)
       
        overlay[self.shrinked_mask > 0] = [255, 0, 0]  # Red color for the mask
        self.overlay = overlay

        self.pore_sizes = []
        self.pore_intensities = []

        for label in range(1, labels.max() + 1):
            pore_mask = (labels == label) & (self.shrinked_mask > 0)
            size = np.sum(pore_mask)
            if size > min_pore_size:
                self.pore_sizes.append(size)
                self.pore_intensities.append(np.sum(self.img_2[pore_mask]))

        self.hist_pore_sizes, self.pbin_edges = np.histogram(self.pore_sizes, bins=bins)
        self.hist_intensities, self.inbin_edges = np.histogram(self.pore_intensities, bins=bins)
        self.num = len(self.pore_sizes)
        
    def display_images(self):
        fig = plt.figure(figsize=(10, 10), layout="constrained")
        spec = fig.add_gridspec(2, 2)
        # Original Image
        ax1 = fig.add_subplot(spec[0, 0])
        ax1.imshow(self.img_1, cmap='gray')
        ax1.set_title('Image for model')
        ax1.axis('off')
        ax2 = fig.add_subplot(spec[0, 1])
        ax2.imshow(self.binary, cmap='gray')
        ax2.set_title('Binary')
        ax2.axis('off')
        ax3 = fig.add_subplot(spec[1, 0])
        ax3.imshow(self.dist_transform, cmap='jet')
        ax3.set_title('Distance Transform')
        ax3.axis('off')
        ax4 = fig.add_subplot(spec[1, 1])
        ax4.imshow(self.overlay)
        ax4.set_title('Mask Overlay on Original Image')
        ax4.axis('off')
        plt.show()
        
    def show_statistics(self):
        fig = plt.figure(figsize=(10, 10), layout="constrained")
        spec = fig.add_gridspec(1, 2)
        ax5 = fig.add_subplot(spec[0, 0])
        ax5.bar(self.pbin_edges[:-1], self.hist_pore_sizes, width=np.diff(self.pbin_edges), align="edge")
        ax5.set_title(f"Pore Size Distribution with {self.num}")
        ax5.set_xlabel("Pore Size (pixels)")
        ax5.set_ylabel("Frequency")

        ax6 = fig.add_subplot(spec[0, 1])
        ax6.bar(self.inbin_edges[:-1], self.hist_intensities, width=np.diff(self.inbin_edges), align="edge")
        ax6.set_title(f"Intensity Distribution within Pores ({self.num})")
        ax6.set_xlabel("Intensity (a.u.)")
        ax6.set_ylabel("Frequency")
        plt.show()
        
    def analyze_pores(self, blur_size=1, min_gap=20, min_pore_size=16, bins=20, mask_shrink=2):
        def update_conditions(blur_size, min_gap, min_pore_size, bins, mask_shrink):
            self.updating_parameters(blur_size, min_gap, min_pore_size, bins, mask_shrink)
            self.display_images()

        interact(
            update_conditions,
            blur_size=widgets.IntSlider(min=1, max=10, value=1),
            min_gap=widgets.IntSlider(min=10, max=100, value=20),
            min_pore_size=widgets.FloatSlider(min=10, max=100, value=16),
            bins=widgets.IntSlider(min=10, max=100, value=20),
            mask_shrink=widgets.IntSlider(min=0, max=10, value=2, description='Mask Shrink')
            )
        
        #self.updating_parameters(blur_size, min_gap, min_pore_size, bins, mask_shrink)

    def get_pore_sizes(self):
        return self.pore_sizes

    def get_pore_intensities(self):
        return self.pore_intensities

    def get_mask(self):
        return self.mask


