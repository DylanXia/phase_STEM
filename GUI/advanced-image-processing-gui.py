"""
PyQt5 is copyright (c) Riverbank Computing Limited.
"""

import sys
import os
from PIL import Image, ImageDraw, ImageFont
from hyperspy.api import load as emdloader
#from rosettasciio import rsciio
import h5py
import numpy as np
import mrcfile
import scipy.ndimage as ndimage
import scipy.fft as fft   # import dctn, idctn

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLineEdit,
                             QPushButton, QWidget, QFileDialog, QListWidget, QMessageBox,
                             QLabel, QSlider, QDialog, QFormLayout, QComboBox, QGridLayout,
                             QDoubleSpinBox, QSpinBox, QMenu, QGraphicsTextItem, QSplitter)
from PyQt5.QtCore import Qt, QPointF, QSize
from PyQt5.QtGui import QIcon
import pyqtgraph as pg
from PyQt5.QtGui import QFont


class ProgressWindow(QWidget):
    def __init__(self, max_value):
        super().__init__()
        self.setWindowTitle('Running...')
        self.setGeometry(100, 100, 500, 100)

        layout = QVBoxLayout(self)
        
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)

        self.plot_widget.setBackground('w')
        self.plot_widget.setXRange(0, 100)
        self.plot_widget.setYRange(0, 1)
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.hideAxis('left')

        self.progress_bar = pg.BarGraphItem(x0=0, y0=0.5, x1=0, y1=1, brush='b')
        self.plot_widget.addItem(self.progress_bar)

        self.percentage_text = pg.TextItem(text='0%', color=(0, 0, 0))
        self.percentage_text.setPos(50, 0.5)
        self.plot_widget.addItem(self.percentage_text)

    def update_progress(self, percentage):
        self.progress_bar.setOpts(x1=percentage)
        self.percentage_text.setText(f'{int(percentage)}%')
        self.percentage_text.setPos(max(percentage, 1), 0.5)  # Ensure text is always visible
        QApplication.processEvents()

class COMProcessor:
    
    def __init__(self, datacube, segment, center):
        
        if datacube.ndim != 4:
            self.show_error_popup("Input image must be a 4-dimensional array")
            
        self.nx, self.ny, self.px, self.py = datacube.shape
        self.center = center
        if segment == "True":
            self.seg = True
        else: 
            self.seg = False
            
        self.datacube = datacube
        self.average = np.average(self.datacube, axis = (0, 1))
            
    def show_error_popup(self, error_message):
        # Create a QMessageBox to display the error
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Hint/Information")
        msg_box.setText("Important result:")
        msg_box.setInformativeText(error_message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    
    def find_center(self, image, mask_size=None):
        x, y = image.shape

        center = ndimage.center_of_mass(image)
            
        rmax = min( center[0], center[1], abs(y - center[0]), abs(x - center[1]))
        # Set the mask size (radius) based on the input or default to half of the image width
        r = mask_size if mask_size is not None else rmax*0.8 
        
        mask = np.zeros((self.px, self.py))
        y, x = np.ogrid[:self.py, :self.px]
        distance_from_center = (x - center[1])**2 + (y - center[0])**2
        mask[(distance_from_center <= r**2)] = 1

        refined_center = ndimage.center_of_mass(image * mask)
            
        return refined_center[1], refined_center[0]

    def segmented_circular_masks(self, pixelsize: float, inner_radius: float, outer_radius: float, angle_ranges: np.ndarray) -> np.ndarray:        
            
        height, width = self.px, self.py
        if self.center is not None:
            center_x, center_y = self.center
        else:
            self.center = self.find_center(self.average, mask_size=None)
            center_x, center_y = self.center

        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        inner_radius /= pixelsize
        outer_radius /= pixelsize
        ring_mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)

        theta = np.arctan2(y - center_y, x - center_x) 
        theta = (theta + 2 * np.pi) % (2 * np.pi)
        angle_ranges_rad = np.deg2rad(angle_ranges)

        masks = np.zeros((len(angle_ranges_rad), height, width), dtype=np.uint8)
        for i, (start_angle, end_angle) in enumerate(angle_ranges_rad):
            start_angle = start_angle % (2 * np.pi)
            end_angle = end_angle % (2 * np.pi)
            angle_mask = (theta >= start_angle) & (theta <= end_angle) if start_angle < end_angle else (theta >= start_angle) | (theta <= end_angle)
            masks[i] = (ring_mask & angle_mask).astype(np.uint8)

        return masks
    
    def segmented_circular_masks(self, pixelsize, nbins_radial, nbins_azimuthal, inner_radius, outer_radius, rotation=np.pi/4):
        # Determine the center
        width, height = self.px, self.py
        if self.center is not None:
            center_x, center_y = self.center
        else:
            self.center = self.find_center(self.average, mask_size=None)
            center_x, center_y = self.center

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
            
        return masks
    
    def segment_intensities(self, semiconv, pixelsize, inner_radius, outer_radius, angle_ranges):
            
        self.mask = self.segmented_circular_masks(pixelsize, inner_radius, outer_radius, angle_ranges) # mask is np.array with a shape of (N, px, py)
        diffx =(self.center[0] - self.px//2)
        diffy =(self.center[1] - self.py//2)
        num = len(angle_ranges)
        qy,qx = np.meshgrid(np.arange(-self.px//2, self.px//2)- diffx, np.arange(-self.py//2, self.py//2)- diffy)
        self.segments = np.zeros((num, self.nx, self.ny))
        
        const = semiconv/(pixelsize*1000)
        if self.seg:
            self.COMs = np.zeros((2, self.nx, self.ny))
        for i in range(self.nx):
            for j in range(self.ny):
                DP = self.datacube[i, j, :, :]              
                for n in range(num):                    
                    self.segments[n, i, j] = np.sum(DP * self.mask[n])*const
                if self.seg:
                    self.COMs[0, i, j] += np.sum(DP * qx)
                    self.COMs[1, i, j] += np.sum(DP * qy)
        if self.seg:

            return self.segments, self.mask, self.COMs
        else: 
            return self.segments, self.mask

# Method for OBF reconstruction

class Aperture:
    def __init__(self, alpha_max, wavelength):
        self.alpha_max = alpha_max
        self.wavelength = wavelength

    def intensity(self, qx, qy):
        qx2 = qx**2
        qy2 = qy**2
        q = np.sqrt(qx2 + qy2)
        ktheta = np.arctan2(q, 1 / self.wavelength)
        return (ktheta < self.alpha_max)


class SpatialFrequency:
    @staticmethod
    def generate(pixel, resolution, dimension=1):
        if np.isscalar(pixel):
            px = py = int(pixel)
        else:
            px = int(pixel[0])
            py = int(pixel[1])

        qMax = 1 / resolution
        dqx = 1 / (resolution * px)
        dqy = 1 / (resolution * py)
        qxx = np.arange((-qMax + dqx) / 2, (qMax + dqx) / 2, dqx)
        qyy = np.arange((-qMax + dqy) / 2, (qMax + dqy) / 2, dqy)

        if dimension == 2:
            QX, QY = np.meshgrid(qxx, qyy)
            return QX, QY
        elif dimension == 1:
            return qxx, qyy
        else:
            raise ValueError("Error: use dimension=1 or dimension=2 for 1D or 2D")


class KSpace:
    @staticmethod
    def one_d(collection, segment, size, wavelength):
        r1 = (collection[0] / 2) / wavelength
        r2 = (collection[1] / 2) / wavelength

        x = np.linspace(-r2, r2, size + 1)
        y = np.linspace(-r2, r2, size + 1)

        X, Y = np.meshgrid(x, y)
        R = X**2 + Y**2
        mask_R = (R >= r1**2) & (R <= r2**2)

        mask_x, mask_y = X[mask_R], Y[mask_R]
        Theta = np.round(np.degrees(np.arctan2(mask_y, mask_x)))
        Theta2 = Theta.copy()
        Theta[Theta < 0] += 360

        if segment[0] < 0:
            mask_Theta2 = (Theta2 >= segment[0]) & (Theta2 <= segment[1])
            x_corr, y_corr = mask_x[mask_Theta2], mask_y[mask_Theta2]
        else:
            mask_Theta = (Theta >= segment[0]) & (Theta <= segment[1])
            x_corr, y_corr = mask_x[mask_Theta], mask_y[mask_Theta]

        return x_corr, y_corr


class BetaFunctionCPU:
    def __init__(self, parameters, aberration_coeffs):
        self.parameters = parameters
        self.aberration_coeffs = aberration_coeffs
        
    def chi_omega(self, kx, ky, wavelength):
        ab = self.aberration_coeffs
        if isinstance(ab, dict):
            aberrations = np.array([
                ab['C1'],     # C1
                ab['A1'],     # A1
                ab['B2'],     # B2
                ab['A2'],     # A2
                ab['C3']      # C3
            ], dtype=np.complex128)
        elif isinstance(ab, np.ndarray) and len(ab) >= 5:
            aberrations = ab
        else:
            raise ValueError("Error: please input correct aberrations!")

        phi = np.arctan2(ky, kx)
        alpha = np.arctan2(np.sqrt(kx ** 2 + ky ** 2), 1 / wavelength)

        chi_value = (aberrations[0].real * (alpha**2 / 2) +
                     aberrations[1].real * (alpha**2 / 2) * np.cos(2 * (phi - aberrations[1].imag)) +
                     aberrations[2].real * (alpha**3 / 3) * np.cos(1 * (phi - aberrations[2].imag)) +
                     aberrations[3].real * (alpha**3 / 3) * np.cos(3 * (phi - aberrations[3].imag)) +
                     aberrations[4].real * (alpha**4 / 4) 
                     )

        return (2 * np.pi * chi_value) / wavelength

    def compute(self, qx, qy, segment):
        slices = int(self.parameters["Slices:"])
        sizeX, sizeY = qx.shape
        tk = self.parameters["sample thickness(nm):"]
        size = 20  #  self.parameters["virtual grids in one segment detector"]
        df = self.aberration_coeffs['C1']
        wavelength = self.parameters["wavelength(nm):"]
        collection_angle = [self.parameters["Mini. collecting angle(rad):"], self.parameters["Max. collecting angle(rad):"]]
        semi_conv = self.parameters["Semi. conv. angle(rad):"]
        
        OptimumFilter = np.zeros((sizeX, sizeY), dtype=np.complex128)
        
        kx1, ky1 = KSpace.one_d(collection_angle, segment, size, wavelength)
        chi_1 = self.chi_omega(kx1, ky1, wavelength)
        S1 = Aperture(semi_conv, wavelength).intensity(kx1, ky1)

        window = ProgressWindow(slices*sizeX*sizeY)
        window.show()
        for n in range(slices):
            dz = df + n * tk / (slices - 1) if slices != 1 else df
            chi_K1 = np.pi * wavelength * dz * (kx1 ** 2 + ky1 ** 2) + chi_1 if slices != 1 or df != 0 else chi_1

            beta_q1 = S1 * np.exp(-1j * chi_K1)
            normalizer = np.sum(np.square(np.abs(beta_q1)))

            for i in range(sizeX):
                for j in range(sizeY):
                    kx2 = kx1 - qx[i, j]
                    ky2 = ky1 - qy[i, j]
                    S2 = Aperture(semi_conv, wavelength).intensity(kx2, ky2)
                    chi_K2 = self.chi_omega(kx2, ky2, wavelength) + np.pi * wavelength * dz * (kx2 ** 2 + ky2 ** 2)
                    beta_q2 = S2 * np.exp(-1j * chi_K2)

                    kx3 = kx1 + qx[i, j]
                    ky3 = ky1 + qy[i, j]
                    S3 = Aperture(semi_conv, wavelength).intensity(kx3, ky3)
                    chi_K3 = self.chi_omega(kx3, ky3, wavelength) + np.pi * wavelength * dz * (kx3 ** 2 + ky3 ** 2)
                    beta_q3 = S3 * np.exp(-1j * chi_K3)

                    temp_array = np.conjugate(beta_q1) * beta_q2 - beta_q1 * np.conjugate(beta_q3)
                    OptimumFilter[i, j] += np.sum(temp_array) / slices
                    window.update_progress((n*sizeX*sizeY+i*sizeY+j+1)/(slices*sizeX*sizeY)*100)
                    QApplication.processEvents()

        return OptimumFilter, normalizer


class PhaseFiltersCPU:
    def __init__(self, ab, segments, parameters):
        self.ab = ab
        self.segments = segments
        self.parameters = parameters
        self.num = len(self.segments)
    def build_filters(self):
        num_pointx = num_pointy = int(self.parameters["pixelnumber of filters"])
        OptimumFilter = [np.zeros((num_pointx, num_pointy), dtype=np.complex128) for _ in range(len(self.segments))]

        qx, qy = SpatialFrequency.generate((num_pointx, num_pointy), self.parameters["Pixel size(nm):"], dimension=2)
        norm = 0
        window = ProgressWindow(self.num)
        window.show()
        for i in range(self.num):            
            beta_function = BetaFunctionCPU(self.parameters, self.ab)
            OptimumFilter[i], px = beta_function.compute(qx, qy, self.segments[i])

            norm += px
            window.update_progress((i+1)/(self.num)*100)
            QApplication.processEvents()
        WPO = []
        for f in OptimumFilter:
            WPO.append(f/norm)
        return WPO
    
class OBFBuilder:
    def __init__(self, DPC_imgs, PCTFs, parameters):
        
        self.DPC_imgs = DPC_imgs
        self.PCTFs = PCTFs
        self.parameters = parameters

        self.num = len(DPC_imgs)
        self.sizeX_img, self.sizeY_img = DPC_imgs[0].shape
        self.sizeX, self.sizeY = PCTFs[0].shape
        self.ratioX = self.sizeX_img / self.sizeX
        self.ratioY = self.sizeY_img / self.sizeY

        if self.parameters["Extreme resolution(rad):"] is not None:
            freq_limit = self.parameters["Extreme resolution(rad):"] # rad
        else:
            freq_limit = 0.015 # 15 mrad is set as a default
        qx, qy = SpatialFrequency.generate((self.sizeX_img, self.sizeY_img), self.parameters["Pixel size(nm):"], dimension=2)
            
        qMax = (2 * freq_limit)/self.parameters["wavelength(nm):"] # nm-1, based on the unit of wavelength
        r_squared = (qx[:None]**2 + qy[None:]**2)
        self.mask = r_squared < qMax**2

    def normalize_PCTFs(self):
        """ Normalize the phase filters based on the size ratios. """
        phase_filters = []
        for p in self.PCTFs:
            if self.ratioX != 1 or self.ratioY != 1:
                temp_filter = ndimage.zoom(1j * p, (self.ratioX, self.ratioY))
            else:
                temp_filter = 1j * p
            phase_filters.append(temp_filter)
        return phase_filters

    def compute_dQ(self):
        """ Compute the d_Q array for background correction. """
        d_Q = np.zeros(self.num)
        for i in range(self.num):
            bkg = min(abs(np.min(self.DPC_imgs[i])), abs(np.max(self.DPC_imgs[i])))
            d_Q[i] = bkg if bkg != 0 else 1/self.num
        return d_Q

    def compute_weighting(self, phase_filters, d_Q):
        """ Compute the weighting array using the phase filters and d_Q. """
        KQ_squared = np.zeros((self.sizeX_img, self.sizeY_img), dtype=np.float64)
        for i in range(self.num):
            KQ_squared += np.real(np.square(np.abs(phase_filters[i]))) / d_Q[i]
        
        weighting = np.sqrt(KQ_squared)
        weighting[weighting == 0] = np.inf  # Avoid division by zero
        return np.reciprocal(weighting)

    def compute_OBF_Q(self, phase_filters, d_Q, weightingInv):
        """ Compute the OBF in the Fourier domain. """
        OBF_Q = np.zeros((self.num, self.sizeX_img, self.sizeY_img), dtype = np.complex128)
        for n in range(self.num):
            wq = np.conj(phase_filters[n]) * weightingInv / d_Q[n]
            dft_DPCs = fft.fft2(self.DPC_imgs[n])
            dft_DPCs = fft.fftshift(dft_DPCs)
            OBF_Q[n] = dft_DPCs * wq * self.mask
        return OBF_Q

    def reconstruct_OBF(self):
        """ Main function to reconstruct the OBF image and Fourier domain representation. """
        phase_filters = self.normalize_PCTFs()
        d_Q = self.compute_dQ()
        weightingInv = self.compute_weighting(phase_filters, d_Q)
        OBF_Q = self.compute_OBF_Q(phase_filters, d_Q, weightingInv)
        OBF_image = fft.ifft2(fft.ifftshift(np.sum(OBF_Q, axis=0)))
        return OBF_image.real

# Method for iDPC-STEM image
class iDPC_DCTBuilder:
    
    def __init__(self, CoMx: np.ndarray, CoMy: np.ndarray, pixel_size_R: float = 0.001, epsilon: float = 0.01):                

        if CoMx.shape != CoMy.shape:
            self.show_error_popup("CoMx and CoMy must have the same shape")
            #raise ValueError("CoMx and CoMy must have the same shape")

    
        self.pad_x, self.pad_y = CoMx.shape
    
        self.CoMx = np.pad(CoMx, ((self.pad_x, self.pad_x), (self.pad_y, self.pad_y)), mode='reflect')
        self.CoMy = np.pad(CoMy, ((self.pad_x, self.pad_x), (self.pad_y, self.pad_y)), mode='reflect')
        
        self.nd = CoMx.ndim
        self.epsilon = epsilon
        self.sampling = pixel_size_R

        self.R_Nx, self.R_Ny = self.CoMx.shape
        
        self._initialize_variables()

    def show_error_popup(self, error_message):
        # Create a QMessageBox to display the error
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Hint/Information")
        msg_box.setText("Important result:")
        msg_box.setInformativeText(error_message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()

    def _initialize_variables(self):

        # Create meshgrid for denominator calculation
        x = np.arange(self.R_Nx)
        y = np.arange(self.R_Ny)
        xm, yn = np.meshgrid(x, y, indexing ='ij')
        # Calculate denominator
        self.denominator = (2 * np.sin(xm * np.pi / (2 * self.R_Nx)))**2 + (2 * np.sin(yn * np.pi / (2 * self.R_Ny)))**2
        self.denominator += self.epsilon
        self.denominator[self.denominator == 0] = np.inf
        self.denominator = 1./self.denominator

    @staticmethod
    def _create_Laplacian(obj_x, obj_y, dxy):

        px, py = obj_x.shape
        Lx = np.zeros((px, py))
        Ly = np.zeros((px, py))
        # The boundary of Laplacian matrix is given by Neumann boundary condition
        Ly[:, 0] = (obj_y[:, 0] + obj_y[:, 1]) / (2 * dxy)
        Ly[:, -1] = -(obj_y[:, -2] + obj_y[:, -1]) / (2 * dxy)
        Lx[0, :] = (obj_x[0, :] + obj_x[1, :]) / (2 * dxy)
        Lx[-1, :] = -(obj_x[-2, :] + obj_x[-1, :]) / (2 * dxy)
        Ly[1:-1, 1:-1] = (obj_y[1:-1, 2:] - obj_y[1:-1, :-2]) / (2 * dxy)
        Lx[1:-1, 1:-1] = (obj_x[2:, 1:-1] - obj_x[:-2, 1:-1]) / (2 * dxy)    

        return Lx + Ly

    def run(self, theta: float, flip: bool, dctn_type: int = 2):

        theta = np.radians(theta)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)

        if not flip:
            self.CoMx_rot = self.CoMx * cos_theta - self.CoMy * sin_theta
            self.CoMy_rot = self.CoMx * sin_theta + self.CoMy * cos_theta
        else:
            self.CoMx_rot = self.CoMx * cos_theta + self.CoMy * sin_theta
            self.CoMy_rot = self.CoMx * sin_theta - self.CoMy * cos_theta

        phase = np.zeros((self.R_Nx, self.R_Ny))

        # Calculate Laplacian matrix
        Laplacian = self._create_Laplacian(self.CoMx_rot, self.CoMy_rot, self.sampling)

        # Compute phase
        dct_lap = fft.dctn(Laplacian, type=dctn_type, norm='ortho')
        #dct_lap = dct(dct(Laplacian, type=dctn_type, axis=0, norm='ortho'), type=dctn_type, axis=1, norm='ortho') 

        phase = fft.idctn(dct_lap * self.denominator, type=dctn_type, norm='ortho')
            #self.phase = idct(idct(dct_lap*self.denominator, type=2, axis=1, norm='ortho'), type=2, axis=0, norm='ortho')
        phase *=self.sampling**2
        return phase[self.pad_x: -self.pad_x, self.pad_y: -self.pad_y]

    @classmethod
    def optimize_rotation(cls, CoMx: np.ndarray, CoMy: np.ndarray, pixel_size_R: float = 0.001, epsilon: float = 1e-2):
        reconstructor = cls(CoMx, CoMy, pixel_size_R, epsilon)
        thetas = np.linspace(-90, 91, 181)
        N_thetas = len(thetas)

        stds = np.zeros((2, N_thetas))
        window = ProgressWindow(N_thetas*2)
        window.show()
        for flip in range(2):
            for i, theta in enumerate(thetas):           
                phase= reconstructor.run(theta, bool(flip), dctn_type = 2)
                stds[flip, i] = np.std(phase)
                window.update_progress(((flip * N_thetas) + i + 1) / (2 * N_thetas) * 100)
                QApplication.processEvents()  # Keep the UI responsive during the loop
                
        flip = np.max(stds[1]) > np.max(stds[0])
        theta = thetas[np.argmax(stds[1 if flip else 0])]
        
        return theta, flip
    
# method for first-moment STEM image
class FMSTEMBuilder:
    """
    It creates a First-moment phase-contrast STEM image based on DPC's datasets or 4DSTEM's datasets.
    """
    def __init__(self, CoMx: np.ndarray, CoMy: np.ndarray, epsilon: float = 1e-4):
        
        if CoMx.shape != CoMy.shape:
            self.show_error_popup("CoMx and CoMy must have the same shape")
        #normalizing the intensity within [0, 1]
        self.CoMx = CoMx 
        self.CoMy = CoMy 
        self.regHighpass = epsilon
        
        self.R_Nx, self.R_Ny = CoMx.shape
        self.qx = fft.fftfreq(self.R_Nx)
        self.qy = fft.fftfreq(self.R_Ny)
        self._initialize_variables()
        
    def show_error_popup(self, error_message):
        # Create a QMessageBox to display the error
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Hint/Information")
        msg_box.setText("Important result:")
        msg_box.setInformativeText(error_message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
        
    def _initialize_variables(self):
        """Initialize variables needed for the reconstruction."""
        qr1 = 1j * self.qx[:, None] + self.qy[None, :]
        self.r = self.qx[:, None]**2 + self.qy[None, :]**2
        self.qr = self.r + self.regHighpass

        self.denominator = np.zeros(self.qr.shape, dtype=np.complex128)
        none_zero = self.qr != 0
        self.denominator[none_zero] = 1 / self.qr[none_zero]
        self.denominator *= qr1
        
        
    def run(self, theta: float, flip: bool):        

        theta = np.radians(theta)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        
        if not flip:
            self.CoMx_rot = self.CoMx * cos_theta - self.CoMy * sin_theta
            self.CoMy_rot = self.CoMx * sin_theta + self.CoMy * cos_theta
        else:
            self.CoMx_rot = self.CoMx * cos_theta + self.CoMy * sin_theta
            self.CoMy_rot = self.CoMx * sin_theta - self.CoMy * cos_theta  

        phase = np.zeros((self.R_Nx, self.R_Ny))

        self.dpc = - (fft.fft2(self.CoMx_rot)  + fft.fft2(self.CoMy_rot) * 1j)* self.denominator                   
            
        phase = fft.ifft2(self.dpc)

        return np.real(phase)

    @classmethod
    def optimize_rotation(cls, CoMx: np.ndarray, CoMy: np.ndarray, epsilon: float = 1e-3):                           
        
        reconstructor = cls(CoMx, CoMy, epsilon)
        thetas = np.arange(-90, 91, 1)
        edge = 16
        N_thetas = len(thetas)
        stds = np.zeros((2, N_thetas))
        window = ProgressWindow(N_thetas*2)
        window.show()

        for flip in range(2):
            for i, theta in enumerate(thetas):           
                phase= reconstructor.run(theta, bool(flip))
                stds[flip, i] = np.std(phase[edge:-edge, edge:-edge])
                window.update_progress(((flip * N_thetas) + i + 1) / (2 * N_thetas) * 100)
                QApplication.processEvents()  # Keep the UI responsive during the loop
        flip = np.max(stds[1]) > np.max(stds[0])
        theta = thetas[np.argmax(stds[1 if flip else 0])]
                
        return theta, flip   

        
class ClickableImageView(pg.ImageView):
    def __init__(self, parent=None, main_window=None, name="ImageView", view=None, imageItem=None, *args, **kargs):
        super().__init__(parent, name, view, imageItem, *args, **kargs)
        self.scene.sigMouseClicked.connect(self.mouseClickEvent)
        self.main_window = main_window  # Reference to the main window
        self.image_name_item = QGraphicsTextItem()  # Text item for image name
        self.ui.graphicsView.scene().addItem(self.image_name_item)
        self.image_name_item.setDefaultTextColor(Qt.white)  # Set text color
        self.scale_bar = None
        self.scale_bar_label = None
        
        
    def mouseClickEvent(self, event):
        if event.button() == Qt.LeftButton and self.main_window:
            self.main_window.set_active_view(self)
            for view in self.main_window.views:
                if view != self:
                    view.setBorder('r')
        elif event.button() == Qt.RightButton and self.main_window:           
            view = self.main_window.get_active_view()
            if view.getImageItem().image is not None:
                self.show_context_menu(event)
                
    def show_context_menu(self, event):
        menu = QMenu()
        info_action = menu.addAction("Show Image Info")
        toggle_scale_bar_action = menu.addAction("Toggle Scale Bar")
        invert_intensity_action = menu.addAction("Invert intensity")
        clear_image  = menu.addAction("Clear window")
        
        action = menu.exec_(event.screenPos().toPoint())
        
        if action == info_action:
            if self.main_window:
                self.main_window.show_image_info(self)
        elif action == toggle_scale_bar_action:
            self.toggle_scale_bar()
        elif action == invert_intensity_action:
            self.invert_intensity()
        elif action == clear_image:
            view = self.main_window.get_active_view()
            view.setImage(np.array([[]]))
            
    def setBorder(self, color):
        self.ui.graphicsView.setStyleSheet(f"border: 5px solid {color};")

    def setImageWithName(self, image, name):
        self.getImageItem().setImage(image)
        self.image_name_item.setPlainText(name)
        self.image_name_item.setPos(10, 10)  # Position at top-left corner
        
    def toggle_scale_bar(self):
        if self.scale_bar:
            self.remove_scale_bar()
        else:
            self.show_scale_bar_dialog()
            
    def invert_intensity(self):
        view = self.main_window.get_active_view()
        image_data = view.getImageItem().image
        image_data *= -1
        view.getImageItem().setImage(image_data)
        
    def show_scale_bar_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Scale Bar Settings")

        layout = QFormLayout(dialog)

        scale_length_input = QLineEdit()
        scale_length_input.setObjectName("length")
        scale_length_input.setText("10")

        scale_height_input = QLineEdit()
        scale_height_input.setObjectName("height")
        scale_height_input.setText("20")

        unit_input = QLineEdit()
        unit_input.setObjectName("units")  # Set a unique name for identification
        unit_input.setText("nm")
        
        resolution_input = QLineEdit()
        resolution_input.setObjectName("size in pixel")  # Set a unique name for identification
        resolution_input.setText("0.037")
        
        unit_font = QLineEdit()
        unit_font.setObjectName("Fontsize")  
        unit_font.setText("16")
        
        layout.addRow("Length in real:", scale_length_input)
        layout.addRow("Height in pixel:", scale_height_input)
        layout.addRow("Size in pixel:", resolution_input)
        layout.addRow("Unit:", unit_input)
        layout.addRow("Unit fontsize:", unit_font)

        button = QPushButton("OK")
        button.clicked.connect(lambda: self.add_scale_bar(
            float(resolution_input.text()),
            float(scale_length_input.text()),
            float(scale_height_input.text()),
            unit_input.text(),
            float(unit_font.text())
            ))
        button.clicked.connect(dialog.accept)
        layout.addWidget(button)

        dialog.exec_()

    def add_scale_bar(self, resolution, length=100, scale_height=10, unit="units", font_size = 16):
        image_data = self.getImageItem().image
        if image_data is not None:
            shape = image_data.shape               
                
            scale_length = int(length/resolution)  # Adjust the scale length as needed
            scale_height = int(scale_height)   # Thickness of the scale bar
            position = (int(shape[1]*0.95) - scale_length, int(shape[0]*0.95) - scale_height)  # Bottom-right corner

            self.scale_bar = pg.ROI(position, [scale_length, scale_height], pen=pg.mkPen(color='w', width =2), movable=True)
            self.addItem(self.scale_bar)
            custom_font = QFont()
            custom_font.setPointSize(int(font_size))
            # Add label for the scale bar
            self.scale_bar_label = pg.TextItem(f"{str(int(length))} {unit}", color='w')
            self.scale_bar_label.setFont(custom_font)
            self.addItem(self.scale_bar_label)
            self.scale_bar_label.setPos(QPointF(int(position[0]*0.95), int(position[1])))  # Adjust the label position based on your preference
            
         #   self.scale_bar_label = QGraphicsTextItem(f"{str(int(length))} {unit}")
         #   self.scale_bar_label.setDefaultTextColor(Qt.white)
          #  self.scale_bar_label.setFont(custom_font)
           # label_width = self.scale_bar_label.boundingRect().width()
      #      label_x = position[0] + (scale_length - label_width) / 2
       #     label_y = position[1] - self.scale_bar_label.boundingRect().height() - 5  # 5 pixels above the scale bar
       #     self.scale_bar_label.setPos(QPointF(label_x, label_y))
            
            self.ui.graphicsView.scene().addItem(self.scale_bar_label)

    def remove_scale_bar(self):
        if self.scale_bar:
            self.removeItem(self.scale_bar)
            self.scale_bar = None
        if self.scale_bar_label:
            self.ui.graphicsView.scene().removeItem(self.scale_bar_label)
            self.scale_bar_label = None
       
class ImageSelectorDialog(QDialog):
    def __init__(self, image_list):
        super().__init__()

        self.setWindowTitle("Select datasets")
        self.selected_images = []  # Store the selected images from each column

        # Set up the main layout for the dialog
        main_layout = QVBoxLayout()

        # Add a horizontal layout to display column titles
        column_titles_layout = QHBoxLayout()
        column_titles =  ['Sub. 1', 'Sub. 2', 'Sub. 3', 'Sub. 4']
        for title in column_titles:
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignCenter)
            column_titles_layout.addWidget(title_label)
        main_layout.addLayout(column_titles_layout)

        # Create a grid layout for the combo boxes (4 columns x 4 rows)
        grid_layout = QGridLayout()
        self.combo_boxes = []  # Store all the combo boxes to retrieve selections later

        # Create combo boxes in a 4x4 grid layout with row titles

        row_titles = ['Section A', 'Section B', 'Section C', 'Section D'] 
        for row in range(4):
            # Add a label for each row title on the left side of the grid
            row_label = QLabel(row_titles[row])
            
            grid_layout.addWidget(row_label, row, 0)

            column_combo_boxes = []  # Temporary list to hold combo boxes for this column
            for col in range(1,5):
                combo_box = QComboBox()
                combo_box.addItems(image_list)  # Add items to each combo box
                grid_layout.addWidget(combo_box, row, col)
                column_combo_boxes.append(combo_box)
            self.combo_boxes.append(column_combo_boxes)  # Add the column's combo boxes to the main list

        main_layout.addLayout(grid_layout)

        # Add a button to confirm the selection
        select_button = QPushButton("Select")
        select_button.clicked.connect(self.select_images)
        main_layout.addWidget(select_button)

        # Set the dialog's main layout
        self.setLayout(main_layout)

    

    def select_images(self):
        """Handle the selection of images from each combo box."""
        self.selected_images = []
        for column_combo_boxes in self.combo_boxes:
            column_selected = [combo_box.currentText() for combo_box in column_combo_boxes]
            self.selected_images.append(column_selected)
        self.accept()  # Close the dialog and return

       
class STEMImageProcessingGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        script_dir = os.path.dirname(os.path.realpath(__file__))
        icon = QIcon(os.path.join(script_dir, 'logo.png'))
        self.setWindowIcon(icon)
        self.setIconSize(QSize(128,128))  # Set the icon size to 48x48 pixels
        self.setWindowTitle("STEM Phase-Contrast Image Processing GUI")
        self.setGeometry(128,128, 1500, 1000)
        
        # Build Lists for stuffs storage
        self.sequence = [] # Recording the relationship between the image_list and the corresponding information
        self.datasets = [] # Input's data, put a None for item selection
        self.info = [] # Information of Input's data
        self.reconstructions = [] # Processed results
        self.image_names = ["None"]  # Make the first item as "None", which is NOT a data
        self.phase_names = []
        self.count = 2
        self.cresult = 3
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Add a data list
        self.image_list = QListWidget()
        self.image_list.addItem("None")
        self.image_list.itemClicked.connect(self.display_image)
        self.image_list.currentRowChanged.connect(self.update_display_on_list)
        left_layout.addWidget(self.image_list)
        
        
        # File operations
        self.file_button = QPushButton("File")
        self.file_menu = QMenu(self)
        self.file_button.setMenu(self.file_menu)

        load_action = self.file_menu.addAction("Load")
        save_action = self.file_menu.addAction("Save")
        clear_action = self.file_menu.addAction("Clear")
        load_action.triggered.connect(self.load_file)
        save_action.triggered.connect(self.save_file)
        clear_action.triggered.connect(self.clear_data)
        left_layout.addWidget(self.file_button)

        # List of tools
        self.tools_list = QListWidget()
        self.tools_list.addItems(["Gaussian Blur", "Median Filter", "Rotation", "Bandpass filter", 
                                  "Wiener filter", "ABS filter", "Nonlinear filter" 
                                     ])
        self.tools_list.itemClicked.connect(self.show_tools)
        left_layout.addWidget(QLabel("List of tools"))
        left_layout.addWidget(self.tools_list)

        # Link to image reconstruction
        self.results_list = QListWidget()
        self.results_list.addItems(["iDPC-STEM by DCT", "First-Moment STEM", "Optimum Bright Field STEM", 
                                    "Create virtual image", "Find CoM(xy)", "Sum DPCs"])
        self.results_list.itemClicked.connect(self.show_results)
        left_layout.addWidget(QLabel("Methods of reconstruction"))
        left_layout.addWidget(self.results_list)
        
        # Presenting phase images
        self.phases_list = QListWidget()
        self.phases_list.itemClicked.connect(self.show_phases)
        self.phases_list.currentRowChanged.connect(self.update_result_on_list)
        left_layout.addWidget(QLabel("List of reconstructed images"))
        left_layout.addWidget(self.phases_list)
        
        # LUT
        self.lut_combo = QComboBox()
        self.lut_combo.addItems(["Viridis", "Grayscale", "Inferno", "Plasma", "rainbow", "hot"])
        self.lut_combo.currentIndexChanged.connect(self.update_image_display)
        left_layout.addWidget(QLabel("LUT"))
        left_layout.addWidget(self.lut_combo)
        
        # Bottom title label
        bottom_title = QLabel("Author: Yu Xia,  Contact me: yu.xia1989@outlook.com", self)
        left_layout.addWidget(bottom_title)
         # Add left panel to splitter
        main_splitter.addWidget(left_panel)

        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Top row of image views

        top_row = QHBoxLayout()
        self.view1 = ClickableImageView(self, main_window=self)
        self.view2 = ClickableImageView(self, main_window=self)
        self.view3 = ClickableImageView(self, main_window=self)
        top_row.addWidget(self.view1)
        top_row.addWidget(self.view2)
        top_row.addWidget(self.view3)
        right_layout.addLayout(top_row)       
        
        
        # Bottom row of image views
        bottom_row = QHBoxLayout()
        self.view4 = ClickableImageView(self, main_window=self)
        self.view5 = ClickableImageView(self, main_window=self)
        self.view6 = ClickableImageView(self, main_window=self)
        bottom_row.addWidget(self.view4)
        bottom_row.addWidget(self.view5)
        bottom_row.addWidget(self.view6)
        right_layout.addLayout(bottom_row)

        # Add right panel to splitter
        main_splitter.addWidget(right_panel)
        # Set the splitter as the central widget
        self.setCentralWidget(main_splitter)
        
        # Store all views in a list for easy access

        self.views = [self.view1, self.view2, self.view3, self.view4, self.view5, self.view6]        

        # Initialize active view
        self.active_view = None
        self.set_active_view(self.view1)  # Set view1 as initially active

        # Frames selection
        frames_layout = QHBoxLayout()
        self.frames_x = QSlider(Qt.Horizontal)
        self.frames_y = QSlider(Qt.Horizontal)
        frames_layout.addWidget(QLabel("Frames_x"))
        frames_layout.addWidget(self.frames_x)
        frames_layout.addWidget(QLabel("Frames_y"))
        frames_layout.addWidget(self.frames_y)
        right_layout.addLayout(frames_layout)
        self.frames_x.valueChanged.connect(self.frame_index_display)
        self.frames_y.valueChanged.connect(self.frame_index_display)
        
    
    def set_active_view(self, view):
        # Remove highlight from previous active view
        if self.active_view:
            self.active_view.setBorder('g')

        # Set new active view and highlight it
        self.active_view = view
        view.setBorder('r')
        self.update_fft_view(view)

    def update_fft_view(self, view):
        # Compute the FFT of the image in the active view and display it in view6
        
        if view != self.view3:
            image_data = view.getImageItem().image
            if image_data is not None:
                if np.sum(image_data)!=0:
                    fft_data = fft.fftshift(fft.fft2(image_data))
                    fft_magnitude = np.abs(fft_data)
                    self.view3.setImageWithName(np.log(fft_magnitude + 1), "FFT")
                else: self.view3.setImage(np.array([[]]))
                #self.view3.getImageItem().setImage(np.log(fft_magnitude + 1), autoLevels=True)
    
    def show_error_popup(self, error_message):
        # Create a QMessageBox to display the error
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Hint/Information")
        msg_box.setText("Important result:")
        msg_box.setInformativeText(error_message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    
    def get_active_view(self):
        return self.active_view
    
    def update_image_display(self, index):
        if self.active_view:
            # Choose the appropriate LUT based on the selected index
            luts = [pg.colormap.get('viridis', source='matplotlib'), 
                    pg.colormap.get('Greys', source='matplotlib'), 
                    pg.colormap.get('inferno', source='matplotlib'), 
                    pg.colormap.get('plasma', source='matplotlib'), 
                    pg.colormap.get('rainbow', source='matplotlib'), 
                    pg.colormap.get('hot', source='matplotlib')]
            lut = luts[index]

            # Update the image display with the selected LUT
            #self.active_view.setImage(image_data, levels=(image_data.min(), image_data.max()), lut=lut)
            self.active_view.getImageItem().setColorMap(lut)
    
    def show_image_info(self, view):
        if view.getImageItem().image is not None:
            current_item = self.image_list.currentItem()
            index = self.image_list.row(current_item)          
                
            info_dict = self.info[index-1]
            
            info_text = "\n".join(f"{key}: {value}" for key, value in info_dict.items())
            QMessageBox.information(self, "Image Information", info_text)
        else:
            self.show_error_popup("NO image is shown in this window")
    
    def load_file(self):
        figures = ['png', 'tif', 'tiff', 'jpg', 'bmp']
        hy = ['h5', 'hdf5']
        gms = ['dm3', 'emd', 'dm4']
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", 
                                                   "Image Files (*.png *.jpg *.tif *.bmp);;Array formats (*.dm3 *.dm4 *.h5 *.emd *.mrc);;All Files (*)")
        if file_path:
            file_extension = file_path.split('.')[-1].lower()
            old_num_files = len(self.image_list)
            if file_extension in figures:
                self.load_png(file_path)
                load_num_files = len(self.image_list) - old_num_files
                self.sequence.append(load_num_files)
            elif file_extension in hy:
                self.load_h5(file_path)
                load_num_files = len(self.image_list) - old_num_files
                self.sequence.append(load_num_files)
            elif file_extension in gms:
                self.load_emd(file_path)
                load_num_files = len(self.image_list) - old_num_files
                self.sequence.append(load_num_files)
            elif file_extension == "mrc":
                self.load_mrc(file_path)
                load_num_files = len(self.image_list) - old_num_files
                self.sequence.append(load_num_files)
            else:
                self.show_error_popup(f"Unsupported file format: {file_extension}")

    def information_data(self, s): 
        #extracting information using hyperspy
        if len(s)==0:
            self.show_error_popup("No image is found!")
        elif len(s)==1:
            img =s
        else: img = s[len(s)-1]
        
        Acquisition_instrument = {} #build a dictionary
        Acquisition_instrument["original_filename"] = str(img.metadata.General.original_filename)
        Acquisition_instrument["data"] = str(img.metadata.General.date)+"/"+str(img.metadata.General.time)
        beam_voltage = img.original_metadata.Optics.AccelerationVoltage
        beam_voltage = float(beam_voltage)/1000
        Acquisition_instrument["beam_voltage (kV)"] = beam_voltage
        Acquisition_instrument["camera_length(mm)"] = round(float(img.original_metadata.Optics.CameraLength)*1000, 2)
        resolution = float(img.axes_manager[1].scale)
        Acquisition_instrument["size_in_pixel"] = round(resolution, 5)
        Acquisition_instrument["unit"] = str(img.axes_manager[1].units)
        Acquisition_instrument["pixels"] = int(img.axes_manager[1].size)
        semi_conv = img.original_metadata.Optics.BeamConvergence
        semi_conv = round(float(semi_conv), 3)
        Acquisition_instrument["semi_convergence_angle (rad)"] = semi_conv
        detector1 = img.original_metadata.Detectors.Detector6.DetectorName
        HAADF_begin = img.original_metadata.Detectors.Detector6.CollectionAngleRange.begin
        HAADF_end = img.original_metadata.Detectors.Detector6.CollectionAngleRange.end
        HAADF_angle = np.array([round(float(HAADF_begin),3), round(float(HAADF_end),3)]) #in rad
        angle_begin = img.original_metadata.Detectors.Detector3.CollectionAngleRange.begin
        angle_end = img.original_metadata.Detectors.Detector3.CollectionAngleRange.end
        collection_angle = np.array([round(float(angle_begin),3), round(float(angle_end),3)]) #in rad
        detector2 = img.original_metadata.Detectors.Detector3.DetectorName
        Acquisition_instrument[detector2+"_collection_angle(rad)"] = collection_angle
        Acquisition_instrument[detector1+"_collection_angle(rad)"] = HAADF_angle
        defocus = img.original_metadata.Optics.Defocus
        Acquisition_instrument["defocus (nm)"] = round(float(defocus)*(10**(9)), 2)
        beam_current = img.original_metadata.Optics.LastMeasuredScreenCurrent
        Acquisition_instrument["Last measured beam_current (pA)"] = round(float(beam_current)*(10**12), 3)
        dwell_time = img.original_metadata.Scan.DwellTime
        dwell_time = float(dwell_time)
        q = 6.242 * (10 ** 18)
        dose_rate = (float(beam_current)*dwell_time*q)/(100*resolution**2)
        Acquisition_instrument["dwell_time(us)"] = round(float(dwell_time)*(10**6), 3)
        mag = float(img.metadata.Acquisition_instrument.TEM.magnification)
        if mag > 10**(6):
            Acquisition_instrument["magnification"] = str(mag/10**(6))+"_Mx"
        elif mag > 10**(3):
            Acquisition_instrument["magnification"] = str(mag/10**(3))+"_Kx"
        else: Acquisition_instrument["magnification"] = str(mag)+"_x"
        Acquisition_instrument["dose_rate(eÅ-2)"] = round(dose_rate, 2)
        Acquisition_instrument["tilt_alpha(deg)"] = round(float(img.metadata.Acquisition_instrument.TEM.Stage.tilt_alpha),3)
        Acquisition_instrument["tilt_beta(deg)"] = round(float(img.metadata.Acquisition_instrument.TEM.Stage.tilt_beta),3)
        Acquisition_instrument["stage_x (um)"] = round(float(img.original_metadata.Stage.Position.x)*10**(6), 3)
        Acquisition_instrument["stage_y(um)"] = round(float(img.original_metadata.Stage.Position.y)*10**(6), 3)
        Acquisition_instrument["stage_z(um)"] = round(float(img.original_metadata.Stage.Position.z)*10**(6), 3)
    
        return Acquisition_instrument    

    def load_png(self, file_path):
        name = os.path.split(file_path)
        name = os.path.splitext(name[1])[0]
        image = Image.open(file_path)
        image_array = np.array(image)
        array = image_array.T
        information = {"Image:": str(name)}
        self.info.append(information)
        if array.ndim > 2:
            array = np.sum(array, axis=0)
            self.datasets.append(array)
            self.image_list.addItem(f"{str(name)}")
            #self.image_names.addItem(f"{str(name)}")
            self.image_names.append(f"{str(name)}")
            self.info.append(information)
        else:
            self.datasets.append(array)
            self.image_list.addItem(f"{str(name)}")
            #self.image_names.addItem(f"{str(name)}")
            self.image_names.append(f"{str(name)}")
            self.info.append(information)

    def data_tree(self, file):
        def head_tree(node, indent=0):
            name = str()
            if isinstance(node, h5py.Group):
                name += str(node.name)
                for key in node.keys():
                    head_tree(node[key], indent+1)
            elif isinstance(node, h5py.Dataset):
                name += str(node.name)
                values = node[()]            
                if isinstance(values, np.ndarray):
                    size = values.shape
                    if size[0]>10:
                        self.datasets.append(values)
                        self.image_list.addItem(name)
                        #self.image_names.addItem(name)
                        self.image_names.append(name)
                        self.info.append({name: size})
                    else:                     
                        if len(values)>2:
                            for c, v in enumerate(values):                            
                                self.datasets.append(v)
                                self.image_list.addItem(name+f"{c+1}")   
                                #self.image_names.addItem(name+f"{c+1}")
                                self.image_names.append(name + f"{c+1}")
                                self.info.append({name+f"{c+1}": v.shape})
                        else:
                            self.datasets.append(values)
                            self.image_list.addItem(name)   
                            #self.image_names.addItem(name)
                            self.image_names.append(name)
                            self.info.append({name: values.shape})
        for key in file.keys():
            head_tree(file[key])

    def load_h5(self, file_path):
        f= h5py.File(file_path, 'r') 
        self.data_tree(f)
        
        
    def load_emd(self, file_path):
        # EMD file handling is more complex and depends on the specific structure
        # This is a placeholder implementation
        s = emdloader(file_path)
        try:
            info = self.information_data(s)
        except:
            info = {"ERROR:": "Failed to get information"}
        if len(s)>1:
            for i in range(len(s)):
                self.datasets.append(s[i].data)
                name = str(s[i].metadata.General.original_filename)
                name = os.path.splitext(name)[0]
                name += "_"
                name += str(s[i].metadata.General.title)
                self.image_list.addItem(f"{name}")
                #self.image_names.addItem(f"{name}")
                self.image_names.append(f"{name}")
                self.info.append(info)
                
        else:
                self.datasets.append(s.data)
                name = str(s.metadata.General.original_filename)
                name = os.path.splitext(name)[0]
                name += "_"
                name += str(s.metadata.General.title)
                self.image_list.addItem(f"{name}")
                #self.image_names.addItem(f"{name}")
                self.image_names.append(f"{name}")
                self.info.append(info)
 
    def load_mrc(self, file_path):
        with mrcfile.open(file_path) as mrc:
            image = mrc.data
            pixel_size = mrc.voxel_size
            labels = mrc.header.label

        name = os.path.split(file_path)
        name = os.path.splitext(name[1])[0]
        info = {"name:": str(name), "size:": image.shape,"pixel size:":[float(pixel_size.x), float(pixel_size.y), float(pixel_size.z)], "labels:": dict(enumerate(labels, 1))}

        self.datasets.append(image)
        self.image_list.addItem(f"{name}")
        self.image_names.append(f"{name}")
        self.info.append(info)
    
    def save_file(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Output datasets")
        layout = QFormLayout(dialog)
        combo_box = QComboBox()
        options = ["Active window", "Reconstructed dataset"]
        combo_box.addItems(options)
        layout.addRow("Choose for saving", combo_box)

        # Create OK and Cancel buttons
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        layout.addRow(ok_button, cancel_button)

        # Connect the button signals to their respective slots
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        ok_button.clicked.connect(dialog.accept)
        # Display the dialog and check the result
        if dialog.exec_() == QDialog.Accepted:
            choose = combo_box.currentText()
            if choose == "Active window":
                self.save_data()
            else:
                file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "MRC Files (*.mrc)")                                                           
                self.save_3Darray(file_path)
                
    def save_3Darray(self, file_path):
        
        def add_line_edit(label, placeholder):
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(placeholder)
            line_edit.setObjectName(label)  # Set a unique name for identification
            layout.addRow(label, line_edit)
            return line_edit
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Output datasets")
        layout = QFormLayout(dialog)
        combo_box = QComboBox()
        if self.phase_names is None:
            return
        else:
            combo_box.addItems(self.phase_names)
            layout.addRow("Choose for saving", combo_box)
            add_line_edit("Pixelsize:", "nm/px")
            add_line_edit("Unit:", "nm")
            ok_button = QPushButton("OK")
            cancel_button = QPushButton("Cancel")
            layout.addRow(ok_button, cancel_button)

            # Connect the button signals to their respective slots
            ok_button.clicked.connect(dialog.accept)
            cancel_button.clicked.connect(dialog.reject)
            ok_button.clicked.connect(dialog.accept)
            # Display the dialog and check the result
            if dialog.exec_() == QDialog.Accepted:
                index = combo_box.currentIndex()
                data = self.reconstructions[index]
                self.save_mrc(file_path, data, dialog)
                
    def save_data(self):
        figures = ['png', 'tif', 'bmp', 'jpg']
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", 
                                                   "PNG Files (*.png);;JPEG Files (*.jpg);;TIF Files (*.tif);;MRC Files (*.mrc);;All Files (*)")
        
        active_view = self.get_active_view()
        image = active_view.getImageItem().image  
        
        def add_line_edit(label, placeholder):
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(placeholder)
            line_edit.setObjectName(label)  # Set a unique name for identification
            layout.addRow(label, line_edit)
            return line_edit
        
        if file_path:
            file_extension = file_path.split('.')[-1].lower()
            if active_view and image is not None:
                dialog = QDialog(self)
                dialog.setWindowTitle("Parameter settings")
                layout = QFormLayout(dialog)
                add_line_edit("Pixelsize:", "nm/px")
                add_line_edit("Unit:", "nm")
                ok_button = QPushButton("OK")
                cancel_button = QPushButton("Cancel")
                layout.addRow(ok_button, cancel_button)
            else:
                self.show_error_popup("The selected item is NOT a image! or EMPTY")   
            if file_extension in figures:                
                ok_button.clicked.connect(lambda: self.save_png(file_path, image, dialog))
            elif file_extension == 'mrc':
                ok_button.clicked.connect(lambda: self.save_mrc(file_path, image, dialog))     
            else:
                self.show_error_popup(f"Unsupported file format for saving: {file_extension}")
                
        cancel_button.clicked.connect(dialog.reject)
        ok_button.clicked.connect(dialog.accept)
        dialog.exec_()
        

    def save_png(self, file_path, image, dialog):
        pixelsize = dialog.findChild(QLineEdit, "Pixelsize:").text()
        unit = dialog.findChild(QLineEdit, "Unit:").text()
        length = dialog.findChild(QLineEdit, "Scale bar length:").text()
        image = image.T
        im= Image.fromarray(np.interp(image, (image.min(), image.max()), (0, 65535)).astype(np.uint16))
        draw = ImageDraw.Draw(im)
        image_height = image.shape[0]
        
        # Calculate scale bar dimensions
        scale_length = int(float(length) / float(pixelsize)) if float(pixelsize) != 0 else int(float(length)) # Length of the scale bar in pixels
        scale_width = int(image_height * 0.01)   # Width of the scale bar
        font_size = int(image_height * 0.06)     # Font size for the scale bar label

        # Position for the scale bar
        left = image_height * 0.05
        top = image_height * 0.95

        font = ImageFont.truetype("arial.ttf", size=font_size)

        draw.line((left, top, left + scale_length, top), fill=255, width=scale_width)
        draw.text((left-0.5*scale_length, top - image_height * 0.085), text= f"{length} {unit}", font=font, align = "middle", color = "w")

        im.save(file_path, dpi=(600, 600))

    def save_mrc(self, file_path, image, dialog):
        pixelsize = dialog.findChild(QLineEdit, "Pixelsize:").text()
        unit = dialog.findChild(QLineEdit, "Unit:").text()            
        active_view = self.get_active_view()
        image = active_view.getImageItem().image
        if active_view and image is not None:
            with mrcfile.new(file_path, overwrite=True) as mrc:
                mrc.set_data(image.astype(np.float32))
                vsize = mrc.voxel_size.copy()
                vsize.x = pixelsize
                vsize.y = pixelsize
                mrc.voxel_size = vsize
                mrc.add_label(f"Unit of scale bar: {unit}")
                
    def clear_data(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Alarm: Clear datasets")
        layout = QFormLayout(dialog)
        combo_box = QComboBox()
        options = ["Loaded data", "Reconstructed data"]
        combo_box.addItems(options)
        layout.addRow("Which will be abandoned", combo_box)

        # Create OK and Cancel buttons
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        layout.addRow(ok_button, cancel_button)

        # Connect the button signals to their respective slots
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # Display the dialog and check the result
        if dialog.exec_() == QDialog.Accepted:
            choose = combo_box.currentText()
            if choose == "Loaded data":
                self.image_list.clear() #QListWidget
                self.image_names.clear()
                self.datasets.clear()
                self.info.clear()
                self.sequence.clear()
                self.view1.setImage(np.array([[]]))
                self.view2.setImage(np.array([[]]))
                self.image_names = ["None"]
                self.image_list.addItem("None")
                self.show_error_popup("The loaded datasets have been delected")
            else:
                self.phases_list.clear() # QListWidget
                self.phase_names.clear()
                self.reconstructions.clear()
                self.view4.setImage(np.array([[]]))
                self.view5.setImage(np.array([[]]))
                self.view6.setImage(np.array([[]]))
                self.show_error_popup("The reconstructed results have been delected")
        
    def frame_index_display(self):
        # Get the current item based on the new index
        active_view = self.get_active_view()
        
        if active_view in [self.view1, self.view2]:
            current_item = self.image_list.currentItem()
            index = self.image_list.row(current_item)
            image_name = self.image_names[index - 1]
            image_array = self.datasets[index - 1]

        elif active_view in [self.view4, self.view5, self.view6]:
            current_item = self.phases_list.currentItem()
            index = self.phases_list.row(current_item)
            image_name = self.phase_names[index]
            image_array = self.reconstructions[index]

        if image_array.ndim ==3:
            self.update_index_ranges(image_array.shape[0])
            frame_index = self.frames_x.value()
            image = image_array[frame_index]
            show_name = image_name + f"_frame_{frame_index + 1}"
            
        elif image_array.ndim ==4:
            self.update_index_ranges((image_array.shape[0], image_array.shape[1]))
            frame_x = self.frames_x.value()
            frame_y = self.frames_y.value()
            image = image_array[frame_x, frame_y]
            show_name = image_name + f"_frameX_{frame_x + 1}, frameY_{frame_y + 1}"
        else:
            image = image_array
            self.show_error_popup("There is ONLY ONE image!")
            
        if active_view and image_array.ndim >2:
            active_view.setImageWithName(image, show_name)
            self.update_fft_view(active_view)
            
    
    def update_display_on_list(self):
        # Get the currently selected image item from the list
        current_item = self.image_list.currentItem()
        if not current_item:
            self.display_image(current_item)
            
    def update_result_on_list(self):
        # Get the currently selected image item from the reconstructed result list
        current_item = self.phases_list.currentItem()
        if not current_item:
            self.show_phases(current_item)
            
    def update_index_ranges(self, shape):
        if isinstance(shape, tuple) and len(shape) > 0:
            self.frames_x.setRange(0, shape[0] - 1)  # Set range based on the first dimension
            if len(shape) > 1:
                self.frames_y.setRange(0, shape[1] - 1)  # Set range based on the second dimension if available
        else:
            self.frames_x.setRange(0, shape - 1)
            self.frames_y.setRange(0, shape - 1)       
        
    def display_image(self, item):
        index = self.image_list.row(item)
        #index = self.image_names.index(item)   
        image_name = self.image_names[index]# because this returned index is the index in the image_list, whose first item is "None"
        if image_name != "None":            
            image_array = self.datasets[index-1]
            remainder = self.count % 2      
            if image_array.ndim ==3:      
                frame_index = 0
                image = image_array[frame_index]
                show_name = image_name + f"_frame_{frame_index + 1}"           
            elif image_array.ndim ==4:           
                frame_x = 0
                frame_y = 0
                image = image_array[frame_x, frame_y]
                show_name = image_name + f"_frameX_{frame_x + 1}, frameY_{frame_y + 1}"            
            elif image_array.ndim == 2:
                image = image_array       
                show_name = image_name  
            else:
                self.show_error_popup("The selected data is NOT an image!")
            if remainder == 0 :
                self.view1.setImageWithName(image, show_name)
                self.view1.getImageItem().setColorMap(pg.colormap.get('viridis', source='matplotlib'))
            
            elif remainder == 1 :
                self.view2.setImageWithName(image, show_name)
                self.view2.getImageItem().setColorMap(pg.colormap.get('viridis', source='matplotlib'))

            self.count +=1
        else:
            self.show_error_popup("The selected item is NOT an image or EMPTY")
        
    def show_phases(self, item):
        index = self.phases_list.row(item)
        image_name = self.phase_names[index]
        image_array = self.reconstructions[index]
        remainder = self.cresult % 3
        
        if image_array.ndim ==3:      
            frame_index = 0
            image = image_array[frame_index]
            show_name = image_name + f"_frame_{frame_index + 1}"
            
        elif image_array.ndim == 2:
            image = image_array       
            show_name = image_name  
        else:
            self.show_error_popup("The selected data is NOT an image!")

        if remainder == 0 :
            self.view4.setImageWithName(image, show_name)
            self.view4.getImageItem().setColorMap(pg.colormap.get('viridis', source='matplotlib'))
            
        elif remainder == 1:
            self.view5.setImageWithName(image, show_name)
            self.view5.getImageItem().setColorMap(pg.colormap.get('viridis', source='matplotlib'))
        else:
            self.view6.setImageWithName(image, show_name)
            self.view6.getImageItem().setColorMap(pg.colormap.get('viridis', source='matplotlib'))
        self.cresult +=1

    def open_image_selector_dialog(self):
        selected_images = self.select_images_from_list(self.image_list)
        if selected_images:
            print("Selected Images:", selected_images)
        else:
            print("No images selected.")
            
    # Employing the Class ImageSelectorDialog to select datasets for phase reconstruction
    def select_images_from_list(self, image_list):
        dialog = ImageSelectorDialog(image_list)
        if dialog.exec_() == QDialog.Accepted:
            selected_images = [[], [], [], []] # the datasets is divided by four groups, stored in four sub-lists
            for idx, ele_x in enumerate(dialog.selected_images):
                for jdx, ele_y in enumerate(ele_x):
                    try:
                        if ele_y != "None":
                            index = self.image_names.index(ele_y)
                            selected_images[idx].append(self.datasets[index-1])
                    except ValueError:
                        index = 0

            return selected_images
        else:
            self.show_error_popup("Sorry! But failed to find datasets for phase reconstruction!")

    def select_4DSTEM(self, image_list):
        dialog = QDialog(self)
        dialog.setWindowTitle("Choose data")

        # Create a form layout for the dialog
        layout = QFormLayout(dialog)

        # Create the combo box and add items to it
        combo_box = QComboBox()
        combo_box.addItems(image_list)
        layout.addRow("Choose data", combo_box)

        # Create OK and Cancel buttons
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        layout.addRow(ok_button, cancel_button)

        # Connect the button signals to their respective slots
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # Display the dialog and check the result
        if dialog.exec_() == QDialog.Accepted:
            index = combo_box.currentIndex()
            data = self.datasets[index-1]
            return data
        else:
            self.show_error_popup("Sorry! But failed to find 4DSTEM datasets!")
            return None  # Return None if the dialog was canceled
            

    def select_result(self, image_list):       
        # Create the dialog for selecting data
        dialog = QDialog(self)
        dialog.setWindowTitle("Choose data")

        # Create a form layout for the dialog
        layout = QFormLayout(dialog)

        # Create the combo box and add items to it
        combo_box = QComboBox()
        combo_box.addItems(image_list)
        layout.addRow("Choose data", combo_box)

        # Create OK and Cancel buttons
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        layout.addRow(ok_button, cancel_button)

        # Connect the button signals to their respective slots
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # Display the dialog and check the result
        if dialog.exec_() == QDialog.Accepted:
            index = combo_box.currentIndex()
            if index <len(self.reconstructions):
                data = self.reconstructions[index]
                return data
            else: return None
        else:
            return None  # Return None if the dialog was canceled


    def show_results(self, results_list):
        # Here provides methods for phase-contrast STEM image reconstructions, which employs datasets of CoMx, CoMy.

        method = results_list.text()

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Parameter settings: {method}")
        pa_layout = QFormLayout(dialog)
        
        def add_line_edit(label, placeholder):
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(placeholder)
            line_edit.setObjectName(label)  # Set a unique name for identification
            pa_layout.addRow(label, line_edit)
            return line_edit
        
        def add_combo_box(label, items):
            combo_box = QComboBox()
            combo_box.addItems(items)
            combo_box.setObjectName(label)
            pa_layout.addRow(label, combo_box)
            return combo_box
        
        # "First-Moment STEM", "Optimum Bright Field STEM"
        if method == "iDPC-STEM by DCT":        
            data_bank_2r = self.select_result(self.phase_names) 
            if data_bank_2r is not None:
                CoMx = data_bank_2r[0] - data_bank_2r[2]
                CoMy = data_bank_2r[1] - data_bank_2r[3]
            else:
                data_bank = self.select_images_from_list(self.image_names)      
                CoMx = np.sum([item for item in data_bank[0] if item is not None], axis=0) - np.sum([item for item in data_bank[2] if item is not None], axis=0)            
                CoMy = np.sum([item for item in data_bank[1] if item is not None], axis=0) - np.sum([item for item in data_bank[3] if item is not None], axis=0)

            add_line_edit("epsilon:", "epsilon")
            add_line_edit("Pixel size(nm):", "size in real")
            add_line_edit("Angle (deg.):", "angle")
            add_combo_box("Flip:", ['True', 'False', 'None'])
            
        elif method == "First-Moment STEM":
            data_bank_2r = self.select_result(self.phase_names) 
            # for 4DSTEM dataset
            if data_bank_2r is not None and data_bank_2r.shape[0] ==2:
                CoMx = data_bank_2r[0] 
                CoMy = data_bank_2r[1] 
            elif data_bank_2r is not None and data_bank_2r.shape[0] !=2:
                self.show_error_popup("Choosing <CoMx,y> produced from 4DSTEM dataset for FM-STEM!")
            else:
                data_bank = self.select_images_from_list(self.image_names)      
                CoMx = np.sum([item for item in data_bank[0] if item is not None], axis=0) - np.sum([item for item in data_bank[2] if item is not None], axis=0)            
                CoMy = np.sum([item for item in data_bank[1] if item is not None], axis=0) - np.sum([item for item in data_bank[3] if item is not None], axis=0)
            add_line_edit("epsilon:", "epsilon")
            add_line_edit("Angle (deg.):", "angle")
            add_combo_box("Flip:", ['True', 'False', 'None'])
        
        elif method == "Optimum Bright Field STEM":
            DPC_imgs = []
            data_bank_2r = self.select_result(self.phase_names) 
            
            if data_bank_2r is not None and data_bank_2r.shape[0] ==4:
                print(data_bank_2r[0].shape)
                print(data_bank_2r[0])
                DPC_imgs.append(data_bank_2r[0] )
                DPC_imgs.append(data_bank_2r[1] )
                DPC_imgs.append(data_bank_2r[2] )
                DPC_imgs.append(data_bank_2r[3] )
            elif data_bank_2r is not None and data_bank_2r.shape[0] !=4:
                self.show_error_popup("Choosing <DPC images> produced from 4DSTEM dataset for OBF-STEM!")
            else:
                data_bank = self.select_images_from_list(self.image_names)   

                seg_1 = []
                seg_2 = []
                seg_3 = []
                seg_4 = []
                
                seg_1.append(item for item in data_bank[0] if item is not None)
                seg_2.append(item for item in data_bank[1] if item is not None)
                seg_3.append(item for item in data_bank[2] if item is not None)
                seg_4.append(item for item in data_bank[3] if item is not None)
                if len(seg_1) >1:
                    DPC_imgs.append(np.concatenate(seg_1, axis =0))
                    DPC_imgs.append(np.concatenate(seg_2, axis =0))
                    DPC_imgs.append(np.concatenate(seg_3, axis =0))
                    DPC_imgs.append(np.concatenate(seg_4, axis =0))
                else:
                    DPC_imgs.append(data_bank[0][0])
                    DPC_imgs.append(data_bank[1][0])
                    DPC_imgs.append(data_bank[2][0])
                    DPC_imgs.append(data_bank[3][0])
                
            add_line_edit("Pixel size(nm):", "size in real")
            add_line_edit("sample thickness(nm):", "evaluation")
            add_line_edit("Slices:", "num. of slices")
            add_line_edit("num of segments:", "num. in total")
            add_line_edit("rotation of segments(deg):", "clockwise")
            add_line_edit("Accelerating volt.(kV):", "300")
            add_line_edit("Mini. collecting angle(rad):", "minimum")
            add_line_edit("Max. collecting angle(rad):", "maximum")
            add_line_edit("Semi. conv. angle(rad):", "probe")
            add_line_edit("Extreme resolution(rad):", "30 mrad")
            add_line_edit("Binning:", "4")

        elif method == "Find CoM(xy)":
            data_bank_2r = self.select_result(self.phase_names) 
            if data_bank_2r is not None:
                CoMx = data_bank_2r[0] - data_bank_2r[2]
                CoMy = data_bank_2r[1] - data_bank_2r[3]
            else:
                data_bank = self.select_images_from_list(self.image_names)     
                segment_A = np.sum([item for item in data_bank[0] if item is not None], axis=0)
                segment_B = np.sum([item for item in data_bank[1] if item is not None], axis=0)
                segment_C = np.sum([item for item in data_bank[2] if item is not None], axis=0)     
                segment_D = np.sum([item for item in data_bank[3] if item is not None], axis=0)
                
                CoMx = segment_A + segment_B - segment_C - segment_D        
                CoMy = segment_A + segment_D - segment_C - segment_B  
            #add_line_edit("Processing DPCs:", "No parameter needed")
            
        elif method == "Sum DPCs":
            data_bank_2r = self.select_result(self.phase_names) 
            if data_bank_2r is not None:
                CoMx = data_bank_2r[0] - data_bank_2r[2]
                CoMy = data_bank_2r[1] - data_bank_2r[3]
            else:
                data_bank = self.select_images_from_list(self.image_names)      
                segment_A = np.sum([item for item in data_bank[0] if item is not None], axis=0)
                segment_B = np.sum([item for item in data_bank[1] if item is not None], axis=0)
                segment_C = np.sum([item for item in data_bank[2] if item is not None], axis=0)     
                segment_D = np.sum([item for item in data_bank[3] if item is not None], axis=0)
                
                CoMx = segment_A + segment_B - segment_C - segment_D        
                CoMy = segment_A + segment_D - segment_C - segment_B  
            #add_line_edit("Summing DPCs:", "No parameter needed")
            
        elif method == "Create virtual image":   
            CoMx = self.select_4DSTEM(self.image_names)  
            CoMy = None
            if CoMx.ndim != 4 or CoMx is None:
                self.show_error_popup("Sorry! But failed to find a 4DSTEM dataset!")
                return
            add_line_edit("Semi-convergence angle (mrad):", "mrad")
            add_line_edit("Pixelsize in detector (mrad/px):", "mrad/px")
            add_line_edit("Inner radius (mrad):", "radius")
            add_line_edit("Outer radius (mrad):", "radius")
            add_combo_box("Segment ?:", ['True', 'False'])
            add_line_edit("Rotating (deg.):", "clockwise")
        # OK and Cancel buttons
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        pa_layout.addRow(ok_button, cancel_button)
        ok_button.clicked.connect(dialog.accept)  
        cancel_button.clicked.connect(dialog.reject)
        # Connect button signals to dialog slots
        if method == "Optimum Bright Field STEM":            
            ok_button.clicked.connect(lambda: self.OBF_methods(DPC_imgs, dialog))
            
        else:            
            ok_button.clicked.connect(lambda: self.apply_methods(method, CoMx, CoMy, dialog))                  

        dialog.exec_()
    
   
    def OBF_methods(self, DPC_imgs, dialog):       

        pixelsize = dialog.findChild(QLineEdit, "Pixel size(nm):").text()
        parameters = {"Pixel size(nm):": float(pixelsize)}
        parameters["sample thickness(nm):"] = float(dialog.findChild(QLineEdit, "sample thickness(nm):").text())
        parameters["Slices:"] = float(dialog.findChild(QLineEdit, "Slices:").text())
        parameters["num of segments:"] = float(dialog.findChild(QLineEdit, "num of segments:").text())
        parameters["rotation of segments(deg):"] = float(dialog.findChild(QLineEdit, "rotation of segments(deg):").text())
        volt = float(dialog.findChild(QLineEdit, "Accelerating volt.(kV):").text())
        parameters["Mini. collecting angle(rad):"] = float(dialog.findChild(QLineEdit, "Mini. collecting angle(rad):").text())
        parameters["Max. collecting angle(rad):"] = float(dialog.findChild(QLineEdit, "Max. collecting angle(rad):").text())
        parameters["Semi. conv. angle(rad):"] = float(dialog.findChild(QLineEdit, "Semi. conv. angle(rad):").text())
        parameters["Extreme resolution(rad):"] = float(dialog.findChild(QLineEdit, "Extreme resolution(rad):").text())
        binning = float(dialog.findChild(QLineEdit, "Binning:").text()) 
        parameters["wavelength(nm):"] = self.wavelength_beam(volt)
        
        angle = parameters["rotation of segments(deg):"]
        ranges = self.virtual_detectors("True" , float(angle))

        ab_dialog = QDialog(self)
        ab_dialog.setWindowTitle("Aberrations are required")
        ab_layout = QFormLayout(ab_dialog)
        def line_edit(label, placeholder):
            text = QLineEdit()
            text.setPlaceholderText(placeholder)
            text.setObjectName(label)  # Set a unique name for identification
            ab_layout.addRow(label, text)
            return text

        line_edit("Aberrations(C1/nm):", "defocus")
        line_edit("Aberrations(A1/nm):", "amplitude")
        line_edit("Aberrations(A1/deg):", "angle")
        line_edit("Aberrations(A2/nm):", "amplitude")
        line_edit("Aberrations(A2/deg):", "angle")
        line_edit("Aberrations(B2/nm):", "amplitude")
        line_edit("Aberrations(B2/deg):", "angle")
        line_edit("Aberrations(C3/nm):", "sphere")
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ab_layout.addRow(ok_button, cancel_button)
        ok_button.clicked.connect(ab_dialog.accept)
        if ab_dialog.exec_() == QDialog.Accepted:
            aberrations = {
                "C1": float(ab_dialog.findChild(QLineEdit, "Aberrations(C1/nm):").text()),
                "A1": float(ab_dialog.findChild(QLineEdit, "Aberrations(A1/nm):").text()) + 1j * float(ab_dialog.findChild(QLineEdit, "Aberrations(A1/deg):").text()),
                "A2": float(ab_dialog.findChild(QLineEdit, "Aberrations(A2/nm):").text()) + 1j * float(ab_dialog.findChild(QLineEdit, "Aberrations(A2/deg):").text()),
                "B2": float(ab_dialog.findChild(QLineEdit, "Aberrations(B2/nm):").text()) + 1j * float(ab_dialog.findChild(QLineEdit, "Aberrations(B2/deg):").text()),
                "C3": float(ab_dialog.findChild(QLineEdit, "Aberrations(C3/nm):").text())
                }
       
        cancel_button.clicked.connect(ab_dialog.reject)
        ab_dialog.exec_()

        nd = DPC_imgs[0].ndim

        phase = np.zeros_like(DPC_imgs[0])
        if nd == 2:
            parameters["pixelnumber of filters"] = int(DPC_imgs[0].shape[0]/binning)
            wpo_builder = PhaseFiltersCPU(aberrations, ranges, parameters)
            wpo = wpo_builder.build_filters()
            OBF_creater = OBFBuilder(DPC_imgs, wpo, parameters)
            phase = OBF_creater.reconstruct_OBF()
            self.reconstructions.append( phase )
            self.phase_names.append("Optimum Bright Field STEM")
            self.phases_list.addItem("Optimum Bright Field STEM")
        elif nd == 3:
            num_iter = DPC_imgs[0].shape[0]
            parameters["pixelnumber of filters"] = int(DPC_imgs[0].shape[1]/binning)
            wpo_builder = PhaseFiltersCPU(aberrations, ranges, parameters)
            wpo = wpo_builder.build_filters()
            for n in range(num_iter):
                imgs = [DPC_imgs[0][n], DPC_imgs[1][n], DPC_imgs[2][n], DPC_imgs[3][n]]
                OBF_creater = OBFBuilder(imgs, wpo, parameters)
                phase[n] = OBF_creater.reconstruct_OBF()
            self.reconstructions.append( phase )
            self.phase_names.append(f"Optimum Bright Field STEM {num_iter+1}")
            self.phases_list.addItem("Optimum Bright Field STEM")
        else:
            self.show_error_popup("Sorry! But failed to construct OBF image!")
        
        
    def apply_methods(self, method, CoMx, CoMy, dialog):
        
        if method == "iDPC-STEM by DCT":
            epsilon = dialog.findChild(QLineEdit, "epsilon:").text()
            pixel_size_R = dialog.findChild(QLineEdit, "Pixel size(nm):").text()
            angle = dialog.findChild(QLineEdit, "Angle (deg.):").text()
            flip = dialog.findChild(QComboBox, "Flip:").currentText()
            nd = CoMx.ndim
            phase = np.zeros_like(CoMx)
            if nd == 2:
                phase = self.idpc_builder(CoMx, CoMy, float(pixel_size_R), float(epsilon), rotations = (float(angle), flip))
            elif nd == 3:
                num_iter = CoMx.shape[0]
                for n in range(num_iter):
                    phase[n] = self.idpc_builder(CoMx[n], CoMy[n], float(pixel_size_R), float(epsilon), rotations = (float(angle), flip))
            self.reconstructions.append( phase )
            self.phase_names.append(method)
            self.phases_list.addItem(method)
            
        elif method == "First-Moment STEM":
            epsilon = dialog.findChild(QLineEdit, "epsilon:").text()
            angle = dialog.findChild(QLineEdit, "Angle (deg.):").text()
            flip = dialog.findChild(QComboBox, "Flip:").currentText()
            nd = CoMx.ndim
            phase = np.zeros_like(CoMx)
            if nd == 2:
                phase = self.FMSTEM_builder(CoMx, CoMy, epsilon=float(epsilon), rotations = (float(angle), flip))
            elif nd == 3:
                num_iter = CoMx.shape[0]
                for n in range(num_iter):
                    phase[n] = self.FMSTEM_builder(CoMx[n], CoMy[n], epsilon=float(epsilon), rotations = (float(angle), flip))
            self.reconstructions.append( phase )
            self.phase_names.append(method)
            self.phases_list.addItem(method)
            
        elif method == "Find CoM(xy)":
           # info = dialog.findChild(QLineEdit, "Processing DPCs:").text()
           # self.show_error_popup(f"{info}")
            self.reconstructions.append(CoMx)
            self.phase_names.append("CoMx")
            self.phases_list.addItem("CoMx")
            self.reconstructions.append(CoMy)
            self.phase_names.append("CoMy")
            self.phases_list.addItem("CoMy")
            
        elif method == "Sum DPCs":
            self.reconstructions.append(CoMx + CoMy)
            self.phase_names.append("Sum DPCs")
            self.phases_list.addItem("Sum DPCs")
            
        elif method == "Create virtual image":          
            semiconv = dialog.findChild(QLineEdit, "Semi-convergence angle (mrad):").text()
            pixelsize = dialog.findChild(QLineEdit, "Pixelsize in detector (mrad/px):").text()
            inner = dialog.findChild(QLineEdit, "Inner radius (mrad):").text()
            outer = dialog.findChild(QLineEdit, "Outer radius (mrad):").text()
            segment = dialog.findChild(QComboBox, "Segment ?:").currentText()
            angle = dialog.findChild(QLineEdit, "Rotating (deg.):").text()     
            if segment == "True":
                virtualImage, detectors, COMs = self.virtual_image(CoMx, float(semiconv), float(pixelsize), float(inner), float(outer), segment, float(angle))
            else:
                virtualImage, detectors = self.virtual_image(CoMx, float(semiconv), float(pixelsize), float(inner), float(outer), segment, float(angle))
                
            self.reconstructions.append( virtualImage )
            self.reconstructions.append( detectors )
            if float(inner) == 0:
                self.phase_names.append("Virtual Bright-field image")
                self.phases_list.addItem("Virtual Bright-field image")
                self.phase_names.append("Virtual Bright-field detector")
                self.phases_list.addItem("Virtual Bright-field detector")
            elif float(inner) != 0 and segment == "False":
                self.phase_names.append("Virtual ADF image")
                self.phases_list.addItem("Virtual ADF image")
                self.phase_names.append("Virtual ADF detector")
                self.phases_list.addItem("Virtual ADF detector")
            elif segment == "True":
                self.phase_names.append("Virtual segmented image")
                self.phases_list.addItem("Virtual segmented image")
                self.phase_names.append("Virtual 4-segmented detectors")
                self.phases_list.addItem("Virtual 4-segmented detectors")
                self.reconstructions.append( COMs )
                self.phase_names.append("Virtual CoMx,y")
                self.phases_list.addItem("Virtual CoMx,y")
        else:
            return

    def virtual_detectors(self, segment, angle):
        if segment == "True":
            ranges = np.array([np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)])
            ranges[0] = np.array([-angle, 90-angle])
            ranges[1] = np.array([90-angle, 180-angle])
            ranges[2] = np.array([180-angle, 270-angle])
            ranges[3] = np.array([270-angle, 360-angle])
        else:
            ranges = np.array([np.array([0,360])])
            
        return ranges
    
    def wavelength_beam(self, voltage_kV):
        m = 9.109383 * (10 ** (-31))  # mass of an electron
        e = 1.602177 * (10 ** (-19))  # charge of an electron
        c = 299792458  # speed of light
        h = 6.62607 * (10 ** (-34))  # Planck's constant
        voltage = voltage_kV * 1000
        numerator = (h ** 2) * (c ** 2)
        denominator = (e * voltage) * ((2 * m * (c ** 2)) + (e * voltage))
        wavelength = (10 ** 9) * ((numerator / denominator) ** 0.5)  # in nm
        return wavelength
    
    def show_tools(self, tools_list):

        filter_name = tools_list.text()
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Parameter settings: {filter_name}")
        layout = QFormLayout(dialog)

        
        def add_spin_box(label, range_vals, step, default):
            spin_box = QSpinBox()
            spin_box.setRange(*range_vals)
            spin_box.setSingleStep(step)
            spin_box.setValue(default)
            spin_box.setObjectName(label)
            layout.addRow(label, spin_box)
            return spin_box

        def add_double_spin_box(label, range_vals, step, default):
            double_spin_box = QDoubleSpinBox()
            double_spin_box.setRange(*range_vals)
            double_spin_box.setSingleStep(step)
            double_spin_box.setValue(default)
            double_spin_box.setObjectName(label)
            layout.addRow(label, double_spin_box)
            return double_spin_box

        def add_line_edit(label, placeholder):
            line_edit = QLineEdit()
            line_edit.setPlaceholderText(placeholder)
            line_edit.setObjectName(label)  # Set a unique name for identification
            layout.addRow(label, line_edit)
            return line_edit

        def add_combo_box(label, items):
            combo_box = QComboBox()
            combo_box.addItems(items)
            combo_box.setObjectName(label)
            layout.addRow(label, combo_box)
            return combo_box
        
        if filter_name == "Gaussian Blur":
            add_double_spin_box("Sigma:", (0.1, 10.0), 0.1, 1.0)

        elif filter_name == "Median Filter":
            add_spin_box("Kernel Size:", (1, 15), 2, 3)

        elif filter_name == "Rotation":
             add_line_edit("Rotating angle (deg.):", "angle")
            
        elif filter_name == "Bandpass filter":
            add_line_edit("Cutoff ratio:", "Set the cutoff ratio on the frequency")
            add_combo_box("Mode Option:", ["high", "low"])
            
        elif filter_name in ["Wiener filter" , "ABS filter"]:                       
            add_line_edit("Cutoff ratio:", "Set the cutoff ratio on the frequency")
            add_combo_box("Lowpass:", ['True', 'False'])
            add_spin_box("Kernel Size:", (1, 15), 2, 3)
            add_spin_box("Wiener order:", (0, 10), 2, 2)     
        elif filter_name == "Nonlinear filter":
            add_line_edit("Cycling:", "Cycling numbers")
            add_combo_box("Filter:", ['Wiener', 'ABS'])
            add_line_edit("Cutoff ratio:", "Set the cutoff ratio on the frequency")
            add_combo_box("Lowpass:", ['True', 'False'])
            add_spin_box("Kernel Size:", (1, 15), 2, 3)
            add_spin_box("Wiener order:", (0, 10), 2, 2)  
    
        # OK and Cancel buttons
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        layout.addRow(ok_button, cancel_button)

        # Connect button signals to dialog slots
        ok_button.clicked.connect(lambda: self.apply_filter(filter_name, dialog))
        cancel_button.clicked.connect(dialog.reject)
        ok_button.clicked.connect(dialog.accept)
        dialog.exec_()

    def apply_filter(self, filter_name, dialog):
        # Get the current image from the active view
        active_view = self.get_active_view()
        if active_view is None:
            return

        image = active_view.getImageItem().image
        filtered_image = None
        
        def put_into_list(data, name):
            self.reconstructions.append(data)
            self.phase_names.append(name)
            self.phases_list.addItem(name)
        
        if filter_name == "Gaussian Blur":
            sigma = dialog.findChild(QDoubleSpinBox).value()
            filtered_image = self.apply_gaussian_blur(image, sigma)
            put_into_list(filtered_image, f"Gaussian blurred image sigma={round(sigma,2)}")
        elif filter_name == "Median Filter":
            kernel_size = dialog.findChild(QSpinBox).value()
            filtered_image = self.apply_median_filter(image, kernel_size)
            put_into_list(filtered_image, f"Median filtered image kernel={kernel_size}")
        elif filter_name == "Rotation":
            rotate = dialog.findChild(QLineEdit).text()
            filtered_image = self.apply_rotate(image, float(rotate))
            put_into_list(filtered_image, f"Rotated image {rotate} deg.")
        elif filter_name == "Bandpass filter":
            cutoff = dialog.findChild(QLineEdit).text()
            mode = dialog.findChild(QComboBox).currentText()
            filtered_image = self.gaussian_bandpass(image, mode, float(cutoff))
            put_into_list(filtered_image, f" Gaussian {mode} bandpassed image")
        elif filter_name in ["Wiener filter" , "ABS filter"]:             
            cutoff = dialog.findChild(QLineEdit).text()
            lowpass = dialog.findChild(QComboBox).currentText()
            delta = dialog.findChild(QSpinBox, "Kernel Size:").value()
            lowpass_order = dialog.findChild(QSpinBox, "Wiener order:").value()  
            if filter_name == "Wiener filter":
                filtered_image = self.wiener_filter(image, delta, lowpass, float(cutoff), lowpass_order)
                put_into_list(filtered_image, f"Wiener filtered image, cutoff={cutoff}")
            elif filter_name == "ABS filter":
                filtered_image = self.abs_filter(image, delta, lowpass, float(cutoff), lowpass_order)
                put_into_list(filtered_image, f"ABS filtered image, cutoff={cutoff}")
        elif filter_name == "Nonlinear filter":                
           num = dialog.findChild(QLineEdit, "Cycling:").text()
           mode = dialog.findChild(QComboBox, "Filter:").currentText()
           cutoff = dialog.findChild(QLineEdit, "Cutoff ratio:").text()
           lowpass = dialog.findChild(QComboBox, "Lowpass:").currentText()
           delta = dialog.findChild(QSpinBox, "Kernel Size:").value()
           lowpass_order = dialog.findChild(QSpinBox, "Wiener order:").value()  
           filtered_image = self.nonlinear_filter(image, float(num), mode, delta, lowpass, float(cutoff), lowpass_order)
           put_into_list(filtered_image, f"Nonlinear filtered image cutoff, ={cutoff}")
        else:
            return

        #active_view.getImageItem().setImage(filtered_image)
        # Close the dialog
        dialog.accept()

    def cartesian_to_polar(self, image_shape):
       sizey, sizex = image_shape
       y, x = np.ogrid[:sizey, :sizex]
       centerx, centery = sizex / 2, sizey / 2
       x = x - centerx
       y = y - centery
       rho = np.hypot(y, x)   
       return rho 

    def apply_gaussian_blur(self, image, sigma):
        return ndimage.gaussian_filter(image, sigma=sigma)

    def apply_median_filter(self, image, kernel_size):
        return ndimage.median_filter(image, size=kernel_size)

    def apply_rotate(self, image, angle):

        return ndimage.rotate(image, angle)

    def gaussian_bandpass(self, img, mode="low", cutoff_ratio=0.1):
        if mode is None:
            mode = "low"
        sizex, sizey = img.shape
        r = self.cartesian_to_polar([sizex, sizey])

        cutoff = sizex * cutoff_ratio
        if mode == "low":
            gaussian_fr = np.exp(- (r**2) / (2 * (cutoff**2)))
        else:
            gaussian_fr = 1 - np.exp(- (r**2) / (2 * (cutoff**2)))

        img_fft = fft.fftshift(fft.fft2(img))
        back = fft.ifft2(fft.ifftshift(img_fft * gaussian_fr))
        return np.real(back)

    def wiener_filter(self, img, delta=5, lowpass='True', lowpass_cutoff=0.3, lowpass_order=2):
        pad_x, pad_y = img.shape[0] // 2, img.shape[1] // 2
        padded_img = np.pad(img, ((pad_x, pad_x), (pad_y, pad_y)), mode='reflect')
        f_img = fft.fftshift(fft.fft2(padded_img))

        fu = np.abs(f_img)
        inverse_fu = np.reciprocal(fu, where = fu !=0)
        inverse_fu[inverse_fu==np.inf] = 1
        fa = self.avg_background(f_img, delta)
        wf = (fu**2 - fa**2)*inverse_fu**2
        wf[wf<0] = 0
        f_img_wf = f_img * wf
    
        if lowpass == 'True':
            sx, sy = padded_img.shape
            r = self.cartesian_to_polar([sx, sy])
            bw = 1/(1+0.414*(r/(lowpass_cutoff * sx))**(2*lowpass_order))
            filtered_fshift = f_img_wf * bw
            img_wf = fft.ifft2(fft.ifftshift(filtered_fshift))            
        else:
            img_wf = fft.ifft2(fft.ifftshift(f_img_wf))
        img_wf = np.real(img_wf)
        return np.single(img_wf[pad_x:-pad_x, pad_y:-pad_y])

    def abs_filter(self, img, delta=5, lowpass=True, lowpass_cutoff=0.4, lowpass_order=2):
        pad_x, pad_y = img.shape[0] // 2, img.shape[1] // 2
        padded_img = np.pad(img, ((pad_x, pad_x), (pad_y, pad_y)), mode='reflect')
        f_img = fft.fftshift(fft.fft2(padded_img))

        fu = np.abs(f_img)
        fa = self.avg_background(f_img, delta)
        inverse_fu = np.reciprocal(fu, where = fu !=0)
        inverse_fu[inverse_fu==np.inf] = 1
        absf = (np.abs(fu) - np.abs(fa))*np.abs(inverse_fu) #the difference with the wiener filter
        absf[absf<0] = 0
        f_img_absf = f_img * absf
    
        if lowpass:
            sx, sy = padded_img.shape
            r = self.cartesian_to_polar([sx, sy])
            bw = 1/(1+0.414*(r/(lowpass_cutoff * sx))**(2*lowpass_order))
            filtered_fshift = f_img_absf * bw
            img_absf = fft.ifft2(fft.ifftshift(filtered_fshift))
        else: 
            img_absf = fft.ifft2(fft.ifftshift(f_img_absf))
        img_absf = np.real(img_absf)        
        return np.single(img_absf[pad_x:-pad_x, pad_y:-pad_y])

    def nonlinear_filter(self, img, N=10, mode='Wiener', delta=10, lowpass = True, lowpass_cutoff=0.3, lowpass_order=2):
        sizex, sizey = img.shape
        r = self.cartesian_to_polar([sizex, sizey])
        cutoff = sizex/2 * lowpass_cutoff
        gaussian_f = np.exp(- (r**2) / (2 * (cutoff**2)))
        x_in = img

        if N ==0:
            N = 1
        for i in range(int(N)): 
            fshift = fft.fftshift(fft.fft2(x_in))
            filtered_fshift = fshift *gaussian_f          

            img_glp = fft.ifft2(fft.ifftshift(filtered_fshift))
            x_lp = np.real(img_glp)
            x_diff = x_in - x_lp
        
            if mode == 'Wiener':
                x_diff_wf = self.wiener_filter(x_diff, delta, lowpass, lowpass_cutoff,  lowpass_order)                                     
                                     
            else:
                x_diff_wf = self.abs_filter(x_diff, delta, lowpass, lowpass_cutoff,  lowpass_order)   
                
            x_in = x_lp + x_diff_wf

        return np.single(x_in)

    def avg_background(self, img, delta=5):
        sizex, sizey = img.shape
        r = self.cartesian_to_polar([sizex, sizey])
        r = r.astype(int)
    
        # Get a Butterworth filter on image to remove the edge effect
        noedgebw = 1/(1+0.414*(r/(0.4 * r.shape[0]))**(2*12))
 
        f_noedge = img * noedgebw
    
        # Light filter the FFT for processing
        f_img = ndimage.median_filter(np.abs(f_noedge), size=5)
    
        # Convert the FFT data into flattened real magnitude data
        f_mag = f_img.flatten()
    
        # Flatten the r array
        r_mag = r.flatten()
    
        # Group and calculate mean for each unique r value
        unique_r = np.unique(r_mag)
        f_bin = [f_mag[r_mag == r] for r in unique_r]
        f_mean0 = [np.mean(bin) for bin in f_bin]
    
        # For each r bin, replace the pixels > mean with mean0, and take the new mean1.
        # Compare the mean1 and mean0, if the difference is < threshold%, stop
        # This needs to be done through all the bins
        f_mean = f_mean0.copy()
        for i, bin in enumerate(f_bin):
            mean = np.mean(bin) * (100 + delta) * 0.01  # Overshoot by the threshold
            diff_pc = np.inf  # Initialize difference as infinity
            while diff_pc > delta and mean > 0:
                bin[bin > mean] = mean  # Replace any data > mean
                new_mean = np.mean(bin) * (100 + delta) * 0.01  # Recalculate mean
                diff_pc = np.abs((new_mean - mean) / mean) * 100  # Calculate the percentage difference
                mean = new_mean  # Update the mean value
            if mean < 0:
                mean = 0
                f_mean[i] = mean  # Update the mean data
    
        # Construct the background array
        f_avg_r = dict(zip(unique_r, f_mean))
        f_avg = np.zeros(f_img.shape)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                f_avg[i, j] = f_avg_r[r[i, j]]
    
        return f_avg

    # Following codes are for phase reconstructions
    
    def idpc_builder(self, CoMx, CoMy, pixel_size_R, epsilon, rotations):
        builder = iDPC_DCTBuilder(CoMx, CoMy, pixel_size_R, epsilon)
        if rotations[1] == "None":            
            angle, flip = builder.optimize_rotation(CoMx, CoMy, pixel_size_R, epsilon)
            idpc_phase = builder.run(angle, flip, dctn_type=2)
            self.show_error_popup(f"Image-diffraction rotation {flip}: {round(angle, 2)} degrees")
        else: 
            if rotations[1] == "True":
                flip = True
            else: flip = False
            idpc_phase = builder.run(rotations[0], flip, dctn_type = 2)
        return idpc_phase

    def FMSTEM_builder(self, CoMx, CoMy, epsilon, rotations):
        builder = FMSTEMBuilder(CoMx, CoMy, epsilon)
        if rotations[1] == "None":            
            angle, flip = builder.optimize_rotation(CoMx, CoMy, epsilon)            
            FM = builder.run(angle, flip)
            self.show_error_popup(f"Image-diffraction rotation {flip}: {round(angle, 2)} degrees")
        else: 
            if rotations[1] == "True":
                flip = True
            else: flip = False
            FM = builder.run(rotations[0], flip)
        return FM
    
    #def OBF_builder(self, ):
        

    def virtual_image(self, datacube, semiconv, pixelsize, inner, outer, segment, angle):

        processor = COMProcessor(datacube, segment, center = None)        
        angle_ranges = self.virtual_detectors(segment, angle)
        if segment == "True":
            virtualImage, mask, COMs = processor.segment_intensities(semiconv, pixelsize, inner, outer, angle_ranges)
        else:

            virtualImage, mask = processor.segment_intensities(semiconv, pixelsize, inner, outer, angle_ranges)
            
        if virtualImage.shape[0] == 1:
            return virtualImage[0], mask[0]
        else: 
            return virtualImage, mask, COMs

        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = STEMImageProcessingGUI()
    gui.show()
    sys.exit(app.exec_())
