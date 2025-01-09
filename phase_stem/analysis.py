import os
import time 
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib_scalebar.scalebar import ScaleBar
import cupy as cp
import scipy.ndimage as ndimage 
from scipy.fft import dctn, idctn
from skimage.registration import phase_cross_correlation
import numpy as np

import cmath 
import math  
from numba import cuda, float64, complex128, int32
import imageio 
import hyperspy.api as hs 
from PIL import Image, ImageDraw, ImageFont
from phase_STEM import tools, denoising
from tqdm.notebook import tqdm
import pyfftw

def nyquist_sampling(angle, wavelength) -> float:
        """
        Nyquist sampling corresponding to the semiangle cutoff of the aperture [Å].
        Semiangle cutoff of the aperture [mrad].
        """
        return 1 / (4 * angle / wavelength * 1e-3)

def periodic_DFT(image, inverse_dft=True):
    """
    It is developed by Lionel Moisan.
    It is helpful to supress the scanning jitter in STEM images.
    The discrete Fourier transform employs:
    import pyfftw

    DOI: 10.1007/s10851-010-0227-1
    https://sbrisard.github.io/posts/20180326-on_the_periodic-plus-smooth_decomposition_of_an_image-07.html
    Compute the periodic component of the 2D image u, which is a real image.

    This function returns the periodic-plus-smooth decomposition of
    the 2D array-like u.

    If inverse_dft is True, then the pair (p, s) is returned
    (p: periodic component; s: smooth component).

    If inverse_dft is False, then the pair

    (np.fft.fft2(p), np.fft.fft2(s))

    is returned.

    This function implements Algorithm 1.
    """
    u = np.asarray(image, dtype=np.float64)

    m, n = u.shape

    arg_m = 2.0 * np.pi * np.fft.fftfreq(m, 1.0)
    cos_m, sin_m = np.cos(arg_m), np.sin(arg_m)
    one_minus_exp_m = 1.0 - cos_m - 1j * sin_m

    arg_n = 2.0 * np.pi * np.fft.fftfreq(n, 1.0)
    cos_n, sin_n = np.cos(arg_n), np.sin(arg_n)
    one_minus_exp_n = 1.0 - cos_n - 1j * sin_n

    w1 = u[:, -1] - u[:, 0]
    w1_dft = pyfftw.interfaces.numpy_fft.fft(w1)
    v_dft = w1_dft[:, None] * one_minus_exp_n[None, :]

    w2 = u[-1, :] - u[0, :]
    w2_dft = pyfftw.interfaces.numpy_fft.fft(w2)
    v_dft += one_minus_exp_m[:, None] * w2_dft[None, :]

    denom = 2.0 * (cos_m[:, None] + cos_n[None, :] - 2.0)
    denom[0, 0] = 1.0
    s_dft = v_dft / denom
    s_dft[0, 0] = 0.0

    if inverse_dft:
        s = pyfftw.interfaces.numpy_fft.ifft2(s_dft)
        return u - s, s
    else:
        u_dft = pyfftw.interfaces.numpy_fft.fft2(u)
        return u_dft - s_dft, s_dft



def get_rotation_and_flip_maxcontrast(CoMx, CoMy, N_thetas, paddingfactor=2,
                                      regLowPass=0.5, regHighPass=100, stepsize=1,
                                      n_iter=1, return_stds=False, process = True):
    """
    This code is copied from py4DSTEM.
    Find the rotation offset between real space and diffraction space, and whether there
    exists a relative axis flip their coordinate systems, starting from the premise that
    the contrast of the phase reconstruction should be maximized when the RQ rotation is
    correctly set.

    The idea of the algorithm is to perform a phase reconstruction for various values of
    the RQ rotation, and with and without an RQ flip, and then calculate the standard
    deviation of the resulting images.  The rotation and flip which maximize the standard
    deviation are then returned. Note that answer should be correct up to a 180 degree
    rotation, corresponding to a complete contrast reversal.  From these two options, the
    correct rotation can then be selected manually by noting that for the correct
    rotation, atomic sites should be bright and the absence of atoms dark.  Physically,
    the presence of two degenerate solutions is related to the electron charge,
    with the incorrect, contrast reversed solution corresponding to electrons with a
    charge of +e.

    Args:
        CoMx (2D array): the x coordinates of the diffraction space centers of mass
        CoMy (2D array): the y coordinates of the diffraction space centers of mass
        N_thetas (int): the number of theta values to use
        regLowPass (float): passed to get_phase_from_CoM; low pass regularization term
            for the Fourier integration operators
        regHighPass (float): passed to get_phase_from_CoM; high pass regularization term
            for the Fourier integration operators
        paddingfactor (int): passed to get_phase_from_CoM; padding to add to the CoM
            arrays for boundry condition handling. 1 corresponds to no padding, 2 to
            doubling the array size, etc.
        stepsize (float): passed to get_phase_from_CoM; the stepsize in the iteration
            step which updates the phase
        n_iter (int): passed to get_phase_from_CoM; the number of iterations
        return_stds (bool): if True, returns the theta values and costs, both with and
            without an axis flip, for all gradient descent steps, for diagnostic purposes
        

    Returns:
        (5-tuple) A 5-tuple containing:

            * **theta**: *(float)* the rotation angle between the real and diffraction
              space coordinates, in radians.
            * **flip**: *(bool)* if True, the real and diffraction space coordinates are
              flipped relative to one another.  By convention, we take flip=True to
              correspond to the change CoMy --> -CoMy.
            * **thetas**: *(float)* returned iff return_costs is True. The theta values.
              In radians.
            * **stds**: *(float)* returned iff return_costs is True. The cost values at
              each gradient descent step for flip=False
            * **stds_f**: *(float)* returned iff return_costs is True. The cost values
              for flip=False
    """

    thetas = cp.linspace(0,2*np.pi,N_thetas)
    stds = cp.zeros(N_thetas)
    stds_f = cp.zeros(N_thetas)
    if process:
        pbar = tqdm(total=(N_thetas*2), desc="Building",unit="iteration", bar_format="{l_bar}{bar} [ time left: {remaining} ]")
    # Unflipped
    for i,theta in enumerate(thetas):
        if process:
            pbar.update(1)
        phase, error = get_phase_from_CoM(CoMx, CoMy, theta=theta, flip=False,
                                          regLowPass=regLowPass, regHighPass=regHighPass,
                                          paddingfactor=paddingfactor, stepsize=stepsize,
                                          n_iter=n_iter, process = False)
        stds[i] = cp.std(phase) #/cp.mean(phase)
        

    # Flipped
    for i,theta in enumerate(thetas):
        if process:
            pbar.update(1)
        phase, error = get_phase_from_CoM(CoMx, CoMy, theta=theta, flip=True,
                                          regLowPass=regLowPass, regHighPass=regHighPass,
                                          paddingfactor=paddingfactor, stepsize=stepsize,
                                          n_iter=n_iter, process = False)
        stds_f[i] = cp.std(phase) #/cp.mean(phase)
    if process:   
        pbar.close()
    flip = cp.max(stds_f)>cp.max(stds)
    if flip:
        theta = thetas[cp.argmax(stds_f)]
    else:
        theta = thetas[cp.argmax(stds)]

    if return_stds:
        return theta, flip, thetas, stds, stds_f
    else:
        return theta, flip



def get_phase_from_CoM(CoMx, CoMy, theta, flip, regLowPass=0, regHighPass=0.001,
                        paddingfactor=2, stepsize=1, n_iter=10, phase_init=None, process = True):
    """
    This code is copied from py4DSTEM.
    Calculate the phase of the sample transmittance from the diffraction centers of mass.
    A bare bones description of the approach taken here is below - for detailed
    discussion of the relevant theory, see, e.g.::

        Ishizuka et al, Microscopy (2017) 397-405
        Close et al, Ultramicroscopy 159 (2015) 124-137
        Wadell and Chapman, Optik 54 (1979) No. 2, 83-96

    The idea here is that the deflection of the center of mass of the electron beam in
    the diffraction plane scales linearly with the gradient of the phase of the sample
    transmittance. When this correspondence holds, it is therefore possible to invert the
    differential equation and extract the phase itself.* The primary assumption made is
    that the sample is well described as a pure phase object (i.e. the real part of the
    transmittance is 1). The inversion is performed in this algorithm in Fourier space,
    i.e. using the Fourier transform property that derivatives in real space are turned
    into multiplication in Fourier space.

    *Note: because in DPC a differential equation is being inverted - i.e. the
    fundamental theorem of calculus is invoked - one might be tempted to call this
    "integrated differential phase contrast".  Strictly speaking, this term is redundant
    - performing an integration is simply how DPC works.  Anyone who tells you otherwise
    is selling something.

    Args:
        CoMx (2D array): the diffraction space centers of mass x coordinates
        CoMy (2D array): the diffraction space centers of mass y coordinates
        theta (float): the rotational offset between real and diffraction space
            coordinates
        flip (bool): whether or not the real and diffraction space coords contain a
                        relative flip
        regLowPass (float): low pass regularization term for the Fourier integration
            operators
        regHighPass (float): high pass regularization term for the Fourier integration
            operators
        paddingfactor (int): padding to add to the CoM arrays for boundry condition
            handling. 1 corresponds to no padding, 2 to doubling the array size, etc.
        stepsize (float): the stepsize in the iteration step which updates the phase
        n_iter (int): the number of iterations
        phase_init (2D array): initial guess for the phase

    Returns:
        (2-tuple) A 2-tuple containing:

            * **phase**: *(2D array)* the phase of the sample transmittance, in radians
            * **error**: *(1D array)* the error - RMSD of the phase gradients compared
              to the CoM - at each iteration step
    """
    assert isinstance(np.bool_(cp.asnumpy(flip)),(bool,np.bool_))
    assert isinstance(paddingfactor,(int,np.integer))
    assert isinstance(n_iter,(int,np.integer))

    # Coordinates
    R_Nx,R_Ny = CoMx.shape
    R_Nx_padded,R_Ny_padded = R_Nx*paddingfactor,R_Ny*paddingfactor #avoid the edge effect in the FFT/iFFT
    #get the Qp = (qx, qy) in the spatial frequency
    qx = cp.fft.fftfreq(R_Nx_padded) #frequency, [0, 1, ...,   n/2-1,     -n/2, ..., -1]
    qy = cp.fft.rfftfreq(R_Ny_padded) #frequency for real input, [0, 1, ...,     n/2-1,     n/2]
    qr2 = qx[:,None]**2 + qy[None,:]**2 #radius of q

    # introducing low-pass and high-pass filters (regHighPass, regLowPass) as regularization parameters
    denominator = qr2 + regHighPass + qr2**2*regLowPass 
    _ = np.seterr(divide='ignore')
    denominator = 1./denominator
    denominator[0,0] = 0
    _ = np.seterr(divide='warn')
    f = 1j * -0.25*stepsize
    qxOperator = f*qx[:,None]*denominator
    qyOperator = f*qy[None,:]*denominator

    # Perform rotation and flipping
    if isinstance(CoMx, np.ndarray):
        CoMx = cp.asarray(CoMx)
    if isinstance(CoMy, np.ndarray):
        CoMy = cp.asarray(CoMy)    
        
    if not flip:
        CoMx_rot = CoMx*cp.cos(theta) - CoMy*cp.sin(theta)
        CoMy_rot = CoMx*cp.sin(theta) + CoMy*cp.cos(theta)
    if flip:
        CoMx_rot = CoMx*cp.cos(theta) + CoMy*cp.sin(theta) 
        CoMy_rot = CoMx*cp.sin(theta) - CoMy*cp.cos(theta) 

    # Initializations
    phase = cp.zeros((R_Nx_padded,R_Ny_padded))
    update = cp.zeros((R_Nx_padded,R_Ny_padded))
    dx = cp.zeros((R_Nx_padded,R_Ny_padded))
    dy = cp.zeros((R_Nx_padded,R_Ny_padded))
    error = cp.zeros(n_iter)
    mask = cp.zeros((R_Nx_padded,R_Ny_padded),dtype=bool)
    mask[:R_Nx,:R_Ny] = True
    maskInv = mask==False
    if phase_init is not None:
        phase[:R_Nx,:R_Ny] = phase_init
    if process:
        pbar = tqdm(total=(n_iter), desc="Building",unit="iteration", bar_format="{l_bar}{bar} [ time left: {remaining} ]")
    # Iterative reconstruction
    sum_xy = cp.sum(CoMx_rot**2 + CoMy_rot**2)
    for i in range(n_iter):
        if process:
            pbar.update(1)
        # Update gradient estimates using measured CoM values
        dx[mask] += CoMx_rot.ravel()
        dy[mask] += CoMy_rot.ravel()
        dx[maskInv] = 0
        dy[maskInv] = 0

        # Calculate reconstruction update
        update = cp.fft.irfft2( cp.fft.rfft2(dx)*qxOperator + cp.fft.rfft2(dy)*qyOperator)

        # Apply update
        phase += stepsize*update

        # Measure current phase gradients
        dx = (cp.roll(phase,(-1,0),axis=(0,1)) - cp.roll(phase,(1,0),axis=(0,1))) / 2.
        dy = (cp.roll(phase,(0,-1),axis=(0,1)) - cp.roll(phase,(0,1),axis=(0,1))) / 2.

        # Estimate error from cost function, RMS deviation of gradients
        #Making the error minimum (least-squares problem)
        xDiff = dx[mask] - CoMx_rot.ravel()
        yDiff = dy[mask] - CoMy_rot.ravel()
        #error reduction
        #error[i] = cp.sqrt(cp.mean((xDiff-cp.mean(xDiff))**2 + (yDiff-cp.mean(yDiff))**2))
        error[i] = cp.sum(xDiff**2 + yDiff**2)/sum_xy
        # Halve step size if error is increasing
        if i>0:
            if error[i] > error[i-1]:
                stepsize /= 2
    if process:
        pbar.close()
    phase = phase[:R_Nx,:R_Ny]

    return phase, error

class iDPC_DCTBuilder:
    """
    The class utilizes DCT instead of DFT to reconstruct the boundary-artifact-free iDPC-STEM image.
    Reference: 
    1. Microscopy, 2017, 397–405
    2. Optics Express Vol. 22, Issue 8, pp. 9220-9244 (2014)
    Usage:
        1. builder = iDPCBuilder(CoMx, CoMy, pixel_size_R = 0.01, epsilon = 0.001)

        2. crop_CoMx = ps.tools.crop_matrix(CoMx, [1024,1024], [512,512])
           crop_CoMy = ps.tools.crop_matrix(CoMy, [1024,1024], [512,512])
        3. optimal_theta, optimal_flip = builder.optimize_rotation(crop_CoMx, crop_CoMy, thetas = np.linspace(60, 120, 30), 
                                                                pixel_size_R=0.037, epsilon = 0.001)
        # Step 4: Perform phase reconstruction
            phase = builder.run(optimal_theta, optimal_flip, dctn_type=2, mask=None)
        you can call variant using:
        for example: 
            dpc = builder.phase
    """
    def __init__(self, CoMx: np.ndarray, CoMy: np.ndarray, pixel_size_R: float = 0.001, epsilon: float = 0.01):
                  

        if CoMx.shape != CoMy.shape:
            raise ValueError("CoMx and CoMy must have the same shape")

        self.pad_x, self.pad_y = CoMx.shape
    
        self.CoMx = np.pad(CoMx, ((self.pad_x, self.pad_x), (self.pad_y, self.pad_y)), mode='reflect')
        self.CoMy = np.pad(CoMy, ((self.pad_x, self.pad_x), (self.pad_y, self.pad_y)), mode='reflect')


        self.epsilon = epsilon
        self.sampling = pixel_size_R
        
        self.R_Nx, self.R_Ny = self.CoMx.shape
        
        self._initialize_variables()

    def _initialize_variables(self):

        # Create meshgrid for denominator calculation
        x = np.arange(self.R_Nx)
        y = np.arange(self.R_Ny)
        xm, yn = np.meshgrid(x, y, indexing ='ij')
        # Calculate denominator
        self.denominator = (np.sin(xm * np.pi / (2 * (self.R_Nx-1))))**2 + (np.sin(yn * np.pi / (2 * (self.R_Ny-1))))**2
        self.denominator += self.epsilon
        self.denominator[self.denominator == 0] = np.inf
        self.denominator = 1./self.denominator

    @staticmethod
    def _create_Laplacian(obj_x, obj_y, dxy):
        """
        It is to calculate the first-order derivative of a Laplacian matrix and tune its boundary with the Neumann boundary condition.
    
        """
        if obj_x.shape != obj_y.shape:
            raise ValueError("CThe input matrix should be the same shape.")
        px, py = obj_x.shape
        Lx = np.zeros((px, py))
        Ly = np.zeros((px, py))
        # The boundary of Laplacian matrix is given by Neumann boundary condition
        Lx = np.gradient(obj_x, dxy, axis = 0)
        Ly = np.gradient(obj_y, dxy, axis = 1)
       # Ly[:, 0] = (obj_y[:, 0] + obj_y[:, 1]) / (2 * dxy)
     #   Ly[:, -1] = -(obj_y[:, -2] + obj_y[:, -1]) / (2 * dxy)
     #   Lx[0, :] = (obj_x[0, :] + obj_x[1, :]) / (2 * dxy)
      #  Lx[-1, :] = -(obj_x[-2, :] + obj_x[-1, :]) / (2 * dxy)
       # Ly[1:-1, 1:-1] = (obj_y[1:-1, 2:] - obj_y[1:-1, :-2]) / (2 * dxy)
       # Lx[1:-1, 1:-1] = (obj_x[2:, 1:-1] - obj_x[:-2, 1:-1]) / (2 * dxy)    

        return Lx + Ly

    def run(self, theta: float, flip: bool, dctn_type: int = 2, mask:np.ndarray=None):
        """
        Perform non-iterative reconstruction of the phase.

        Parameters:
            theta (float): Rotation angle in radians.
            flip (bool): Whether to flip the coordinates.
            dctn_type: int, the type of DCT, default 2.
            mask: np.ndarray
        
        Returns:
            np.ndarray: Reconstructed phase.
        """
        if mask is not None:
            if mask.shape != (self.R_Nx, self.R_Ny):
                mx, my = mask.shape
                ratioX = self.R_Nx / mx
                ratioY = self.R_Ny / my
                mask = ndimage.zoom(mask, (ratioX, ratioY))

        theta = np.radians(theta)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        
        if not flip:
            self.CoMx_rot = self.CoMx * cos_theta - self.CoMy * sin_theta
            self.CoMy_rot = self.CoMx * sin_theta + self.CoMy * cos_theta
        else:
            self.CoMx_rot = self.CoMx * cos_theta + self.CoMy * sin_theta
            self.CoMy_rot = self.CoMx * sin_theta - self.CoMy * cos_theta


        self.phase = np.zeros((self.R_Nx, self.R_Ny))

        # Calculate Laplacian matrix
        Laplacian = self._create_Laplacian(self.CoMx_rot, self.CoMy_rot, self.sampling)

        # Compute phase
        dct_lap = dctn(Laplacian, type=dctn_type, norm='ortho')
        #dct_lap = dct(dct(Laplacian, type=dctn_type, axis=0, norm='ortho'), type=dctn_type, axis=1, norm='ortho') 

        if mask is None:
            self.phase = idctn(dct_lap * self.denominator, type=dctn_type, norm='ortho')
            #self.phase = idct(idct(dct_lap*self.denominator, type=2, axis=1, norm='ortho'), type=2, axis=0, norm='ortho')
        else:
             mask = np.fft.fftshift(mask) 
             self.phase = idctn(mask*dct_lap* self.denominator, type=dctn_type, norm='ortho')
             #self.phase = idct(idct(mask*dct_lap* self.denominator, type=2, axis=1, norm='ortho'), type=2, axis=0, norm='ortho')
        self.phase *= self.sampling**2

        
        return self.phase[self.pad_x: -self.pad_x, self.pad_y: -self.pad_y]

    @classmethod
    def optimize_rotation(cls, CoMx: np.ndarray, CoMy: np.ndarray, thetas: list, pixel_size_R: float = 0.001, epsilon: float = 1e-3, plot = False):
        """
        Optimize theta and determine if flipping is needed.

        Parameters:
        CoMx, CoMy (np.ndarray): Input coordinate arrays.
        thetas (list): the range of thetas for searching, eg.g np.linspace(0, 360, 360).
        pixel_size_R (float): Step size for the update.
        epsilon (float): Controls the size of high-pass filtering.

        Returns:
        Tuple[float, bool]: Optimal theta and flip status.
        """
        reconstructor = cls(CoMx, CoMy, pixel_size_R, epsilon)
        N_thetas = len(thetas)
        stds = np.zeros((2, N_thetas))

        with tqdm(total=N_thetas*2, desc="Calculating", unit="iteration") as pbar:
            for flip in range(2):
                for i, theta in enumerate(thetas):
                    pbar.update(1)                
                    phase = reconstructor.run(theta, bool(flip), dctn_type=2, mask=None)                    
                    stds[flip, i] = np.std(phase)
        
        flip = np.max(stds[1]) > np.max(stds[0])
        theta = thetas[np.argmax(stds[1 if flip else 0])]
        if plot:
            cls._plot_optimization_results(thetas, stds)
        print(f"Does it need to flip: {flip}")
        print(f"The rotation angle is: {theta:.2f} degrees")
        return theta, flip

    @staticmethod
    def _plot_optimization_results(thetas: np.ndarray, stds: np.ndarray):
        """Plot the optimization results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
        x = thetas
        ax1.scatter(x, stds[0], color='#2393CD', label='No flip')
        ax2.scatter(x, stds[1], color='#003561', label='Flip')
        ax1.legend(loc='upper right', shadow=True, fontsize=14)
        ax2.legend(loc='upper right', shadow=True, fontsize=14)
        ax1.set_xlabel('Degrees (deg)', fontsize=18)
        ax1.set_ylabel('Standard Deviation', fontsize=18)
        ax2.set_xlabel('Degrees (deg)', fontsize=18)
        ax2.set_ylabel('Standard Deviation', fontsize=18)
        plt.show()



class Noniterate_iDPCBuilder:
    """
    Noniterate_iDPCBuilder class for phase reconstruction using optimized DPC methods.
    This method can avoid/suppress the boundary-artefacts by antisymmetric extending image.
    Alarm: If you got an image with bright and dark edges, it indicates that the input image is overflow in computation.
    The solution for this issue is to change the dtype of your input images, like converting it from np.float64 to np.int16.
    e.g. data.astype(np.int16)

    Usage:
        1. builder = DFT_iDPCBuilder(CoMx, CoMy, epsilon=0.001, mask = None)

        2. crop_CoMx = ps.tools.crop_matrix(CoMx, [1024, 1024], [512, 512])
           crop_CoMy = ps.tools.crop_matrix(CoMy, [1024, 1024], [512, 512])
        3. optimal_theta, optimal_flip = builder.optimize_rotation(crop_CoMx, crop_CoMy, thetas=np.arange(-90, 91, 1), plot=True)
        4. phase = builder.run(optimal_theta, optimal_flip, expanding=True)
    """

    def __init__(self, CoMx: np.ndarray, CoMy: np.ndarray, epsilon: float = 0.01, mask = None):
        if CoMx.shape != CoMy.shape:
            raise ValueError("CoMx and CoMy must have the same shape")

        self.CoMx = CoMx
        self.CoMy = CoMy
        self.regHighpass = epsilon
        self.R_Nx, self.R_Ny = CoMx.shape
        self.mask = mask

    def _expand_matrix(self, matrix, mode='CoMx'):
        """
        Expand the matrix by combining it with its flipped versions.
    
        Parameters:
        - matrix: The input 2D matrix.
        - mode: The type of expansion ('CoMx' or 'CoMy').
    
        Returns:
        - A new matrix expanded based on the specified mode.
        """
        y_flipped = np.flip(matrix, axis=1)
        x_flipped = np.flip(matrix, axis=0)
        both_flipped = np.flip(matrix, axis=(0, 1))
    
        if mode == 'CoMx':
            top_row = np.hstack((-both_flipped, -x_flipped))
            bottom_row = np.hstack((y_flipped, matrix))
        elif mode == 'CoMy':
            top_row = np.hstack((-both_flipped, x_flipped))
            bottom_row = np.hstack((-y_flipped, matrix))
        else:
            raise ValueError("Invalid mode. Use 'CoMx' or 'CoMy'.")
        return np.vstack((top_row, bottom_row))

    def _initialize_variables(self, DPCx, DPCy):
        R_Nx_padded, R_Ny_padded = DPCx.shape
        kx = np.fft.fftfreq(R_Nx_padded).astype(
            np.float32
            )
        ky = np.fft.fftfreq(R_Ny_padded).astype(
            np.float32
            )
        kya, kxa = np.meshgrid(ky, kx)

        k_den = kxa**2 + kya**2
        k_den += self.regHighpass 
        k_den[k_den==0] = np.inf
        k_den = 1 / k_den

        qxOperator = -1j * kxa * k_den
        qyOperator = -1j * kya * k_den
        
        return qxOperator, qyOperator

    def run(self, theta: float, flip: bool, expanding: bool = True):
        """
        Perform iterative reconstruction of the phase.
        """

        theta_rad = np.radians(theta)
        cos_theta, sin_theta = np.cos(theta_rad), np.sin(theta_rad)

        if not flip:
            CoMx_rot = self.CoMx * cos_theta - self.CoMy * sin_theta
            CoMy_rot = self.CoMx * sin_theta + self.CoMy * cos_theta
        else:
            CoMx_rot = self.CoMx * cos_theta + self.CoMy * sin_theta
            CoMy_rot = self.CoMx * sin_theta - self.CoMy * cos_theta

        if expanding:
            DPCx = self._expand_matrix(CoMx_rot, mode='CoMy')#avoiding the data overflow
            DPCy = self._expand_matrix(CoMy_rot, mode='CoMx')
        else:
            DPCx = CoMx_rot
            DPCy = CoMy_rot
            
        # the qx and qy are antisymmetric in the Cartesian coordinate system
        qxOperator, qyOperator = self._initialize_variables(DPCx, DPCy)

        self.dpc = (pyfftw.interfaces.numpy_fft.fft2(DPCx) * qxOperator +
                    pyfftw.interfaces.numpy_fft.fft2(DPCy) * qyOperator)
        if self.mask is not None and self.mask.shape ==self.dpc.shape:
            self.mask = pyfftw.interfaces.numpy_fft.ifftshift(self.mask)
            update = pyfftw.interfaces.numpy_fft.ifft2(self.dpc*self.mask)
        else: update = pyfftw.interfaces.numpy_fft.ifft2(self.dpc)

        if expanding:
            self.phase = update[self.R_Nx:, self.R_Ny:]
        else:
            self.phase = update

        return np.real(self.phase)

    @classmethod
    def optimize_rotation(cls, CoMx: np.ndarray, CoMy: np.ndarray, thetas: list, epsilon: float = 1e-3, plot=False):
        """
        Optimize theta and determine if flipping is needed.
        """
        reconstructor = cls(CoMx, CoMy, epsilon, mask = None)
        num_thetas = len(thetas)
        edge = 16
        stds = np.zeros((2, 2, num_thetas))  # Shape: (flip_state, positive/negative, theta_index)

        with tqdm(total=num_thetas * 2, desc="Calculating", unit="iteration") as pbar:
            for flip in range(2):  # flip_state = 0 (no flip), 1 (flip)
                for i, theta in enumerate(thetas):
                    pbar.update(1)
                    phase = reconstructor.run(theta, bool(flip), expanding=False)
                    cropped_phase = phase[edge:-edge, edge:-edge] # avoid the influence of intensities on the edge
                    stds[flip, 0, i] = np.std(cropped_phase)
                    stds[flip, 1, i] = np.std(-cropped_phase)

        # Find the maximum standard deviation and its position
        max_value = np.max(stds)
        flip, sign, theta_idx = np.unravel_index(np.argmax(stds), stds.shape)

        # Choose which array to plot based on sign (positive/negative)
        if sign == 0:
            if plot:
                cls._plot_optimization_results(thetas, stds[:, 0])
            print("It is needed to preserve the intensity")
        else:
            if plot:
                cls._plot_optimization_results(thetas, stds[:, 1])
            print("It is needed to reverse the intensity")
        theta = thetas[theta_idx]
        flip = bool(flip)
            
        print(f"Does it need to flip: {bool(flip)}")
        print(f"The rotation angle is: {theta:.2f} degrees")
        return theta, bool(flip)

    @staticmethod
    def _plot_optimization_results(thetas: np.ndarray, stds: np.ndarray):
        """Plot the optimization results."""
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
        for ax, std, label in zip(axes, stds, ['No flip', 'Flip']):
            ax.scatter(thetas, std)
            ax.set_xlabel('Degrees (deg)')
            ax.set_ylabel('Standard Deviation')
            ax.legend([label])
        plt.show()

class Noniterate_FMBuilder:
    """
    This method can avoid/suppress the boundary-artefacts by antisymmetric extending image.
    The principal is to build a complex plane using DPCx and DPCy.
    Imaging DPC = DPCx + 1j * DPCy
    So,  FT{IiDPC(rp)}(kp) = [kx - i* ky]*[ FT{IDPCx(rp)} + i*FT{IDPCy(rp)} ](kp) / (2*pi*(kp**2 + epsilon))

    where kp**2 = kx**2 + ky**2 and epsilon is to avoid zero in denominator.
    Reference:
    1. Pierre Bon, Serge Monneret & Benoit Wattellier. Noniterative boundary-artifact-free wavefront reconstruction from its derivatives.
        Applied Optics, (2012), 51, 23

    Noniterate_FMBuilder class for phase reconstruction with boundary-artefacts free.
    Usage:
        1. builder = Noniterate_FMBuilder(CoMx, CoMy, epsilon = 0.01,mask = None)

        2. crop_CoMx = ps.tools.crop_matrix(CoMx, [1024, 1024], [512, 512])
           crop_CoMy = ps.tools.crop_matrix(CoMy, [1024, 1024], [512, 512])
        3. optimal_theta, optimal_flip = builder.optimize_rotation(crop_CoMx, crop_CoMy, thetas=np.arange(-90, 91, 1), plot=True)
        4. phase = builder.run(optimal_theta, optimal_flip, expanding=True)
    """

    def __init__(self, CoMx: np.ndarray, CoMy: np.ndarray, epsilon: float = 0.01, mask = None):
        if CoMx.shape != CoMy.shape:
            raise ValueError("CoMx and CoMy must have the same shape")

        self.CoMx = CoMx
        self.CoMy = CoMy
        self.R_Nx, self.R_Ny = CoMx.shape
        self.mask = mask
        self.regHighpass = epsilon
    def _expand_matrix(self, matrix, mode='CoMx'):
        """
        Expand the matrix by combining it with its flipped versions.
    
        Parameters:
        - matrix: The input 2D matrix.
        - mode: The type of expansion ('CoMx' or 'CoMy').
    
        Returns:
        - A new matrix expanded based on the specified mode.
        """
        y_flipped = np.flip(matrix, axis=1)
        x_flipped = np.flip(matrix, axis=0)
        both_flipped = np.flip(matrix, axis=(0, 1))
    
        if mode == 'CoMx':
            top_row = np.hstack((-both_flipped, -x_flipped))
            bottom_row = np.hstack((y_flipped, matrix))
        elif mode == 'CoMy':
            top_row = np.hstack((-both_flipped, x_flipped))
            bottom_row = np.hstack((-y_flipped, matrix))
        else:
            raise ValueError("Invalid mode. Use 'CoMx' or 'CoMy'.")
    
        return np.vstack((top_row, bottom_row))


    def _initialize_variables(self, DPCx, DPCy):
        R_Nx_padded, R_Ny_padded = DPCx.shape
        kx = np.fft.fftfreq(R_Nx_padded).astype(
            np.float32
            )
        ky = np.fft.fftfreq(R_Ny_padded).astype(
            np.float32
            )
        kya, kxa = np.meshgrid(ky, kx)
        k_den = kxa**2 + kya**2
        k_den += self.regHighpass 
        k_den[k_den==0] = np.inf
        k_den = 1 / k_den

        Operator =  k_den * (kxa - 1j*kya)

        return Operator

    def run(self, theta: float, flip: bool, expanding: bool = True):
        """
        Perform iterative reconstruction of the phase.
        """

        theta_rad = np.radians(theta)
        cos_theta, sin_theta = np.cos(theta_rad), np.sin(theta_rad)

        if not flip:
            CoMx_rot = self.CoMx * cos_theta - self.CoMy * sin_theta
            CoMy_rot = self.CoMx * sin_theta + self.CoMy * cos_theta
        else:
            CoMx_rot = self.CoMx * cos_theta + self.CoMy * sin_theta
            CoMy_rot = self.CoMx * sin_theta - self.CoMy * cos_theta

        if expanding:
            DPCx = self._expand_matrix(CoMx_rot, mode='CoMx')
            DPCy = self._expand_matrix(CoMy_rot, mode='CoMy')
        else:
            DPCx = CoMx_rot
            DPCy = CoMy_rot
            
        Operator = self._initialize_variables(DPCx, DPCy)

        self.dpc = (pyfftw.interfaces.numpy_fft.fft2(DPCx) + 1j * pyfftw.interfaces.numpy_fft.fft2(DPCy)) * Operator
                    
        if self.mask is not None:
            mx, my = self.mask.shape
            px, py = DPCx.shape
            ratio_x, ratio_y = px/mx, py/my
            if ratio_x != 1 and ratio_y!=1:
                self.mask = ndimage.zoom(self.mask, (ratio_x, ratio_y))
            self.mask = pyfftw.interfaces.numpy_fft.ifftshift(self.mask)
            self.update = pyfftw.interfaces.numpy_fft.ifft2(self.dpc*self.mask)
        else: self.update = pyfftw.interfaces.numpy_fft.ifft2(self.dpc)

        if expanding:
            self.phase = self.update[self.R_Nx:, self.R_Ny:]
        else:
            self.phase = self.update

        return np.real(self.phase)

    @classmethod
    def optimize_rotation(cls, CoMx: np.ndarray, CoMy: np.ndarray, thetas: list, epsilon: float = 1e-3, plot= False):
        """
        Optimize theta and determine if flipping is needed.
        """
        reconstructor = cls(CoMx, CoMy, epsilon, mask = None)
        num_thetas = len(thetas)
        edge = 16
        stds = np.zeros((2, 2, num_thetas))  # Shape: (flip_state, positive/negative, theta_index)

        with tqdm(total=num_thetas * 2, desc="Calculating", unit="iteration") as pbar:
            for flip in range(2):  # flip_state = 0 (no flip), 1 (flip)
                for i, theta in enumerate(thetas):
                    pbar.update(1)
                    phase = reconstructor.run(theta, bool(flip), expanding=False)
                    cropped_phase = phase[edge:-edge, edge:-edge] # avoid the influence of intensities on the edge
                    stds[flip, 0, i] = np.std(cropped_phase)
                    stds[flip, 1, i] = np.std(-cropped_phase)

        # Find the maximum standard deviation and its position
        max_value = np.max(stds)
        flip, sign, theta_idx = np.unravel_index(np.argmax(stds), stds.shape)

        # Choose which array to plot based on sign (positive/negative)
        if sign == 0:
            if plot:
                cls._plot_optimization_results(thetas, stds[:, 0])
            print("It is needed to preserve the intensity")
        else:
            if plot:
                cls._plot_optimization_results(thetas, stds[:, 1])
            print("It is needed to reverse the intensity")
        theta = thetas[theta_idx]
        flip = bool(flip)
            
        print(f"Does it need to flip: {bool(flip)}")
        print(f"The rotation angle is: {theta:.2f} degrees")
        return theta, bool(flip)


    @staticmethod
    def _plot_optimization_results(thetas: np.ndarray, stds: np.ndarray):
        """Plot the optimization results."""
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
        for ax, std, label in zip(axes, stds, ['No flip', 'Flip']):
            ax.scatter(thetas, std)
            ax.set_xlabel('Degrees (deg)')
            ax.set_ylabel('Standard Deviation')
            ax.legend([label])
        plt.show()

class FMSTEMBuilder:
    """
        Perform phase recovery using the approach described in:
        (the ﬁrst-moment STEM [FM-STEM] reconstruction)
            1. Close et al, Ultramicroscopy 159 (2015) 124-137
            2. Rodenburg J. M. and Bates R. H. T. 1992. The theory of super-resolution electron microscopy via Wigner-distribution deconvolution.
                Philosophical Transactions of the Royal Society of London. Series A: Physical and Engineering Sciences339521–553
                http://doi.org/10.1098/rsta.1992.0050

        The principal can be regarded as the object's and probe's Wigner distribution.

        --- FT{IiDPC(rp)}(kp) = [i*kx + ky]*[ FT{IDPCx(rp)} + i*FT{IDPCy(rp)} ](kp) / (2*pi*(kp**2 + regHighpass))

        If no filter applied (regHighpass = 0), the reconstructed iDPC image is needed to deconvolute using a filter or 
        directly apply a frequency band pass filter (phase_STEM.EMFilters.frequency_pass_filter or gaussian_bandpass)
        
        The filter should be built from the probe, whose intensity fits the normal distribution.
        The deconvolution method :
        skimage.restoration.wiener/unsupervised_wiener/richardson_lucy

        Usage:
            step1: reconstructor = FMSTEMBuilder(CoMx, CoMy, stepsize=1, n_iter=30, regHighpass=1e-4)
            step2: optimal_theta, optimal_flip = reconstructor.optimize_rotation(CoMx, CoMy, thetas = np.linspace(10, 90, 30))
                    "thetas" is a list containing the degrees for finding the best rotation angle and judging whether the flip is needed.
            step3: phase, error = reconstructor.run(optimal_theta, optimal_flip, point = [512,1024], crop_size =512, process=True)
            step4: reconstructor.display()
    """
    def __init__(self, CoMx: np.ndarray, CoMy: np.ndarray, stepsize: float = 1, n_iter: int = 10, regHighpass: float = 1e-4):
        """
        Initialize the iDPCReconstructor.

        Parameters:
            CoMx, CoMy (np.ndarray): Input arrays.
            stepsize (float): Step size for the update.
            n_iter (int): Number of iterations for optimization.
            regHighpass (float): Controls the size of bandpass filter.
        """
        if CoMx.shape != CoMy.shape:
            raise ValueError("CoMx and CoMy must have the same shape")
        #normalizing the intensity within [0, 1]
        self.CoMx = CoMx #(CoMx - np.min(CoMx))/(np.max(CoMx) - np.min(CoMx))
        self.CoMy = CoMy #(CoMy - np.min(CoMy))/(np.max(CoMy) - np.min(CoMy))
        self.stepsize = stepsize
        self.n_iter = n_iter
        self.regHighpass = regHighpass
        
        self.R_Nx, self.R_Ny = CoMx.shape
        self.size = self.R_Nx * self.R_Ny
        self.qx = pyfftw.interfaces.numpy_fft.fftfreq(self.R_Nx)
        self.qy = pyfftw.interfaces.numpy_fft.fftfreq(self.R_Ny)
        self._initialize_variables()

    def _initialize_variables(self):
        """Initialize variables needed for the reconstruction."""
        qr1 = 1j * self.qx[:, None] + self.qy[None, :]
        self.r = self.qx[:, None]**2 + self.qy[None, :]**2
        self.qr = self.r + self.regHighpass

        self.denominator = np.zeros(self.qr.shape, dtype=np.complex128)
        none_zero = self.qr != 0
        self.denominator[none_zero] = 1 / self.qr[none_zero]
        self.denominator *= qr1
        
    def _crop_image(self, image, centre, crop_size):
        start_row = max(0, centre[0] - crop_size // 2)
        end_row = min(self.R_Nx, start_row + crop_size)
        start_col = max(0, centre[1] - crop_size // 2)
        end_col = min(self.R_Ny, start_col + crop_size)
        cropped_matrix = image[start_row:end_row, start_col:end_col]    
        return cropped_matrix
        
    def run(self, theta: float, flip: bool, process: bool = False, epsilon=1e-6):
        """
        Perform iterative reconstruction of the phase.

        Parameters:
            theta (float): Rotation angle in degrees.
            flip (bool): Whether to flip the coordinates.
            point: (Turple): the coordinate of point
            crop_size: (Int): the size wanted to crop
            process (bool): Whether to show progress bar.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Reconstructed phase and error history.
        """

        theta = np.radians(theta)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        
        if not flip:
            self.CoMx_rot = self.CoMx * cos_theta - self.CoMy * sin_theta
            self.CoMy_rot = self.CoMx * sin_theta + self.CoMy * cos_theta
        else:
            self.CoMx_rot = self.CoMx * cos_theta + self.CoMy * sin_theta
            self.CoMy_rot = self.CoMx * sin_theta - self.CoMy * cos_theta  

        self.phase = np.zeros((self.R_Nx, self.R_Ny))
        self.error = np.zeros(self.n_iter)

        self.dpc = - (pyfftw.interfaces.numpy_fft.fft2(self.CoMx_rot)  +
                    pyfftw.interfaces.numpy_fft.fft2(self.CoMy_rot) * 1j)* self.denominator

        iterator = tqdm(range(self.n_iter), desc="Building", unit="iteration") if process else range(self.n_iter)
        
        # Initialize momentum term
        momentum = 0.6
        prev_grad = np.zeros_like(self.dpc)
        for i in iterator:
            self.phase = pyfftw.interfaces.numpy_fft.ifft2(self.dpc)
            self.phase = np.real(self.phase)
            dx = (np.roll(self.phase, -1, axis=0) - np.roll(self.phase, 1, axis=0)) / 2.0
            dy = (np.roll(self.phase, -1, axis=1) - np.roll(self.phase, 1, axis=1)) / 2.0

            xDiff = dx - self.CoMx_rot
            yDiff = dy - self.CoMy_rot

            #L2-norm
            self.error[i] = np.sqrt(np.sum((xDiff - self.CoMx_rot) ** 2) + np.sum((yDiff - self.CoMy_rot) ** 2)) / self.size                           

            grad = - (pyfftw.interfaces.numpy_fft.fft2(xDiff)  +
                      pyfftw.interfaces.numpy_fft.fft2(yDiff) * 1j)* self.denominator
            #if np.linalg.norm(grad) is equal to zero, it means all of elements in grad are zero.
            grad_norm = grad / np.linalg.norm(grad) if np.linalg.norm(grad) != 0 else grad  
            # Update the gradient with momentum
            grad_momentum = momentum * prev_grad + (1 - momentum) * grad_norm
            prev_grad = grad_momentum
            # Adaptive step size adjustment
            if i > 0:
                if self.error[i] >= self.error[i-1]:
                    self.stepsize = 0.5 * self.stepsize
                else: 
                    self.dpc += grad_momentum * self.stepsize
            else: self.dpc += grad_momentum * self.stepsize
            #controlling the convergence
            if i > 2 and np.abs((self.error[i] - self.error[i-2])/self.error[i]) < epsilon:
                print(f"Iteration: {i}: break")
                print(f"You can change the iterations through <epsilon>")
                break
        self.phase = pyfftw.interfaces.numpy_fft.ifft2(self.dpc)
        self.phase = np.real(self.phase)
        return self.phase, self.error

    @classmethod
    def optimize_rotation(cls, CoMx: np.ndarray, CoMy: np.ndarray, thetas: list, 
                            n_iter: int = 1, stepsize: float = 1, regHighpass: float = 1e-3):
        """
        Optimize theta and determine if flipping is needed.

        Parameters:
            CoMx, CoMy (np.ndarray): Input coordinate arrays.
            thetas (list): the range of thetas for searching, eg.g np.linspace(0, 360, 360).
            n_iter (int): Number of iterations for optimization.
            stepsize (float): Step size for the update.
            regHighpass (float): Controls the size of high-pass filtering.

        Returns:
        Tuple[float, bool]: Optimal theta and flip status.
        """
        reconstructor = cls(CoMx, CoMy, stepsize, n_iter, regHighpass)
        N_thetas = len(thetas)
        stds = np.zeros((2, N_thetas))
        with tqdm(total=N_thetas*2, desc="Calculating", unit="iteration") as pbar:
            for flip in range(2):
                for i, theta in enumerate(thetas):
                    pbar.update(1)                
                    phase, _ = reconstructor.run(theta, bool(flip), process=False)
                    stds[flip, i] = np.std(phase)
        
        flip = np.max(stds[1]) > np.max(stds[0])
        theta = thetas[np.argmax(stds[1 if flip else 0])]
        
        cls._plot_optimization_results(thetas, stds)
        print(f"Does it need to flip: {flip}")
        print(f"The rotation angle is: {theta:.2f} degrees")
        return theta, flip

    @staticmethod
    def _plot_optimization_results(thetas: np.ndarray, stds: np.ndarray):
        """Plot the optimization results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        x = thetas
        ax1.scatter(x, stds[0], color='#2393CD', label='No flip')
        ax2.scatter(x, stds[1], color='#003561', label='Flip')
        ax1.legend(loc='upper right', shadow=True, fontsize=14)
        ax2.legend(loc='upper right', shadow=True, fontsize=14)
        ax1.set_xlabel('Degrees (deg)', fontsize=18)
        ax1.set_ylabel('Error', fontsize=18)
        ax2.set_xlabel('Degrees (deg)', fontsize=18)
        ax2.set_ylabel('Error', fontsize=18)
        plt.tight_layout()
        plt.show()

    def display(self):
        """Display the reconstructed phase and error history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        im = ax1.imshow(self.phase, cmap='viridis', interpolation = 'gaussian')
        ax1.set_title('Reconstructed Phase')
        plt.colorbar(im, ax=ax1)
        indices = np.nonzero(self.error)
        ax2.scatter(indices, self.error[indices])
        ax2.set_title('Error History')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Error')

        plt.tight_layout()
        plt.show()


def wavelength_beam(voltage_kV):
    """
    Calculates the relativistic electron wavelength
    in angstroms based on the microscope accelerating
    voltage

    Arg:
    ----------
    voltage_kV: float
                microscope operating voltage in kilo
                electronVolts

    Return:
    -------
    wavelength: float
                relativistic electron wavelength in nm

    """
    m = 9.109383 * (10 ** (-31))  # mass of an electron
    e = 1.602177 * (10 ** (-19))  # charge of an electron
    c = 299792458  # speed of light
    h = 6.62607 * (10 ** (-34))  # Planck's constant
    voltage = voltage_kV * 1000
    numerator = (h ** 2) * (c ** 2)
    denominator = (e * voltage) * ((2 * m * (c ** 2)) + (e * voltage))
    wavelength = (10 ** 9) * ((numerator / denominator) ** 0.5)  # in nm
    return wavelength

def interaction_constant(E):
    """
    Calculates the interaction constant, sigma, to convert electrostatic potential (in
    V Angstroms) to radians. Units of this constant are rad/(V Angstrom).
    See, e.g., Kirkland, 'Advanced Computing in Electron Microscopy', Eq. 2.5.

    Args:
        E (float): electron energy, in keV

    Returns:
        (float): relativistically corrected electron mass
    """
    E *=1e3
    h = 6.62607004e-34      # Planck's constant in Js
    me = 9.10938356e-31     # Electron rest mass in kg
    qe = 1.60217662e-19     # Electron charge in C
    hc = 1.23984193e3       # Planck's constant times the speed of light in eV nm
    m0c2 = 5.109989461e5    # electron rest mass, in eV
    k0 = np.sqrt( E*(E + 2*m0c2) / hc) # Electron wavenumber in inverse nm

    gamma = (m0c2 + E)/m0c2   # Relativistic mass correction
    return 2*np.pi*gamma*me*qe*1e-20/(k0*h**2)

def chi_omega(kx, ky, ab, wavelength):
    """
    Reference: 1. Earl J.Kirkland. On the optimum probe in aberration corrected ADF-STEM. Ultramicroscopy 111 (2011) 1523-1530.
    2. H. Sawada, et.al..Measurement method of aberration from Ronchigram by autocorrelation function.Ultramicroscopy 108 (2008) 1467-1475.

    \chi ( {\alpha, \phi} ) = \displaystyle{{2\pi} \over \lambda} \mathop \sum \limits_{n,m\;}
   \displaystyle{{C_{n,m}\alpha ^{n + 1}\cos ( {m( {\phi -\phi_{n,m}} ) } ) } \over {n + 1}}

    Aberration coefficients noted by Krivanek symbol
    aberrations.real stores the real-value measured by Cs-corrector, 
    while the imaginary components keep the orientation in the rochigram
    Cn,m and ϕn,m describe a geometric aberration's magnitude and orientation, 
    m is the order of rotational symmetry (0 for cylindrically symmetric aberrations, 
    otherwise 2π/m is the smallest angle such that the phase shift of the aberration is equivalent), 
    n is the order of the aberration.

        ab['C1'],     #C1, defocus
        ab['A1'],       #A1
        ab['B2'],        #B2, coma
        ab['A2'],          #A2
        ab['C3'],           #C3, spherical aberration
        ab['A3'],             #A3
        ab['S3'],               #S3, star aberration
        ab['A4']                  #A4

    The observed Ronchigram is the electron probability density on the diffraction plane and is equivalent to 
    the square of the modulus of the Fourier transform of the transmitted wavefunction g(k).
    """
    if isinstance(ab, dict):
        aberrations = np.array([

        ab['C1'],     #C1
        ab['A1'],       #A1
        ab['B2'],        #B2
        ab['A2'],          #A2
        ab['C3'],           #C3
        ab['A3'],             #A3
        ab['S3'],               #S3
        ab['A4']                  #A4
          ], dtype=np.complex128)
    elif isinstance(ab, np.ndarray) and len(ab)>=8:
        aberrations = ab
    else: print("Error: please input correct aberrations!")

    phi = np.arctan2(ky, kx) # azimuthal, orientation
    alpha = np.arctan2(np.sqrt(kx ** 2 + ky ** 2), 1/wavelength) #radial, convergence angle
    
    chi_value = (aberrations[0].real*(alpha**2/2) +                                     #defocus C1 /C10 (n=1, m=0)
                aberrations[1].real*(alpha**2/2)*np.cos(2*(phi-aberrations[1].imag)) +  #A1/C12+j*phi12 (n=1, m=2)
                aberrations[2].real*(alpha**3/3)*np.cos(1*(phi-aberrations[2].imag)) +  #Axial coma:B2/C21+j*phi21 (n=2, m=1)
                aberrations[3].real*(alpha**3/3)*np.cos(3*(phi-aberrations[3].imag)) +  #A2/C23+j*phi23 (n=2, m=3)
                aberrations[4].real*(alpha**4/4) +                                      #3rd order spherical:C3/C30 (n=3, m=0)
                aberrations[5].real*(alpha**4/4)*np.cos(4*(phi-aberrations[5].imag)) +  #A3/C34+j*phi34 (n=3, m=4)
                aberrations[6].real*(alpha**4/4)*np.cos(2*(phi-aberrations[6].imag)) +  #3rd order axial star:S3/C32+j*phi32 (n=3, m=2)
                aberrations[7].real*(alpha**5/5)*np.cos(5*(phi-aberrations[7].imag)) )  #A4/C45+j*phi45 (n=4, m=5)

    return  (2*np.pi*chi_value)/wavelength  #wavenumber

def chi_function(kx, ky, ab, wavelength):
    """
    The expression of the geometric aberrations is referred from:
        https://www.sciencedirect.com/science/article/pii/S0304399116303874
        Ultramicroscopy, Volume 182, , November 2017, Pages 195-204

    """
    if isinstance(ab, dict):
        aberrations = np.array([

        ab['C1'],     #C10
        ab['A1'],       #C12
        ab['B2'],        #C21, axial coma
        ab['A2'],          #C23
        ab['C3'],           #C30
        ab['A3'],             #C34, 4-fold
        ab['S3'],               #C32, star aberration
        ab['A4']                  #C45
          ], dtype=np.complex128)
    elif isinstance(ab, np.ndarray) and len(ab)>=8:
        aberrations = ab
    else: print("Error: please input correct aberrations!")

    phi = np.arctan2(ky, kx) # azimuthal, orientation
    alpha = np.arctan2(np.sqrt(kx ** 2 + ky ** 2), 1/wavelength) #radial, convergence angle

    array = 1/ 2* alpha**2* (
                    aberrations['C1'].real
                    + aberrations['A1'].real * np.cos(2 * (phi - aberrations['A1'].imag))
                )              
    array = array + (1/ 3
                * alpha**3
                * (
                    aberrations['B2'].real * np.cos(phi - aberrations['B2'].imag)
                    + aberrations['A2'].real * np.cos(3 * (phi - aberrations['A2'].imag))
                )                              
            )
    array = array + (
                1
                / 4
                * alpha**4
                * (
                    aberrations['C3'].real
                    + aberrations['S3'].real * np.cos(2 * (phi - aberrations['S3'].imag))
                    + aberrations['A3'].real * np.cos(4 * (phi - aberrations['A3'].imag))
                )
            )
    array = array + (
                1
                / 5
                * alpha**5
                * (
                    aberrations['A4'].real * np.cos(5 * (phi - aberrations['A4'].imag))
                )
            )
    


    return  (2*np.pi*array)/wavelength  #wavenumber

def construct_probe(pixel, semi_conv, aberration_coeffs, wavelength):
    """
    It is for reconstructing STEM probe beam and showing it in real space and reciprocal space.
    The probe is determined by the aberrations.
    It defaults the field of view is 10 Å * 10 Å in real space.
    Args:
        pixel: int, the pixel size of probe shown in image
        semi_conv: float, the semi-convergence angle (unit: rad) of probe
        aberration_coeffs: dictionary, storing the aberration coefficiencies
        wavelength: float, the wavelength of probe ( unit: nm )
    return:
        probe_r: np.array, the reconstructed probe in real space
        probe_k: np.array, the reconstructed probe in k-space
    """
    alpha = np.linspace(-semi_conv, semi_conv, pixel*2)

    kx = ky = alpha/wavelength # k range
    kx, ky = np.meshgrid(kx, ky)
    chi_k = chi_omega(kx, ky, aberration_coeffs, wavelength)    
    aperture_k = aperture(kx, ky, semi_conv, wavelength)
    probe_k = aperture_k * np.exp(-1j * chi_k)   
    probe_k /= np.sqrt(np.sum(np.square(np.abs(probe_k))))
    qsize = pixel
    qx = np.fft.fftshift(np.fft.fftfreq(qsize, d=1/qsize))
    qy = np.fft.fftshift(np.fft.fftfreq(qsize, d=1/qsize))
    qx, qy = np.meshgrid(qx, qy)
    chi_q = chi_omega(qx, qy, aberration_coeffs, wavelength)    
    aperture_q = aperture(qx, qy, semi_conv, wavelength)
    probe_q = aperture_q * np.exp(-1j * chi_q)
    probe_q /= np.sqrt(np.sum(np.square(np.abs(probe_q))))
    probe_r = np.fft.fftshift(np.fft.ifft2(probe_q))
    
    FOV = 5
    extend = [-FOV,FOV,-FOV,FOV]
    fig, axes = plt.subplots(2,2, figsize=(10, 10))
    fig.suptitle('Reconstructed probe beam')
    img =axes[0,0].imshow(np.absolute(probe_r), extent = extend)
    axes[0,0].set_title('Probe beam in r-space')
    axes[0,0].set_xlabel('Distance (Å)')
    fig.colorbar(img, ax=axes[0,0],shrink=0.8)

    expanded_array = tools.expand_matrix(probe_k.real, [pixel*4, pixel*4])
    range = [-2*semi_conv*1e3, 2*semi_conv*1e3, -2*semi_conv*1e3, 2*semi_conv*1e3]
    p = axes[0,1].imshow(expanded_array, extent = range)
    axes[0,1].set_title('probe beam in k-space')
    axes[0,1].set_xlabel('kx (mrad)')
    axes[0,1].set_ylabel('ky (mrad)')
    fig.colorbar(p, ax=axes[0,1],shrink=0.8)
    axes[1,0].plot(np.linspace(-FOV, FOV, qsize), np.absolute(probe_r)[int(qsize/2), :])
    axes[1,0].set_title('Intensity profile of probe beam in r-space along x-axis')
    axes[1,0].set_xlabel('x (Å)')
    axes[1,0].set_ylabel('y (Å)')
    axes[1,0].set_ylabel('Intensity (a.u.)')
    axes[1,1].plot(np.linspace(-semi_conv*1e3, semi_conv*1e3, pixel*2), pixel*(probe_k.real)[int(pixel),:])
    axes[1,1].set_title('Intensity profile of probe beam in k-space along x-axis')
    axes[1,1].set_xlabel('k (mrad)')
    axes[1,1].set_ylabel('Normalized intensity')
    
    plt.tight_layout()
    plt.show()
    
    return probe_r, probe_k


def aperture(qx, qy, alpha_max, lam):
    """
    Return the aperture intensity given (qx, qy) points, wavelength, and angular ranges.

    Args:
        qx (float): Qx components
        qy (float, float, 1D): Qy components
        lam (float): wavelength in nm
        alpha_max (float): maximum allowed angle in rad

    Returns:
        float, aperture intensity
    """
    qx2 = qx**2
    qy2 = qy**2
    q = np.sqrt(qx2 + qy2)
    ktheta = np.arctan2(q , 1/lam)
    
    return (ktheta < alpha_max)

def spatial_frequency(pixel, resolution, dimension=1):
    """
    It is used to generate coordinates in q_space.
    On a rectangular grid of Nx*Ny grid points or pixels, given an orthogonal cell with the sidelengths Lx and Ly in the x and y-direction, 
    the real space sampling is dx = Lx/Nx in x and dy = Ly/Ny in y.

    x_i = i * dx, i = 0,1, ... , N_x - 1 
    y_j = j * dy, j = 0,1, ... , N_y - 1 

    The Fourier transform of this grid of values (or image) will also have the same grids.
    however, in reciprocal space the sampling is determined by the supercell dimensions given by the inverse relations:
    dkx = 1/Lx, dky = 1/Ly
    The reciprocal space coordinates take on values:
    kx_i = i * dkx - kx_max, i = 0,1, ... , N_x - 1 
    ky_j = j * dky - ky_max, j = 0,1, ... , N_y - 1 
    kx_max = 1/(2*dx)
    ky_max = 1/(2*dy)

    As "dimension=1" means the coordinates are one-dimension, while "dimension=2" will give two-dimensional coordinates.
    GPU device uses diemnsion=1, while CPU device uses dimension=2
    """
    if np.isscalar(pixel):
        px = py = int(pixel)
    else: 
        px = int(pixel[0])
        py = int(pixel[1])

    qMax = 1 / resolution
    dqx   = 1 / (resolution * px)
    dqy   = 1 / (resolution * py)
    qxx = np.arange((-qMax + dqx) / 2, (qMax + dqx) / 2, dqx)
    qyy = np.arange((-qMax + dqy) / 2, (qMax + dqy) / 2, dqy)

    if dimension==2:
        QX, QY = np.meshgrid(qxx, qyy) 
        return QX, QY
    elif dimension==1:
        return qxx, qyy
    else: print("Error: using dimension=1 or dimension=2 for 1D or 2D")



def K_space_1D(collection, segment, size, wavelength):    
    """
    This code is to create virtual grid coordinates of segmented detectors in k-space.
    It builds an aperture for each segment.
    
    Kmax(nm-1) = semi_convergence angle(rad)/wavelength(nm)
    
    Args:
    
        collection: a list recording the collection angle (in rad) of the detector, e.g. (0.008, 0.042) rad
        
        segment: the geographic shape of segments in detector, discribed using angles in degrees, e.g. (-45, 45) degree
        
        size: how many grids of one segment divided, it is suggested that this value is set as 2*semi-convergence angle of beam
              e.g. if the semi-convergence angle is 17 mrad, then the size is suggested as 2*17 = 34
              
        wavelength: the wave length of beam
        
    return:
        (x, y): coordinates in k-space

    """
    r1 = (collection[0]/2)/ wavelength #collection is in radians
    
    r2 = (collection[1]/2)/ wavelength 

    x = np.linspace(-r2, r2, size+1)
    y = np.linspace(-r2, r2, size+1)

    X, Y = np.meshgrid(x, y)

    R = X**2 + Y**2

    mask_R = (R >= r1**2) & (R <= r2**2) 
    
    mask_x, mask_y = X[mask_R], Y[mask_R]

    Theta = np.round(np.degrees(np.arctan2(mask_y , mask_x)))
    Theta2 = Theta.copy()
    Theta[Theta<0]+=360
  
    if segment[0] < 0:
        mask_Theta2 = (Theta2 >= segment[0]) & (Theta2 <= segment[1])
        x_corr, y_corr = mask_x[mask_Theta2], mask_y[mask_Theta2]
    else: 
        mask_Theta = (Theta >= segment[0]) & (Theta <= segment[1])
        x_corr, y_corr = mask_x[mask_Theta], mask_y[mask_Theta]

    return x_corr , y_corr
    

def beta_function_CPU(qx, qy,parameters, segment, aberration_coeffs, slices=1, process = False):
    """
    Generating the integrated PCTF(phase contrast transfer function, 
    reference: DOI: 10.1016/j.ultramic.2018.08.008) for thick samples
    based on the assumpation of ignoring multiple scattering.
    The thick sample is treated as a stack of thinly sliced samples, which are approached to WPOA.
    PCTF is based on the assumpation of the WPOA (weak phase object approximation: exp(iσV(x)) = 1+iσV(x)) used in thin sample.
    The phase aberration χ(k, z) = pi*lamda*z *k**2 + χ0(k), where the χ0(k) is the pahse aberration without defocus.
    z is positive for the incident direction of the electron beam.
    The defocus Δf is defined with reference to the entrance surface, i.e., Δf is
    equal to the z-coordinate of the entrance surface, and positive and
    negative defocus represent over- and under-focus, respectively.

    example: 1. if slices= 1 , then this function is the same as the single PCTF.


    """
    try: from tqdm.notebook import tqdm
    except ImportError: print("There is no tqdm module!")
    
    [sizeX, sizeY] = qx.shape
    tk = parameters["sample thickness(nm)"]
    size = parameters["virtual grids in one segment detector"]
    df = aberration_coeffs['C1'] #defocus
    wavelength = parameters["wavelength(nm)"]
    collection_angle = parameters["collection angles(rad)"]
    semi_conv = parameters["semi_convergence angle(rad)"]
    if slices is None:
        slices =1
    OptimumFilter = np.zeros((sizeX, sizeY), dtype=np.complex128) 
    #generating coordinates of segmented detectors in k space
    #(kx1, ky1) consisting of coordinates in k space
    kx1, ky1 = K_space_1D(collection_angle, segment, size, wavelength)
    #px = len(kx1) #normalizing the probe beam as 1
    chi_1 = chi_omega(kx1, ky1, aberration_coeffs, wavelength)
    S1 = aperture(kx1, ky1, semi_conv, wavelength)

    if process: #reminding the running time
        beginning = time.time()
        if slices:
            print(f"There will be {slices} slices for computation!\n")
        pbar = tqdm(total=sizeX* sizeY*(slices), desc="Building",unit="iteration", bar_format="{l_bar}{bar} [ time left: {remaining} ]")
        
    for n in range(slices):
        temp_in = np.zeros((sizeX, sizeY), dtype=np.complex128) 
        dz = df + n* tk/(slices-1) if slices!=1 else df
        if slices==1 and df !=0:
            chi_K1 = chi_1
        else:
            chi_K1 = np.pi*wavelength*dz*(kx1**2 + ky1**2) + chi_1      
        
        beta_q1 = S1*np.exp(-1j*chi_K1)   
        normalizer = np.sum(np.square(np.abs(beta_q1)))
        #calculation for each shift (kx-qx, ky-qy)
        for i in range (sizeX):
            for j in range (sizeY):                          
                kx2 = kx1 - qx[i, j]
                ky2 = ky1 - qy[i, j]   
                S2 = aperture(kx2, ky2, semi_conv, wavelength) 
                chi_K2 = chi_omega(kx2, ky2, aberration_coeffs, wavelength) + np.pi*wavelength*dz*(kx2**2 + ky2**2)
                beta_q2 = S2*np.exp(-1j*chi_K2)

                kx3 = kx1 + qx[i, j]
                ky3 = ky1 + qy[i, j]                
                S3 = aperture(kx3, ky3, semi_conv, wavelength)
                chi_K3 = chi_omega(kx3, ky3, aberration_coeffs, wavelength) + np.pi*wavelength*dz*(kx3**2 + ky3**2)               
                beta_q3 = S3*np.exp(-1j*chi_K3) 

                temp_array = np.conjugate(beta_q1)*beta_q2 - beta_q1*np.conjugate(beta_q3)

                OptimumFilter[i, j] += np.sum(temp_array)/(slices) 
                if process:
                    pbar.update(1)
               
    if process:
        pbar.close()
        print(f"The whole process takes {round(time.time()-beginning)} seconds.")

    return OptimumFilter, normalizer

def phase_filters_CPU(ab, segments, parameters, slices = 1, process = False):

    """
    It is to build frequency filters for the OBF image reconstruction.
    
    Single-slice mode is assumed that the sample is a whole;
    while the multi-slice mode (slices>1) is ragarded the sample sliced into multi-layers, 
    which the corresponding phase contrast transfer function is the average of cntrast transfer function.
    The details are discussed in the reference: 
    1. Rodenburg, J. M., Mccallum, B. C. & Nellist, P. D. Experimental Tests on Double-Resolution Coherent Imaging via STEM. Ultramicroscopy vol. 48 (1993)304-314.
    2. 1. DOI: 10.1016/j.ultramic.2018.08.008

    Args:
        ab: a dictionary, storing the aberrations used for building the frequency-filters
        segments: np.array, providing the geometric segments in the DPC detetor
        parameters: dictionary, recording the key information of experiments for the calculation.
        slices: int, meaning the sample is treated as multi-slice
        process: True or False, showing the calculating process using a process bar.
    

    Return:
        OptimumFilters: ndarray with a shape of (len(segments), pixels*pixels)
    
    """
    try: from tqdm.notebook import tqdm
    except ImportError: print("There is no tqdm module!")
    if np.isscalar(parameters["pixelnumber of filters"]):
        num_pointx = num_pointy = int(parameters["pixelnumber of filters"])
    else:
        num_pointx = int(parameters["pixelnumber of filters"][0])
        num_pointy = int(parameters["pixelnumber of filters"][1])
    wavelength = parameters["wavelength(nm)"]
    OptimumFilter = [np.zeros((num_pointx, num_pointy), dtype=np.complex128) for _ in range(len(segments))]
    qx, qy = spatial_frequency((num_pointx, num_pointy), parameters["pixel size(nm)"], dimension=2)

    #freq_limit = parameters["cutoff_frequency(rad)"] # rad
    #qMax = (2 * freq_limit)/wavelength # nm-1, based on the unit of wavelength
    #r_squared = (qx[:None]**2 + qy[None:]**2)
    # mask = r_squared < qMax**2

    norm = 0
    if process:
        beginning = time.time()   
        
        print(f"The specimen is treated as {slices} slices!\n")
        pbar = tqdm(total=len(segments), desc="Building",unit="iteration", bar_format="{l_bar}{bar} [ time left: {remaining} ]")
                      
    for i in range(len(segments)):
        if process:
            timestart = time.time()    
                              
        OptimumFilter[i], px = beta_function_CPU(qx, qy, parameters, segments[i], ab, slices, process)
        norm += px
        if process:
            pbar.update(1)
            print(f"--- {(len(segments)-i)*round(time.time()-timestart)} seconds left!")
    if process:
        pbar.close()
        print(f"The whole process takes {round(time.time()-beginning)} seconds.")
    
    return OptimumFilter, norm

@cuda.jit
def integrated_beta_function_GPU_kernel(Qx_all, Qy_all, Kx_all, Ky_all, alpha, wavelength,
                            aberrations, dz, output):
    """
    alpha: the probe semi-convergence angle (rad)
    wavelength: the beam wave length, the default value is given (cooresponding to 300 kV)
    aberrations: np.ndarray
    dz: float, the defocusing value 
    Return:
    output: ndarray, the constructed filters
    """
    TWO_PI_OVER_WAVELENGTH = 2 * math.pi / wavelength
    PI_WAVELENGTH_DZ = math.pi * wavelength * dz
    def aperture2(qx, qy, alpha, wavelength, ratio):
        qx2 = qx ** 2
        qy2 = qy ** 2
        q = math.sqrt(qx2 + qy2)
        ktheta = math.asin(q * wavelength)
        w = alpha*ratio
        n = alpha*(1-ratio)
        if ktheta <= n:
            val = float(1)
        elif ktheta > n and ktheta < alpha:
            val = math.cos((q-n)*math.pi/(2*w))
        else: val = float(0)
        return val
    
    def chi_func(kx, ky, aberrations, wavelength):
        phi = math.atan2(ky, kx)                                                          # orientation
        alpha = math.atan2(math.sqrt(kx ** 2 + ky ** 2), 1 / wavelength)                  #convergence angle
        chi_value = (aberrations[0].real*(alpha**2/2) +                                   #defocus C1 /C10 (n=1, m=0)
                aberrations[1].real*(alpha**2/2)*math.cos(2*(phi-aberrations[1].imag)) +  #A1/C12+j*phi12 (n=1, m=2)
                aberrations[2].real*(alpha**3/3)*math.cos(1*(phi-aberrations[2].imag)) +  #Axial coma:B2/C21+j*phi21 (n=2, m=1)
                aberrations[3].real*(alpha**3/3)*math.cos(3*(phi-aberrations[3].imag)) +  #A2/C23+j*phi23 (n=2, m=3)
                aberrations[4].real*(alpha**4/4) +                                        #3rd order spherical:C3/C30 (n=3, m=0)
                aberrations[5].real*(alpha**4/4)*math.cos(4*(phi-aberrations[5].imag)) +  #A3/C34+j*phi34 (n=3, m=4)
                aberrations[6].real*(alpha**4/4)*math.cos(2*(phi-aberrations[6].imag)) +  #3rd order axial star:S3/C32+j*phi32 (n=3, m=2)
                aberrations[7].real*(alpha**5/5)*math.cos(5*(phi-aberrations[7].imag)) )  #A4/C45+j*phi45 (n=4, m=5)
        return TWO_PI_OVER_WAVELENGTH*chi_value

    n, m = cuda.grid(2)   

    if n < (len(Qx_all)* len(Qy_all)) and m < (len(Kx_all)* len(Ky_all)):
        
        iqx = n//len(Qx_all)       
        iqy = (n-iqx*len(Qx_all))
        
        ikx = m //len(Kx_all)       
        iky = (m-ikx*len(Kx_all))
        
        Qx = Qx_all[iqx]
        Qy = Qy_all[iqy]

        Kx = Kx_all[ikx]
        Ky = Ky_all[iky]
        ksquare = Kx**2 + Ky**2
        chi = chi_func(Kx, Ky, aberrations, wavelength)
        chi_K = chi + PI_WAVELENGTH_DZ*ksquare
        A = aperture2(Kx, Ky, alpha, wavelength, 0) * cmath.exp(-1j * chi_K) 
        chi_KplusQ = chi_func(Kx + Qx, Ky + Qy, aberrations, wavelength) + PI_WAVELENGTH_DZ*((Kx + Qx)**2 + (Ky + Qy)**2)
        A_KplusQ = aperture2(Kx + Qx, Ky + Qy, alpha, wavelength, 0) * cmath.exp(-1j * chi_KplusQ)
        chi_KminusQ = chi_func(Kx - Qx, Ky - Qy, aberrations, wavelength) + PI_WAVELENGTH_DZ*((Kx - Qx)**2 + (Ky - Qy)**2)
        A_KminusQ = aperture2(Kx - Qx, Ky - Qy, alpha, wavelength, 0) * cmath.exp(-1j * chi_KminusQ)

        beta_Q = A.conjugate() * A_KminusQ - A * A_KplusQ.conjugate()   
        cuda.atomic.add(output.real, (iqy, iqx), beta_Q.real)
        cuda.atomic.add(output.imag, (iqy, iqx), beta_Q.imag)

def beta_function_GPU_reconstruction(Qx_all, Qy_all, Kx_all, Ky_all, alpha, wavelength, aberration_coeffs, dz, output):
    """
    It recalls the kernel of 'integrated_beta_function_GPU_kernel' to run the calculation using cuda.jit decorator.
    Here the threads grid size and the blocks grid size are set according to the size of Qx, Qy, Kx, and Ky.
    """
    threadsperblock = (16,16)
    #blockspergrid_x = math.ceil((len(Qx_all)*len(Qy_all)) / threadsperblock[0])
    #blockspergrid_y = math.ceil((len(Kx_all)*len(Ky_all)) / threadsperblock[1])
    #blockspergrid = (blockspergrid_x, blockspergrid_y) 
    grid_dim_x = (len(Qx_all) * len(Qy_all) + threadsperblock[0] - 1) // threadsperblock[0]
    grid_dim_y = (len(Kx_all) * len(Ky_all) + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (grid_dim_x, grid_dim_y)
    cuda.synchronize()
    integrated_beta_function_GPU_kernel[blockspergrid, threadsperblock](Qx_all, Qy_all, Kx_all, Ky_all, alpha, wavelength,
                           aberration_coeffs, dz, output)
    cuda.synchronize()

def phase_filters_GPU(ab, segments, parameters, slices = 1, process = True):

    """
    It is another way for frequency filters reconstruction.
    The algorithm is the same as the function "phase_filters_CPU".

    The normalization is conducted for each (K, Q):
    the reference is : https://doi.org/10.1016/j.ultramic.2016.09.002

    Args:

        ab: a dictionary, storing the aberrations used for building the frequency-filters

        aberrations = np.array([

        ab['C1'],     #C1
        ab['A1'],       #A1
        ab['B2'],        #B2
        ab['A2'],          #A2
        ab['C3'],           #C3
        ab['A3'],             #A3
        ab['S3'],               #S3
        ab['A4']                  #A4
          ], dtype=np.complex128)

        segments: np.array, providing the geometric segments in the DPC detetor
        parameters: dictionary, recording the key information of experiments for the calculation.
        slices: a two-member list, the first one: True, meaning the sample is treated as multi-slice; False, the sample is a whole.
                the second is a integer value, which is the slice number of the sample, working as the first one is True.

        process: True or False, showing the calculating process using a process bar.
        single_side_band: if True, using the single-side-band filters

    Return:

        OptimumFilters: ndarray with a shape of (len(segments), pixels*pixels)
    """
    try: from tqdm.notebook import tqdm
    except ImportError: print("There is no tqdm module!")

    
    aberrations = np.array([

        ab['C1'],     #C1
        ab['A1'],       #A1
        ab['B2'],        #B2
        ab['A2'],          #A2
        ab['C3'],           #C3
        ab['A3'],             #A3
        ab['S3'],               #S3
        ab['A4']                  #A4
          ], dtype=np.complex128)
                            
    df = ab['C1'] #defocus
    tk = parameters["sample thickness(nm)"]
    #make sure slices[1]!=0
    if slices is None:
        slices =1
        tk = 1

    if np.isscalar(parameters["pixelnumber of filters"]):
        num_pointx = num_pointy = int(parameters["pixelnumber of filters"])
    else:
        num_pointx = int(parameters["pixelnumber of filters"][0])
        num_pointy = int(parameters["pixelnumber of filters"][1])

    collection_angle = parameters["collection angles(rad)"]
    alpha = parameters["semi_convergence angle(rad)"]
    wavelength = parameters["wavelength(nm)"]

    OptimumFilter = [np.zeros((num_pointx, num_pointy), dtype=np.complex128) for _ in range(len(segments))]
    qx, qy = spatial_frequency((num_pointx, num_pointy), parameters["pixel size(nm)"], dimension=1)

    #QXX, QYY = np.meshgrid(qx, qy)
    #freq_limit = parameters["cutoff_frequency(rad)"] # rad
    #qMax = (2 * freq_limit)/wavelength # nm-1, based on the unit of wavelength
    #r_squared = (QXX[:None]**2 + QYY[None:]**2)
    #mask = r_squared < qMax**2

    #transferring the parameters to GPU device
    Qx_all = cuda.to_device(qx)
    Qy_all = cuda.to_device(qy)
    
    if process:
        beginning = time.time()
        pbar = tqdm(total=len(segments)*slices, desc="Building",unit="iteration", bar_format="{l_bar}{bar} [ time left: {remaining} ]")
   
    d_aberrations = cuda.to_device(aberrations)
        
    size = parameters["virtual grids in one segment detector"] #pixel sizes of segmented detector

    temp_in = [np.zeros((num_pointx, num_pointy), dtype=np.complex128) for _ in range(len(segments))]
    norm = 0
    for n in range(slices):
        dz = df + n* tk/(slices-1) if (slices-1)!=0 else df
        
        for i in range(len(segments)):
            if process:
                starting = time.time()
            
            kx, ky  = K_space_1D(collection_angle, segments[i], size, wavelength)
            norm += len(kx)**2
            Kx_all = cuda.to_device(kx)
            Ky_all = cuda.to_device(ky)
            d_output = cuda.to_device(temp_in[i])

            beta_function_GPU_reconstruction(Qx_all, Qy_all, Kx_all, Ky_all, alpha, wavelength, d_aberrations, dz, d_output)

            OptimumFilter[i] += d_output.copy_to_host()/(slices)
            if process:
                pbar.update(1)
        if process:
            print(f"--- {(len(segments)-i)*round(time.time()-starting)} seconds left!")
    if process:
        pbar.close()
        print(f"The whole process takes {round(time.time()-beginning)} seconds.")

    return OptimumFilter, norm

class OBFBuilder:
    def __init__(self, DPC_imgs, PCTFs, parameters, mask=None):
        """
        Class to reconstruct the OBF image based on segmented images using pyFFTW for enhanced performance.

        Args:
            DPC_imgs (list): A list of segmented images (numpy.ndarray).
            PCTFs (list): A list of corresponding phase filters (numpy.ndarray).
            parameters (dict): Dictionary containing key information for the experiment such as:
                - 'wavelength(nm)': Wavelength of light used.
                - 'collection angles(rad)': Collection angles of the imaging system.
                - 'pixel size(nm)': the real size of each pixel.
            mask: np.ndarray, which is applied on the Fourier domain for the image registration.
                  The shape of mask should be the same with the images (DPC_imgs).
                  A mask could promote the registration of images with low SNR.

        Usage example:
        obf_reconstructor = ps.analysis.OBFBuilder(DPC_imgs, PCTFs, parameters, mask = None)

        #OBF_imgs is the image in real space, while OBF_Q is the recovered segmented image in Fourier domain
        OBF_imgs, OBF_Q = obf_reconstructor.reconstruct_OBF()

        """
        self.DPC_imgs = DPC_imgs
        self.PCTFs = PCTFs
        self.resolution = parameters["pixel size(nm)"]
        self.num = len(DPC_imgs)
        self.sizeX_img, self.sizeY_img = DPC_imgs[0].shape
        self.sizeX, self.sizeY = PCTFs[0].shape
        self.ratioX = self.sizeX_img / self.sizeX
        self.ratioY = self.sizeY_img / self.sizeY

        if mask is not None:
            if mask.shape == DPC_imgs[0].shape:
                self.mask = mask
            else:
                self.mask = ndimage.zoom(mask, (self.ratioX, self.ratioY))
            
        else: self.mask = 1
        
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
        OBF_Q = pyfftw.empty_aligned((self.num, self.sizeX_img, self.sizeY_img), dtype=np.complex128)
        OBF_Q.fill(0)
        for n in range(self.num):
            wq = np.conj(phase_filters[n]) * weightingInv / d_Q[n]
            dft_DPCs = pyfftw.empty_aligned((self.sizeX_img, self.sizeY_img), dtype=np.complex128)
            dft_DPCs[:] = pyfftw.interfaces.numpy_fft.fft2(self.DPC_imgs[n])
            dft_DPCs = pyfftw.interfaces.numpy_fft.fftshift(dft_DPCs)
            OBF_Q[n] = dft_DPCs * wq * self.mask
        return OBF_Q

    def align_images(self, OBF_Q, upsampling = 1):
        """ Align segmented images using the phase cross-correlation method. """
        values = np.zeros(self.num)
        for i in range(self.num):
            values[i] = np.std(OBF_Q[i].real)
        index = np.argmax(values)
        reference = OBF_Q[index]
        referred_image = pyfftw.interfaces.numpy_fft.ifft2(pyfftw.interfaces.numpy_fft.ifftshift(reference))
        self.aligned_images = [referred_image.real]
        
        for i in range(0, self.num):
            if i != index:
                compare = pyfftw.interfaces.numpy_fft.ifft2(pyfftw.interfaces.numpy_fft.ifftshift(OBF_Q[i]))
                moving, _, _ = phase_cross_correlation(reference, OBF_Q[i], space='fourier', upsample_factor = upsampling)
                corrected_image = ndimage.shift(compare.real, moving)
                self.aligned_images.append(corrected_image)

        #reshape the aligned image
        OBFimage = np.sum(np.array(self.aligned_images), axis=0)
        img_width, img_height = OBFimage.shape
        center_x, center_y = img_width // 2, img_height // 2
        half_sizex, half_sizey = self.sizeX_img//2, self.sizeY_img//2
        start_x = max(center_x - half_sizex, 0)
        start_y = max(center_y - half_sizey, 0)
        return OBFimage[start_x:start_x + self.sizeX_img, start_y:start_y + self.sizeY_img]

    def reconstruct_OBF(self,upsampling = 1):
        """ Main function to reconstruct the OBF image and Fourier domain representation. """
        phase_filters = self.normalize_PCTFs()
        d_Q = self.compute_dQ()
        weightingInv = self.compute_weighting(phase_filters, d_Q)
        OBF_Q = self.compute_OBF_Q(phase_filters, d_Q, weightingInv)
        if upsampling != 0:
            OBF_image = self.align_images(OBF_Q, upsampling)
        else:
            OBF_image = pyfftw.interfaces.numpy_fft.ifft2(pyfftw.interfaces.numpy_fft.ifftshift(np.sum(OBF_Q, axis=0)))
            OBF_image = np.real(OBF_image)

        self.plot(OBF_image, OBF_Q)
        return OBF_image, OBF_Q

    def plot(self, OBF_image, OBF_Q):
        """ Plot the results of the reconstruction in real and Fourier space. """
        reciprocal_res = 1 / self.resolution
        extend_edge = 0.5 * reciprocal_res
        extend = [-extend_edge, extend_edge, -extend_edge, extend_edge]

        summation = np.sum(OBF_Q, axis=0)
        summation[self.sizeX_img // 2, self.sizeY_img // 2] = 0
        dft_amp = np.log(np.abs(summation / self.sizeX_img) + 1)

        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.imshow(dft_amp, extent=extend,  vmax=np.max(dft_amp) * 1.1)
        ax0.set_title('Amplitude of OBF in Fourier domain')
        ax0.set_xlabel('Length (1/Å)')
        ax0.set_ylabel('Length (1/Å)')

        length = [-self.resolution * self.sizeX_img / 2, self.resolution * self.sizeX_img / 2, 
                  -self.resolution * self.sizeY_img / 2, self.resolution * self.sizeY_img / 2]
        ax1.imshow(OBF_image, extent=length, interpolation='gaussian')
        ax1.set_title('Reconstructed OBF')
        ax1.set_xlabel('Length (nm)')
        ax1.set_ylabel('Length (nm)')
        plt.tight_layout()
        plt.show()

 



def DFT_toK(array):
    """
    Foureir transformed the R-space to K-space with a normalization coefficients (1/mn).
    array is cp.ndarray
    """
    M, N = array.shape
    normalizing = 1/(M*N)
    return normalizing * np.fft.fftshift(np.fft.fft2(array))

def invDFT_toR(array):
    return np.fft.ifftshift(np.fft.ifft2(array))


def calculate_CTF(aberrations, parameters, segments, N=512):
    """
    Corrected version of calculate_CTF function.

    https://github.com/Pr4Et/SavvyScan/blob/main/Post_Processing/CTF_defocus_plots.m
    """
       
    lambda_ = parameters["wavelength(nm)"]
    semi_kBF = parameters["semi_convergence angle(rad)"]
    # Convert rad to 1/nm using theta/lambda
    kBF = semi_kBF/lambda_

    k_DPC_min, k_DPC_max = parameters["collection angles(rad)"]
    k_DPC_min *= 0.5 / lambda_
    k_DPC_max *= 0.5 / lambda_
    
    VN = cp.arange(-N//2, N//2, 1)
    Nc = N // 2
    dk = 2*kBF / Nc    # limiting the K range within 2*kBF
    dr = 1 / (dk * N)
    ekx = VN * dk
    eky = VN * dk
    ky, kx = cp.meshgrid(eky, ekx)
    ksquare = kx ** 2 + ky ** 2
    knorm = cp.sqrt(ksquare)
    
    condition1 = tools.create_angle_mask(N, N, angle_range=segments[0])    
    condition2 = tools.create_angle_mask(N, N, angle_range=segments[1])    
    condition3 = tools.create_angle_mask(N, N, angle_range=segments[2])
    condition4 = tools.create_angle_mask(N, N, angle_range=segments[3])

    mask = ((knorm >= k_DPC_min) & (knorm <= k_DPC_max))

    seg_1 = cp.logical_and(mask, cp.array(condition1))
    seg_2 = cp.logical_and(mask, cp.array(condition2))
    seg_3 = cp.logical_and(mask, cp.array(condition3))
    seg_4 = cp.logical_and(mask, cp.array(condition4))
    W_DPC_kx = (seg_1.astype(int) - seg_3.astype(int))*(cp.pi * kBF / 2)
    W_DPC_ky = (seg_2.astype(int) - seg_4.astype(int))*(cp.pi * kBF / 2)  
    
    chi_q = chi_omega(kx, ky, aberrations, lambda_)
    S1 = aperture(kx, ky, semi_kBF, lambda_)
    psi_k = cp.array(S1.astype(int)) * cp.exp(-1j * cp.array(chi_q)) # K-space
    
    psi_in_r = F_r(psi_k, dk)
    psi_in_k = F_k(psi_in_r, dr)
    #CTFicomS=(1/(2*cp.pi))*cp.conj(F_k(psi_in_r*cp.conj(psi_in_r), dr)) # ADF
    CTFSx_sh = -1j * (cp.conj(F_k(psi_in_r * F_r(W_DPC_kx * invF_k(cp.conj(psi_in_r), dr, N), dk), dr)) 
                      - cp.conj(F_k(cp.conj(psi_in_r) * invF_r(W_DPC_kx * F_k(psi_in_r, dr), dk, N), dr)))
    
    CTFSy_sh = -1j * (cp.conj(F_k(psi_in_r * F_r(W_DPC_ky * invF_k(cp.conj(psi_in_r), dr, N), dk), dr)) 
                      - cp.conj(F_k(cp.conj(psi_in_r) * invF_r(W_DPC_ky * F_k(psi_in_r, dr), dk, N), dr)))
    
    CTFCx_sh = -1 * (cp.conj(F_k(psi_in_r * F_r(W_DPC_kx * invF_k(cp.conj(psi_in_r), dr, N), dk), dr)) 
                     + cp.conj(F_k(cp.conj(psi_in_r) * invF_r(W_DPC_kx * F_k(psi_in_r, dr), dk, N), dr)))
    
    CTFCy_sh = -1 * (cp.conj(F_k(psi_in_r * F_r(W_DPC_ky * invF_k(cp.conj(psi_in_r), dr, N), dk), dr)) 
                     + cp.conj(F_k(cp.conj(psi_in_r) * invF_r(W_DPC_ky * F_k(psi_in_r, dr), dk, N), dr)))

    nominator = 2 * 1j * ksquare 
    denominator = cp.reciprocal(nominator)
    denominator[cp.isinf(denominator)] = 0
    denominator[0,0]=1
    qxOperator = kx * denominator
    qyOperator = ky * denominator
    
    CTFiS = (qxOperator * CTFSx_sh + qyOperator * CTFSy_sh)

    CTFiC = (qxOperator * CTFCx_sh + qyOperator * CTFCy_sh)
     
    CTFiS_mean = tools.radialAverage(cp.real(CTFiS), Nc, Nc, Nc)
    CTFiC_mean = tools.radialAverage(cp.real(CTFiC), Nc, Nc, Nc)

    k_profile = (dk * cp.arange(Nc)) / kBF

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    axes[0].plot(cp.asnumpy(k_profile), cp.asnumpy(CTFiS_mean*2*cp.pi))
    axes[0].set_xlabel("k/$k_{BF}$")
    axes[0].set_ylabel("2$\pi$ CTF in sinusoidal component")

    axes[1].plot(cp.asnumpy(k_profile), cp.asnumpy(CTFiC_mean*2*cp.pi))
    axes[1].set_xlabel("k/$k_{BF}$")
    axes[1].set_ylabel("2$\pi$ CTF in cosinusoidal component")
    fig.suptitle('CTF integrated curves')
    plt.show()

    return cp.asnumpy(CTFiS), cp.asnumpy(CTFiC)




class JacobiSolver:
    """
    Numerical integration of the second-order partial differential equation 
    using the Jacobi iterative method.
    """

    def __init__(self, fx, fy, dxy=1.0, tolerance=1e-10, max_iterations=100):
        """
        Initializes the Jacobi solver with the given parameters.

        Parameters
        ----------
        fx : ndarray
            Partial derivative with respect to x.
        fy : ndarray
            Partial derivative with respect to y.
        dxy : float, optional
            Grid spacing, by default 1.0.
        tolerance : float, optional
            Convergence tolerance, by default 1e-10.
        max_iterations : int, optional
            Maximum number of iterations, by default 100.
        """
        self.fx = fx
        self.fy = fy
        self.dxy = dxy
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.p_current = np.zeros_like(fx)
        
        self.convergence_history = []

    def _rotation(self, theta, flip):
        """
        Rotates the coordinate system by a given angle and optionally flips axes.

        Parameters
        ----------
        theta : float
            Angle of rotation in degrees.
        flip : bool
            Whether to flip the axes.

        Returns
        -------
        tuple of ndarray
            Rotated and optionally flipped coordinates (CoMx, CoMy).
        """
        theta_rad = np.radians(theta)
        cos_theta, sin_theta = np.cos(theta_rad), np.sin(theta_rad)

        if not flip:
            CoMx_rot = self.fx * cos_theta - self.fy * sin_theta
            CoMy_rot = self.fx * sin_theta + self.fy * cos_theta
        else:
            CoMx_rot = self.fx * cos_theta + self.fy * sin_theta
            CoMy_rot = self.fx * sin_theta - self.fy * cos_theta
            
        return CoMx_rot, CoMy_rot
    
    def _reconstruct_matrix(self, CoMx, CoMy):
        """
        Reconstructs the matrix from fx and fy using their gradients.

        Returns
        -------
        ndarray
            The reconstructed matrix.
        """
        grad_x = np.gradient(CoMx, self.dxy, axis=0)
        grad_y = np.gradient(CoMy, self.dxy, axis=1)

        # Handle edges for better numerical stability
        grad_y[:, 0] = (CoMy[:, 1] + CoMy[:, 2]) / 2  # Left edge
        grad_y[:, -1] = (CoMy[:, -2] + CoMy[:, -3]) / 2  # Right edge
        grad_x[0, :] = (CoMx[1, :] + CoMx[2, :]) / 2  # Bottom edge
        grad_x[-1, :] = (CoMx[-2, :] + CoMx[-3, :]) / 2  # Top edge

        return grad_x + grad_y

    def _l2_difference(self, f1):
        """
        Computes the L2 norm of the difference between the current solution and the gradients.

        Parameters
        ----------
        f1 : ndarray
            Current solution.

        Returns
        -------
        float
            The L2 norm of the difference.
        """
        difx = np.gradient(f1, self.dxy, axis=0)
        dify = np.gradient(f1, self.dxy, axis=1)
        difference = np.sqrt(np.sum((difx - self.fx) ** 2) + np.sum((dify - self.fy) ** 2)) / f1.size
        return difference

    def solve(self, theta, flip):
        """
        Solves the equation using the Jacobi iterative method.

        Returns
        -------
        ndarray
            The solution matrix.
        """
        nx, ny = self.fx.shape
        p_next = self.p_current.copy()
        iteration = 0
        CoMx, CoMy = self._rotation(theta, flip)
        b = self._reconstruct_matrix(CoMx, CoMy)
        # Initialize progress bar
        with tqdm(total=self.max_iterations, desc="Iteration Progress") as pbar:
            while iteration < self.max_iterations:
                np.copyto(self.p_current, p_next)

                # Update interior points using Jacobi update
                p_next[1:-1, 1:-1] = 0.25 * (
                    self.p_current[:-2, 1:-1] +
                    self.p_current[2:, 1:-1] +
                    self.p_current[1:-1, :-2] +
                    self.p_current[1:-1, 2:] -
                    b[1:-1, 1:-1] * self.dxy**2
                )

                # Compute L2 difference and check convergence
                diff = self._l2_difference(p_next)
                self.convergence_history.append(diff)
                pbar.update(1)

                if iteration > 0 and np.abs(self.convergence_history[-1] - self.convergence_history[-2]) <= self.tolerance:
                    print(f"\nConverged after {iteration} iterations.")
                    break

                iteration += 1

        # Check if convergence was achieved
        if iteration >= self.max_iterations:
            print("\nSolution did not converge within the maximum number of iterations.")
            print(f"Last L2 difference: {diff:.5e}")

        return p_next
        
    @classmethod
    def optimize_rotation(cls, CoMx: np.ndarray, CoMy: np.ndarray, thetas: list, pixel_size_R: float = 0.001, epsilon: float = 1e-3, plot = False):
        """
        Optimize theta and determine if flipping is needed.

        Parameters:
        CoMx, CoMy (np.ndarray): Input coordinate arrays.
        thetas (list): the range of thetas for searching, eg.g np.linspace(0, 360, 360).
        pixel_size_R (float): Step size for the update.
        epsilon (float): Controls the size of high-pass filtering.

        Returns:
        Tuple[float, bool]: Optimal theta and flip status.
        """
        reconstructor = cls(CoMx, CoMy, dxy=pixel_size_R, tolerance=epsilon, max_iterations=100)
        N_thetas = len(thetas)
        stds = np.zeros((2, N_thetas))

        with tqdm(total=N_thetas*2, desc="Calculating", unit="iteration") as pbar:
            for flip in range(2):
                for i, theta in enumerate(thetas):
                    pbar.update(1)                
                    phase = reconstructor.solve(theta, bool(flip))                    
                    stds[flip, i] = np.std(phase)
        
        flip = np.max(stds[1]) > np.max(stds[0])
        theta = thetas[np.argmax(stds[1 if flip else 0])]
        if plot:
            cls._plot_optimization_results(thetas, stds)
        print(f"Does it need to flip: {flip}")
        print(f"The rotation angle is: {theta:.2f} degrees")
        return theta, flip

    @staticmethod
    def _plot_optimization_results(thetas: np.ndarray, stds: np.ndarray):
        """Plot the optimization results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
        x = thetas
        ax1.scatter(x, stds[0], color='#2393CD', label='No flip')
        ax2.scatter(x, stds[1], color='#003561', label='Flip')
        ax1.legend(loc='upper right', shadow=True, fontsize=14)
        ax2.legend(loc='upper right', shadow=True, fontsize=14)
        ax1.set_xlabel('Degrees (deg)', fontsize=18)
        ax1.set_ylabel('Standard Deviation', fontsize=18)
        ax2.set_xlabel('Degrees (deg)', fontsize=18)
        ax2.set_ylabel('Standard Deviation', fontsize=18)
        plt.show()
        
    def plot_convergence(self):
        """
        Plots the convergence history.
        """
        if not self.convergence_history:
            print("No convergence history to plot.")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(self.convergence_history, label="L2 Difference")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("L2 Difference (Log Scale)")
        plt.title("Convergence History")
        plt.legend()
        plt.grid(True)
        plt.show()


class Conjugate_gradient_iDPC:
    """
    (1) DS Watkins. Fondamentals of matrix computations - third edition. 2010.

    (2) J Shewchuk. An Introduction to the Conjugate Gradient Method Without the Agonizing Pain. 1994.

    Usage:
        solver = Conjugate_gradient_iDPC(CoMx, CoMy, dxy=1, tolerance=1e-5, max_iterations=1000)
        th, fl = solver.optimize_rotation(CoMx, CoMy, thetas = np.arange(-90, 90, 1), epsilon = 1e-5, plot=False)
        phase = solver.solve(th, fl)
    """

    def __init__(self, fx, fy, dxy=1.0, tolerance=1e-10, max_iterations=100):
        """
        Initializes the Jacobi solver with the given parameters.

        Parameters
        ----------
        fx : ndarray
            Partial derivative with respect to x.
        fy : ndarray
            Partial derivative with respect to y.
        dxy : float, optional
            Grid spacing, by default 1.0.
        tolerance : float, optional
            Convergence tolerance, by default 1e-10.
        max_iterations : int, optional
            Maximum number of iterations, by default 100.
        """
        self.fx = fx
        self.fy = fy
        self.dxy = dxy
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.p_current = np.zeros_like(fx)
        
        self.convergence_history = []

    def _rotation(self, theta, flip):
        """
        Rotates the coordinate system by a given angle and optionally flips axes.

        Parameters
        ----------
        theta : float
            Angle of rotation in degrees.
        flip : bool
            Whether to flip the axes.

        Returns
        -------
        tuple of ndarray
            Rotated and optionally flipped coordinates (CoMx, CoMy).
        """
        theta_rad = np.radians(theta)
        cos_theta, sin_theta = np.cos(theta_rad), np.sin(theta_rad)

        if not flip:
            CoMx_rot = self.fx * cos_theta - self.fy * sin_theta
            CoMy_rot = self.fx * sin_theta + self.fy * cos_theta
        else:
            CoMx_rot = self.fx * cos_theta + self.fy * sin_theta
            CoMy_rot = self.fx * sin_theta - self.fy * cos_theta
            
        return CoMx_rot, CoMy_rot
    
    def _reconstruct_matrix(self, CoMx, CoMy):
        """
        Reconstructs the matrix from fx and fy using their gradients.

        Returns
        -------
        ndarray
            The reconstructed matrix.
        """
        grad_x = np.gradient(CoMx, self.dxy, axis=0)
        grad_y = np.gradient(CoMy, self.dxy, axis=1)

        # Handle edges for better numerical stability
        grad_y[:, 0] = (CoMy[:, 1] + CoMy[:, 2]) / (2*self.dxy)  # Left edge
        grad_y[:, -1] = (CoMy[:, -2] + CoMy[:, -3]) / (2*self.dxy)  # Right edge
        grad_x[0, :] = (CoMx[1, :] + CoMx[2, :]) / (2*self.dxy)  # Bottom edge
        grad_x[-1, :] = (CoMx[-2, :] + CoMx[-3, :]) / (2*self.dxy)  # Top edge

        return grad_x + grad_y

    def A(self, v, dxy):
        """
        Computes the action of (-) the Poisson operator on any
        vector v_{ij} for the interior grid nodes
    
        Parameters
        ----------
        v : numpy.ndarray
            input vector
        dx : float
             grid spacing in the x direction
        dy : float
            grid spacing in the y direction
        

        Returns
        -------
        Av : numpy.ndarray
            action of A on v
        """
    
        Av = -((v[:-2, 1:-1]-2.0*v[1:-1, 1:-1]+v[2:, 1:-1])/dxy**2 
           + (v[1:-1, :-2]-2.0*v[1:-1,1:-1]+v[1:-1, 2:])/dxy**2)
    
        return Av
    

    def solve(self, theta, flip):
        """
        Solves the equation using the Jacobi iterative method.

        Returns
        -------
        ndarray
            The solution matrix.
        """
        nx, ny = self.fx.shape
        size = nx * ny
        p = self.p_current.copy()
        r = np.zeros((nx, ny))
        Ad = np.zeros((nx, ny))
        iteration = 0
        CoMx, CoMy = self._rotation(theta, flip)
        b = self._reconstruct_matrix(CoMx, CoMy)
        # Initial residual r0 and initial search direction d0
        r[1:-1, 1:-1] = -b[1:-1, 1:-1] - self.A(p, self.dxy)
        d = r.copy()
        # Initialize progress bar
        with tqdm(total=self.max_iterations, desc="Iteration Progress") as pbar:
            while iteration < self.max_iterations:
                # Laplacian of the search direction.
                Ad[1:-1, 1:-1] = self.A(d, self.dxy)
                # Magnitude of jump.
                alpha = np.sum(r*r) / np.sum(d*Ad)
                # Iterated solution
                pnew = p + alpha*d
                # Intermediate computation
                beta_denom = np.sum(r*r)
                # Update the residual.
                r = r - alpha*Ad
                # Compute beta
                beta = np.sum(r*r) / beta_denom
                # Update the search direction.
                d = r + beta*d

                # Compute L2 difference and check convergence
                difx = np.gradient(pnew, self.dxy, axis=0)
                dify = np.gradient(pnew, self.dxy, axis=1)
                diff = np.sqrt(np.sum((difx - CoMx) ** 2) + np.sum((dify - CoMy) ** 2)) / size
                self.convergence_history.append(diff)
                pbar.update(1)

                if iteration > 0 and np.abs(self.convergence_history[-1] - self.convergence_history[-2]) <= self.tolerance:
                    print(f"\nConverged after {iteration} iterations.")
                    break

                iteration += 1
                np.copyto(p, pnew)
        # Check if convergence was achieved
        if iteration >= self.max_iterations:
            print("\nSolution did not converge within the maximum number of iterations.")
            print(f"Last L2 difference: {diff:.5e}")

        return pnew
        
    @classmethod
    def optimize_rotation(cls, CoMx: np.ndarray, CoMy: np.ndarray, thetas: list, dxy: float = 0.001, tolerance: float = 1e-3, plot = False):
        """
        Optimize theta and determine if flipping is needed.

        Parameters:
        CoMx, CoMy (np.ndarray): Input coordinate arrays.
        thetas (list): the range of thetas for searching, eg.g np.linspace(0, 360, 360).
        dxy (float): pixel size in real.
        tolerance (float): Controls the convergence of iterations.

        Returns:
        Tuple[float, bool]: Optimal theta and flip status.
        """
        reconstructor = cls(CoMx, CoMy, dxy=dxy, tolerance=tolerance, max_iterations=100)
        N_thetas = len(thetas)
        stds = np.zeros((2, N_thetas))

        with tqdm(total=N_thetas*2, desc="Calculating", unit="iteration") as pbar:
            for flip in range(2):
                for i, theta in enumerate(thetas):
                    pbar.update(1)                
                    phase = reconstructor.solve(theta, bool(flip))                    
                    stds[flip, i] = np.std(phase)
        
        flip = np.max(stds[1]) > np.max(stds[0])
        theta = thetas[np.argmax(stds[1 if flip else 0])]
        if plot:
            cls._plot_optimization_results(thetas, stds)
        print(f"Does it need to flip: {flip}")
        print(f"The rotation angle is: {theta:.2f} degrees")
        return theta, flip

    @staticmethod
    def _plot_optimization_results(thetas: np.ndarray, stds: np.ndarray):
        """Plot the optimization results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), layout='constrained')
        x = thetas
        ax1.scatter(x, stds[0], color='#2393CD', label='No flip')
        ax2.scatter(x, stds[1], color='#003561', label='Flip')
        ax1.legend(loc='upper right', shadow=True, fontsize=14)
        ax2.legend(loc='upper right', shadow=True, fontsize=14)
        ax1.set_xlabel('Degrees (deg)', fontsize=18)
        ax1.set_ylabel('Standard Deviation', fontsize=18)
        ax2.set_xlabel('Degrees (deg)', fontsize=18)
        ax2.set_ylabel('Standard Deviation', fontsize=18)
        plt.show()
        
    def plot_convergence(self):
        """
        Plots the convergence history.
        """
        if not self.convergence_history:
            print("No convergence history to plot.")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(self.convergence_history, label="L2 Difference")
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("L2 Difference (Log Scale)")
        plt.title("Convergence History")
        plt.legend()
        plt.grid(True)
        plt.show()
