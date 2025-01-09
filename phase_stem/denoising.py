import numpy as np
import time
from skimage.restoration import denoise_tv_bregman, calibrate_denoiser, cycle_spin, denoise_tv_chambolle
from tqdm.notebook import tqdm
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
from phase_STEM import tools

def poisson_noise(image, percent_noise=5.0, seed=None):
    """
    Add Poisson noise to an image with controlled noise level.
    The Poisson distribution has a mean of lamda=
        area per pixel * dose per area * intensity
        where it is assumed that intensity of the reciprocal space probe is normalized to integrate to 1.
    Parameters:
    image (numpy.ndarray): The input image.
    percent_noise (float): The percentage of noise to add.
    seed (int, optional): The random seed for reproducibility.

    Returns:
    numpy.ndarray: The noisy image.
    """
    np.random.seed(seed)
    noisy_image = image.copy()
    
    unique_vals = np.unique(image)
    
    for val in unique_vals:
        idx = image == val
        count = np.sum(idx)
        
        if count > 0:
            current_mean = val
            # Generate Poisson noise
            noisy_values = poisson.rvs(current_mean, size=count)
            
            # Adjust the standard deviation to the required level
            new_mean = np.sqrt(current_mean) / (percent_noise / 100.0)
            noisy_values = noisy_values + (new_mean - current_mean)
            
            # Adjust back to starting mean, but with noise added
            noisy_values = noisy_values * (current_mean / new_mean) if new_mean !=0 else noisy_values * current_mean
            
            noisy_image[idx] = noisy_values
    
    return noisy_image

def add_poisson_noises(image, electron_dose, pixelsize, seed=None):
    """
    Add Poisson noise to an image array.
    
    Parameters:
        image (np.ndarray): Input image array with shape:
                            - (px, py): Single 2D image
                            - (N, px, py): Stack of 2D images
                            - (N, M, px, py): 4D image stacks
        electron_dose (float): Electron dose in e/Å^2.
        pixelsize (float): Pixel size in Å/px.
        seed (int, optional): Random seed for reproducibility.
    
    Returns:
        np.ndarray: Noisy image array with the same shape as the input.
    """
    
    def _apply_noise(array, dose, pixel_size, rng):
        """Applies Poisson noise to a single array."""
        total_dose = array.size * dose * pixel_size**2
        normalized_array = array / np.sum(array)  # Normalize the array
        noisy_array = rng.poisson(normalized_array * total_dose)
        return noisy_array.astype(np.float64)
    
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    # Set up the random number generator
    rng = np.random.default_rng(seed)
    
    # Process the array based on its dimensions
    noisy_image = np.zeros_like(image, dtype=np.float64)
    
    def process_slice(slice_):
        """Processes a single slice, handling positive and negative values."""
        if np.min(slice_) < 0:
            # Separate positive and negative parts
            positive_part = np.where(slice_ > 0, slice_, 0)
            negative_part = np.where(slice_ < 0, -slice_, 0)
            
            # Add noise separately for positive and negative parts
            noisy = _apply_noise(positive_part, electron_dose, pixelsize, rng)
            noisy -= _apply_noise(negative_part, electron_dose, pixelsize, rng)
        else:
            noisy = _apply_noise(slice_, electron_dose, pixelsize, rng)
        return noisy
    
    # Process based on dimensions
    if image.ndim == 2:
        noisy_image = process_slice(image)
    elif image.ndim == 3:
        for i in range(image.shape[0]):
            noisy_image[i] = process_slice(image[i])
    elif image.ndim == 4:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                noisy_image[i, j] = process_slice(image[i, j])
    else:
        raise ValueError("Input array must have 2, 3, or 4 dimensions.")
    
    return noisy_image


def add_noises(original, noises, mode, size = (128,128), color='viridis', plot=False):
    """
    It provides to add noises to original images or noisy images.
    Gaussian and Poisson distributions can be chosen to generate noises for S/TEM images.
    Reference of Method:
        R. Ishikawa, S.J. Pennycook, A.R. Lupini, S.D. Findlay, N. Shibata, Y. Ikuhara,
        Appl. Phys. Lett. 109 (2016) 163102, http://dx.doi.org/10.1063/1.4965709.

    Method descriptions:
        The number of detected electrons deviates from that predicted by a noise-free simulation, 
        which the fluctuations can be modeled by Poisson's statistics, where the probability of detecting 
        k electrons is determined by the number of expected electrons ν as: p(k, v) = v^k*exp(-v)/k!.
        Each probe position (or pixel) is an independent event, and we simulate the intensity fluctuation by the following process: 
            (1) multiply the number of incident electrons per pixel (Dp: pixel dose) by the simulated fractional STEM image (normal to the incident beam) 
            (2) read out each pixel value as the expected value ν and create a noisy image using Poisson's random values generated by a Monte Carlo method.
    Args:
        original: ndarray, the raw image wanted to be noised. If it is not provided, then only noise will be return.
        noises: float, the total noises added to original image
        mode: string, 'Gaussian' or 'Poisson' or 'PoissonBYGaussian'
        size: the image size, 
        color: string, the cmap for showing images
    Return:
        noisy, noised_image: the noises and noised_image will be output
    """
    if original is None:
        original = np.zeros(size, dtype=np.float32)
        print('No original image is given, an empty image will be insteaded')
    if noises is None:
        dose_rate = 1
    else: dose_rate = noises
    if mode is None:
        mode = 'Poisson'
    # Step 1: Multiply the pixel dose by the fractional image to get the expected number of electrons per pixel
    if np.min(original) < 0:
        image = original - np.min(original)
    else: image = original
    image *= dose_rate 
    image /= np.linalg.norm(image)
    if mode =='Poisson':  
        # Step 2: Generate noisy image using Poisson distribution
        noised_image = poisson.rvs(image)
        noisy = noised_image - image
        noisy[noisy<0] = 0
    elif mode=='Gaussian':
        noisy = norm.rvs(scale=np.sqrt(image), size=size)   # noises
        noised_image = noisy + image
    elif mode=='PoissonBYGaussian':
        noisy = norm.rvs(loc=image, scale=np.sqrt(image), size=size)   # noises, Normal approximation of Poisson
        noised_image = noisy + image
    else: 
        print('ERROR: Please choose the noise-distribution: <Poisson> , <PoissonBYGaussian>, or <Gaussian>')
        
    
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        im = axs[0].imshow(image, cmap=color, vmin=0)  # Assuming grayscale data
        fig.colorbar(im, ax=axs[0], label='Intensity')  
        axs[0].set_title('Original Image')
        axs[0].axis('off')  

        im = axs[1].imshow(noisy, cmap=color, vmin=0)  
        fig.colorbar(im, ax=axs[1], label='Intensity')  
        axs[1].set_title('Noises by '+mode)
        axs[1].axis('off')

        im = axs[2].imshow(noised_image, cmap=color, vmin =0)  
        fig.colorbar(im, ax=axs[2], label='Intensity')  
        axs[2].set_title('Noised Image')
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()

    return noisy, noised_image

def tv_denoiser(phase: np.ndarray, weights: dict | float, max_iter: int=100, method: str = 'chambolle', shift: int=6) -> np.ndarray:
    """
    Apply total variation denoising to an image.

    Parameters:
    phase (np.ndarray): Input image to be denoised.
    weights (dict or float): If dict, should contain parameters for the denoising function, e.g. {'weight':np.arange(1,10,1)}.
                            If float, it's the denoising weight.
    max_iter (int): Maximum number of iterations.
    method (str): Denoising method, either 'chambolle' or 'bregman'.
    shift: int, working when weights is float
    Returns:
    np.ndarray: Denoised image.
    
    Raises:
    ValueError: If 'weight' is not a dict or float.
    """
    if isinstance(weights, dict):    
        # Ensure all values in the dictionary are iterable
        weights = {k: v if isinstance(v, (list, np.ndarray)) else [v] for k, v in weights.items()}
        weights['eps'] = [1e-6]
        weights['max_num_iter'] = [int(max_iter)]
        
        denoise_func = denoise_tv_chambolle if method == 'chambolle' else denoise_tv_bregman
        denoising_function = calibrate_denoiser(phase, denoise_func, denoise_parameters=weights)
        denoised_phase = denoising_function(phase)
        
    elif isinstance(weights, float):
        denoise_kwargs = {'weight':weights, 'eps':1e-6}
        denoise_func = denoise_tv_chambolle if method == 'chambolle' else denoise_tv_bregman
        denoised_phase = cycle_spin(phase, func=denoise_func, max_shifts=shift, func_kw=denoise_kwargs)
    else:
        raise ValueError("Invalid 'weights' parameter. It should be a float or a dict.")
    
    return denoised_phase

class GMRF_Denoising:
    """
    reference:
        Yasuda, M., Watanabe, J., Kataoka, S. & Tanaka, K. (2017) “Linear-Time Algorithm in Bayesian Image Denoising based on Gaussian Markov Random Field”, 
        arxiv.org: https://arxiv.org/abs/1710.07393

    structure of class:
        __init__: Initializes the class with data paths and sets up basic parameters.
       initialize: Sets up the graph structure and initializes variables.
        _create_graph: Helper method to create the graph structure.
        _vectorize_images: Helper method to vectorize the input images.
       denoise: The main denoising algorithm.
       plot_results: Plots the denoised, and comparison images.

    To use this class, you would create an instance of BayesianImageDenoising, 
    initialize it, run the denoising algorithm, evaluate the results, and then print or plot them as needed.

    This object-oriented approach makes the code more modular and easier to maintain. 
    It also allows for easier extension or modification of the algorithm in the future.
"""
    def __init__(self, noised_images, iter_step=None, initial_para=None):
        if initial_para is None:
            initial_para = {
                'b': 0, 'alpha': 1e-4, 'lambda': 1e-7, 'sigma2':2000, 'epsilon': 1e-3,
                'max_iterations': 10
            }
        if iter_step is None:
            iter_step = {
                'b': 1e-9, 'alpha': 1e-9, 
                'lambda': 1e-13, 'sigma2': 2000,
                }
        self.noised_images = noised_images
        self.averaged_image = np.mean(self.noised_images, axis=0)
        self.K, self.ny, self.nx = self.noised_images.shape
        self.n_total = self.ny * self.nx
        self.ini_b = initial_para['b']
        self.ini_alpha = initial_para['alpha']
        self.ini_lambda = initial_para['lambda']
        self.ini_sigma2 = initial_para['sigma2']
        self.epsilon = initial_para['epsilon']
        self.max_iterations = initial_para['max_iterations']
        self.iter_b = iter_step['b']
        self.iter_alpha = iter_step['alpha']
        self.iter_lambda = iter_step['lambda']
        self.iter_sigma2 = iter_step['sigma2']
        
    def initialize(self):
        self.degree, self.neighbor, self.psi = self._create_graph()
        self.degree_vec = self.degree.T.flatten()
        self.Phi = self.psi.T.flatten()
        self.Y = self._vectorize_images()
        self.Theta = np.array([self.ini_b, self.ini_alpha, self.ini_lambda, self.ini_sigma2])
        self.y_hat = self.Y.T.mean(axis=0)
        self.m_post = self.y_hat.copy()

    def _create_graph(self):
        degree = np.zeros((self.ny, self.nx))
        neighbor = np.zeros((self.n_total, 8), dtype=int)
        psi = np.zeros((self.ny, self.nx))

        for i in range(self.ny):
            for j in range(self.nx):
                # Determine the degree based on the position in the grid
                if i > 0 and i < self.ny-1 and j > 0 and j < self.nx-1:
                    degree[i, j] = 8  # Internal node
                elif (i == 0 and j == 0) or (i == self.ny-1 and j == 0) or (i == 0 and j == self.nx-1) or (i == self.ny-1 and j == self.nx-1):
                    degree[i, j] = 3  # Corner nodes
                else:
                    degree[i, j] = 5  # Edge nodes
        
                vector_id = self.nx * i + j
        
                # Calculate neighbor indices
                id_up = vector_id - self.nx
                id_down = vector_id + self.nx
                id_left = vector_id - 1
                id_right = vector_id + 1
                id_top_left = vector_id - self.nx - 1
                id_top_right = vector_id - self.nx + 1
                id_bottom_left = vector_id + self.nx - 1
                id_bottom_right = vector_id + self.nx + 1
        
                # Assign neighbors if within bounds
                if i-1 >= 0:
                    neighbor[vector_id, 0] = id_up
                if i+1 < self.ny:
                    neighbor[vector_id, 1] = id_down
                if j-1 >= 0:
                    neighbor[vector_id, 2] = id_left
                if j+1 < self.nx:
                    neighbor[vector_id, 3] = id_right
                if i-1 >= 0 and j-1 >= 0:
                    neighbor[vector_id, 4] = id_top_left
                if i-1 >= 0 and j+1 < self.nx:
                    neighbor[vector_id, 5] = id_top_right
                if i+1 < self.ny and j-1 >= 0:
                    neighbor[vector_id, 6] = id_bottom_left
                if i+1 < self.ny and j+1 < self.nx:
                    neighbor[vector_id, 7] = id_bottom_right
        
                # Calculate psi value
                psi[i, j] = 4 * np.sin(np.pi * (i) / (2 * self.ny))**2 + 4 * np.sin(np.pi * (j) / (2 * self.nx))**2

        return degree, neighbor, psi

    def _vectorize_images(self):
        return self.noised_images.reshape(self.K, self.n_total).T

    def denoise(self):
        starting = time.time()

        Theta_old = self.Theta.copy()
        b_old, lambda_old, alpha_old, sigma_2_old = Theta_old
        Theta = Theta_old
        m_post_old = self.y_hat
        m_post = m_post_old
        
        for _ in range(self.max_iterations):                                
            Theta_old = self.Theta
            b_old, lambda_old, alpha_old, sigma_2_old = Theta_old
            sigma_2, dQ_lambda, m_edge = 0, 0, 0
            Q_alpha1 = 0
            dQ_alpha2 = 0
            #Update the mean-vector of posterior distributions (m_post)
            for i in range(self.n_total):
                m_neighbor = 0
                for j in range(8):
                    if self.neighbor[i, j] != 0:
                        m_neighbor += m_post_old[self.neighbor[i, j]]  

                denominator = lambda_old + alpha_old * self.degree_vec[i] + self.K / sigma_2_old
                m_post[i] = (b_old + self.K / sigma_2_old * self.y_hat[i] + alpha_old * m_neighbor) / denominator

                sigma_2 += 1 / self.n_total * 1 / (lambda_old + self.K/sigma_2_old+ alpha_old * self.Phi[i])
                dQ_lambda += 0.5 * (1 / (lambda_old + alpha_old * self.Phi[i]) - 1 / denominator)
            
            sigma_2 += np.sum(np.linalg.norm(m_post[:, np.newaxis] - self.Y, axis=0)**2) / (self.n_total * self.K)
            
            #update b
            dQ_b = (self.n_total * b_old + self.n_total * self.K / sigma_2_old * self.y_hat.mean()) / (lambda_old + self.K / sigma_2_old) 
            dQ_b += - self.n_total * b_old / lambda_old
            b = b_old + self.iter_b / self.n_total * dQ_b
            
            #update lambda
            dQ_lambda -= 0.5 * np.linalg.norm(m_post)**2 - b_old**2 / (2 * lambda_old**2)            
            lambda_ = lambda_old + self.iter_lambda / self.n_total * dQ_lambda #control the step size
            
            #update alpha
            for i in range(self.n_total):
                valid_neighbors = self.neighbor[i, self.neighbor[i] != 0]
                m_edge += np.sum((m_post[valid_neighbors] - m_post[i])**2)
            m_edge /= 2

            dQ_alpha1 = np.sum(self.Phi / (lambda_old + self.K / sigma_2_old + alpha_old * self.Phi)) + m_edge
            alpha = (self.n_total - 1) / dQ_alpha1
            #update alpha2
            dQ_alpha2 += np.sum(self.Phi / (lambda_ + alpha * self.Phi))
            dQ_alpha = - 0.5 * dQ_alpha1 + 0.5 * dQ_alpha2 - 0.5 * m_edge
            alpha +=  self.iter_alpha / self.n_total * dQ_alpha

            self.Theta = np.array([b, lambda_, alpha, sigma_2])

            if np.max(np.abs(self.Theta - Theta_old)) < self.epsilon:
                print('The error now is %f.\n',np.max(np.abs(self.Theta-Theta_old)))
                break


        self.restored_image = m_post.reshape((self.ny, self.nx))
        self.reconstructed_image = self.restored_image + np.mean(self.averaged_image)

        spendTime = time.time()-starting
        if spendTime>1:
            print(f"The whole procedure takes {round(spendTime,1)} seconds.")
        elif spendTime>0.001 and spendTime<1:
            print(f"The whole procedure takes {round(spendTime*1000)} milliseconds.")
        else:
            print(f"The whole procedure takes {round(spendTime*10**6)} microseconds.")

    def plot(self):
        print(f"The input image is denoised using hyper-parameters: ")
        print(f"    b = {self.Theta[0]} for controlling the brightness")
        print(f"    \u03BB = {self.Theta[1]} for controlling the variance of the intensities of pixels")
        print(f"    \u03B1 = {self.Theta[2]} for controlling the smoothness of the image")
        print(f"    \u03C3^2 = {self.Theta[3]} is the variance of the white Gaussian noises in images")

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        im = axs[0].imshow(self.restored_image, cmap='viridis')  # Assuming grayscale data
        fig.colorbar(im, ax=axs[0], label='Intensity')  
        axs[0].set_title('Restored Image')
        axs[0].axis('off')  

        im = axs[1].imshow(self.reconstructed_image, cmap='viridis')  
        fig.colorbar(im, ax=axs[1], label='Intensity')  
        axs[1].set_title('Reconstructed Image (aligning intensity)')
        axs[1].axis('off')

        im = axs[2].imshow(self.averaged_image, cmap='viridis')  
        fig.colorbar(im, ax=axs[2], label='Intensity')  
        axs[2].set_title(f'Average ({self.K} images)')
        axs[2].axis('off')
        plt.tight_layout()
        plt.show()


# it seems succeed

class DCT_GMRF_Denoiser:
    """
     DCT_GMRF_Denoiser: implements a Bayesian image denoising technique using the Discrete Cosine Transform (DCT) 
                        and Fast Fourier Transform (FFT) methods, which assumes the noise follows a Gaussian distribution. 

    If the image stacks exist drifting, the image alignment should be done before denoising.

    Purpose: 
        This class aims to denoise a set of degraded images using Bayesian inference and the DCT-FFT method.
    Parameters:
        iter_step: Dictionary containing step sizes for hyperparameter updates.
        initial_para: Dictionary containing initial values for the hyperparameters.
    """
    def __init__(self, iter_step = None, initial_para=None):
        if initial_para is None:
            initial_para = {
                'b': 0, 'alpha': 1e-4, 'lambda': 1e-7, 'sigma2': 4000, 
                'max_iterations': 10,
                'epsilon':1e-3
            }
        if iter_step is None:
            iter_step = {
                'b': 1e-9, 'alpha': 1e-9, 
                'lambda': 1e-13, 'sigma2': 2000,
                'step_length':2
                }
        self.ita_b = iter_step['b']
        self.ita_alpha = iter_step['alpha']
        self.ita_lambda = iter_step['lambda']
        self.ita_sigma2 = iter_step['sigma2']
        self.step_length = iter_step['step_length']
        self.max_iter = initial_para['max_iterations']
        self._lambda = initial_para['lambda']
        self.alpha = initial_para['alpha']
        self.b = initial_para['b']
        self.sigma2 = initial_para['sigma2']
        self.epsilon = initial_para['epsilon']
    def denoise(self, y):
        """
        Input: 
            y, a numpy array of shape (K, v, v) representing K degraded images.
        Process:
            Compute the mean of the degraded images and apply DCT.
            Initialize hyperparameters.
            Iteratively update hyperparameters and compute the Maximum A Posteriori (MAP) estimate until convergence or maximum iterations.
            
        Output: 
        The denoised image, final hyperparameters, and the energy of the iterations.
        """
        K, v, u = y.shape
        y_hat = np.mean(y, axis=0).flatten()
        energy_ita = []
        # Compute DCT of y_hat
        z = dct(dct(y_hat.reshape(v, u), axis=0, norm='ortho'), axis=1, norm='ortho').flatten()
        # Initialize hyperparameters
        theta = {'sigma2': self.sigma2, 'b': self.b, 'lambda': self._lambda, 'alpha': self.alpha}
        
        step = {'b':self.ita_b, 'alpha':self.ita_alpha, 'lambda':self.ita_lambda, 'sigma2':self.ita_sigma2}

        for _ in range(self.max_iter):            
            old_theta = np.array([theta['sigma2'], theta['b'], theta['lambda'], theta['alpha']])
            m = self._compute_map_estimate(y_hat, K, v, u, theta)
            theta = self._update_hyperparameters(y, z, m, K, v, u, theta, step)
            new_theta = np.array([theta['sigma2'], theta['b'], theta['lambda'], theta['alpha']])
            updated_energy = self._calculate_energy(y, m, theta)
            energy_ita.append(updated_energy)
            if np.max(np.abs(new_theta - old_theta)) < self.epsilon:
                break

        print('\u03C3^2 =', theta['sigma2'], 'is the variance of the white Gaussian noise')
        print('b =', theta['b'], 'controls the brightness of the image')
        print('\u03BB =', theta['lambda'], 'controls the variance of the intensities of pixels')
        print('\u03B1 =', theta['alpha'], 'controls the smoothness of the image')

        return m.reshape(v, u), theta, energy_ita

    def _calculate_energy(self, y, m, theta):
        K, v, u = y.shape
        n = v * u
        m_2d = m.reshape(v, u) # m is updated from y-hat

        # Compute necessary variables for gradients
        Phi = self._compute_Phi(v, u)
        ave = np.mean(y)
        
        lambda_, alpha, sigma2, b = theta['lambda'], theta['alpha'], theta['sigma2'], theta['b']

        # Compute the terms from the Fpost equation (free energy), enabling it to be maximum
        delta = np.zeros(n)
        delta[1] = 1  # Set the element at index 1 to 1
        term0 = (lambda_ + K / sigma2 + alpha * Phi)
        term1 = -1 / (2 * sigma2) * np.sum([np.linalg.norm(y[k] - m_2d)**2 for k in range(K)])
        #z is the DCT of the average image of K degraded images y_hat
        z_i = dct(dct(m_2d, axis=0, norm='ortho'), axis=1, norm='ortho').flatten()
        term2 = -0.5 * np.sum((np.sqrt(n) * b * delta + (K / sigma2) * z_i)**2 / term0)

        term3 = 0.5 * np.sum(np.log(np.clip(term0, a_min=1e-20, a_max=None)))
        term4 = -n / 2 * np.log(2 * np.pi)
        posteriori = -(term1 + term2 + term3 + term4)
        return np.sign(posteriori) * np.log1p(np.abs(posteriori))
    
    def _compute_map_estimate(self, y_hat, K, v, u, theta):
        """Compute MAP estimate using DCT"""
        c = theta['b'] + (K / theta['sigma2']) * y_hat

        # Compute DCT of c
        c_dct = dct(dct(c.reshape(v, u), axis=0, norm='ortho'), axis=1, norm='ortho').flatten()

        # Compute eigenvalues of the precision matrix
        Phi = self._compute_Phi(v, u)
        eigenvalues = theta['lambda'] + (K / theta['sigma2']) + theta['alpha'] * Phi

        # Compute MAP estimate in DCT domain
        m_dct = c_dct / eigenvalues 

        # Inverse DCT to get MAP estimate
        m = idct(idct(m_dct.reshape(v, u), axis=1, norm='ortho'), axis=0, norm='ortho').flatten()
        return m

    def _compute_Phi(self, v, u):
        """Compute the Phi matrix using a more efficient algorithm"""
        i, j = np.meshgrid(np.arange(v), np.arange(u), indexing='ij')
        Phi = 4 * (np.sin(np.pi * i / (2 * v))**2 + np.sin(np.pi * j / (2 * u))**2)
        return Phi.flatten()

    def _update_hyperparameters(self, y, z, m, K, v, u, theta, step):
        """Update hyperparameters using L-BFGS-B optimization"""
        n = v * u
        m_2d = m.reshape(v, u)
   
        # Compute necessary variables for gradients
        Phi = self._compute_Phi(v, u)
        ave = np.mean(y)
        delta = np.zeros(n)
        delta[1] = 1  # Set the element at index 1 to 1
        b_term = np.sqrt(n) * theta['b'] * (delta) + (K / theta['sigma2']) * z
        denominator = (theta['lambda'] + K / theta['sigma2'] + theta['alpha'] * Phi)

        grad_lambda = 0.5 * np.sum((b_term / denominator)**2 + 1 / denominator)
        grad_alpha = 0.5 * np.sum((b_term / denominator)**2 * Phi + Phi / denominator)
        grad_sigma2 = -0.5 * np.sum([np.linalg.norm(y[k])**2 for k in range(K)]) / (theta['sigma2']**2) + \
                      (K / theta['sigma2']**2) * np.sum(b_term / denominator - 0.5*(b_term)**2 / denominator**2 - 0.5 / denominator)
        grad_b = -(n * theta['b'] + n * (K / theta['sigma2']) * ave) / (theta['lambda'] + K / theta['sigma2'])
        #update the thera using the gradients, which determine the ascents
        update_sigma2 = theta['sigma2'] + step['sigma2']  *grad_sigma2
        if update_sigma2 <0: #sigma2 is larger than 0
            update_sigma2 = theta['sigma2'] + 0.5*(step['sigma2']  *grad_sigma2 - update_sigma2)
        update_b = theta['b'] + step['b']  *grad_b
        if update_b <0:
            update_b = theta['b'] + 0.5*(step['b']  *grad_b - update_b)
        update_lambda = theta['lambda'] + step['lambda'] *grad_lambda
        if update_lambda<0:
            update_lambda = theta['lambda'] + 0.5*(step['lambda'] *grad_lambda - update_lambda)
        update_alpha = theta['alpha'] + step['alpha']  *grad_alpha
        if update_alpha<0:
            update_alpha = theta['alpha'] + 0.5*(step['alpha']  *grad_alpha - update_alpha)
            
        new_theta = {
            'b': update_b,
            'lambda': update_lambda,
            'sigma2': update_sigma2,
            'alpha': update_alpha
        }
        return new_theta



