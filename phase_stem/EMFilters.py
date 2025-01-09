import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifftshift, ifft2, fftfreq
import pandas as pd
from phase_STEM import tools
from scipy.ndimage import gaussian_filter, median_filter
from phase_STEM import tools, analysis
import cupy as cp

def cartesian_to_polar(image_shape, center):
    """
    Convert the cartesian coordinate of an image into the polar coordinate.
    """
    sizex, sizey = image_shape
    y, x = np.ogrid[:sizey, :sizex]

    x = x - center[0]
    y = y - center[1]
    rho = np.hypot(y, x) 
  
    return rho 


def frequency_pass_filter(image, space ='real', freq_low=0.001, freq_high=1, inverse = True):
    """
    Do filtering an image with a band-pass filter.
    can be deleted due to the same as circle_mask
    Args:
    image: np.ndarray or cp.ndarray
    space: string, 'real' or 'reciprocal'
    freq_low, freq_high: float, the ratio of the maximum of frequency, which freq_high=1 represents there is no high requency cut off, 
                                freq_low = 0 indicates no cut off on the low frequency domain
    inverse: True, it will return an image with np.array
             False, it will return an FFT with frequency cut off, using np.array
    return:
    np.ndarray(2D), with complex
    """
   
    if isinstance(image, cp.ndarray):
        img = cp.asnumpy(image)
    else: img = np.asarray(image)
    if space=='real':
        try: 
            _, spectrum = fftshift(analysis.periodic_DFT(img, inverse_dft=False))
        except: spectrum = fftshift(fft2(img))
    else: spectrum = img
    # Create a frequency domain array
    sizeX, sizeY = image.shape
    
    freq_u = fftshift(fftfreq(sizeX))
    freq_v = fftshift(fftfreq(sizeY))
    freq_u, freq_v = np.meshgrid(freq_u, freq_v)
    freq = np.sqrt(freq_u**2 + freq_v**2)
    # build a bandpass filter
    freq_high = np.max(freq) * freq_high
    freq_low  = np.max(freq) * freq_low

    spectrum_filtered = spectrum.copy()
    #filter the raw frequency domain with a band-pass filter
    spectrum_filtered[(freq > freq_high)] = 0 
    spectrum_filtered[(freq < freq_low)] = 0

    if inverse:
        signal_filtered = ifft2(fftshift(spectrum_filtered))
        return signal_filtered.real
    else: return spectrum_filtered

def crosshair_filter(image_shape, center, crosshairwidth=8, holeradius=8, cutoff_ratio=0.02, butterworth_order = 5):
    """
    Creating a crosshair-shaped filter.
    like this:
             *****
             *****
        *****     *****
        *****     *****
             *****
             *****

    Args:
       image_shape: [xsize, ysize], the shape size of filter
       center: [x, y], the center of the cross hair filter
       crosshairwidth: int
       holeradius: float,  the radius of the circular gap at the centre of the cross hair
       cutoff_ratio: float, the cutoff ratio in frequency domain
       butterworth_order: int, order of the Butterworth filter

    Return:
       filter
    """
    xsize, ysize = image_shape
    crosshair = np.zeros((ysize, xsize), dtype=float)

    halfx = center[0]
    halfy = center[1]
    offset = int(crosshairwidth / 2)  # the half width of the crosshair

    # Create the crosshair
    crosshair[:, halfy-offset: halfy+offset] = -1
    crosshair[halfx-offset:halfx+offset, :] = -1

    # Use butterworth filtering
    
    butterworth_img = np.zeros((ysize, xsize), dtype=float)
    halfpoint_const = 0.414
    zeroradius = cutoff_ratio*xsize
    
    distance_from_center = cartesian_to_polar(image_shape, center)
    crosshair[distance_from_center < holeradius] = 0
    butterworth_img = 1 / (1 + halfpoint_const * (distance_from_center / zeroradius)**(2 * butterworth_order))

    filtered_crosshair = butterworth_img * crosshair
    filtered_crosshair = filtered_crosshair + 1  # makes the cross hair zero and the rest of the image 1
    
    return filtered_crosshair

def radial_difference_filter(image, space = 'real'):
    """
    Extracting the noise of image from its Fourier transformed space
    assuming its radial distribution.

    # input an image with np.numpy
    # space = 'real', means the image is in real space, otherwise the image is in Fourier space
    # return an image in real domain but its background in reciprocal space
    """
    def filtering_image(img_in_real, space):
        if space=='real':
            img = fftshift(fft2(img_in_real))
        else: img = img_in_real
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2
        distances = np.sqrt((np.arange(img.shape[0])[:, np.newaxis] - center_y) ** 2 + 
                            (np.arange(img.shape[1]) - center_x) ** 2)
        unique_distances = np.unique(np.round(distances).astype(int))
        radial_background = np.zeros_like(img)
        for r in unique_distances:
            radial_background[np.round(distances) == r] = np.mean(img[np.round(distances) == r])
        filtered_img = img - radial_background

        return np.real(ifft2(fftshift(filtered_img))), radial_background
    
    if isinstance(image, list):
        filtered_image, bkg = [filtering_image(np.asarray(img), space) for img in image]
    else:
        filtered_image, bkg = filtering_image(np.asarray(image), space)

    return filtered_image.real, bkg

def adaptive_Fourier_filter(image, mag=50, limit=4, disk_size = 10, neighbor_size = (15,15), threshold=2):
    """
    Manually chooses diffraction spots using interactive point selection and applies inverse 2D FFT.

    It needs self-made functions: "find_local_max", "excluding_spots"

    Note that: this function should be run using interactive mode, e.g. %matplotlib qt5 on jupyter notebook

    Args:
        image (numpy.ndarray): The input image

        mag: showing the FFT image
        limit: int, 1/limit of frequency cutoff
        disk_size: int, the size of filters on each spot
        neighbor_size: a turple, storing the size of seraching area
        threshold: a threshold to pick out spots from their backgrounds beased on the intensity
                   if threshold is assigned with a large value, e.g. 100, then the find_local_max doesnot work.
    Returns:
        filtered_result: image in real space
        filtered_image: image in reciprocal space
    """
    if mag is None:
        mag = 50
    if limit is None:
        limit=4
    if threshold is None:
        threshold = 2
    #Zero-padding the data before performing the transform can help reduce the boundary effects by adding extra zeros around your data, which minimizes the discontinuities.
    sizex, sizey = image.shape
    padded_img = np.pad(image, ((sizex, sizex), (sizey, sizey)), mode='constant', constant_values=0)
    
    px, py = padded_img.shape 

    if neighbor_size is None:
        neighbor_size = (0.01*px, 0.01*py)

    fft_transform = fftshift(fft2(padded_img))
    magnitude_fft = np.log(1+np.abs(fft_transform ))

    gaussian_1= gaussian_filter(magnitude_fft, sigma=(5,5), order=0)
    gaussian_2= gaussian_filter(gaussian_1, sigma=(1,1), order=0)
    mag_fft2 = gaussian_2 + mag*(gaussian_1 - gaussian_2)
    diffraction = tools.linscale(mag_fft2)
    # Interactively pick four points
    plt.imshow(diffraction, cmap = plt.cm.gray)
    center = plt.ginput(5, timeout = -1)
    plt.close()

    chosen_points = tools.find_local_max(diffraction, center, neighbor_size, 1)
    k1 = chosen_points[2] - chosen_points[1]
    k2 = chosen_points[3] - chosen_points[1]

    min_distance = np.min((np.linalg.norm(k1), np.linalg.norm(k2), np.linalg.norm(k1+k2)))
    beam = np.array([[px/2, py/2]])

    points = []
    range_x = int(px / limit)
    arange = np.arange(-range_x, range_x+1)
    for i in arange:
        for j in arange:
            k = i * k1 + j * k2 + np.array([beam[0][0], beam[0][1]])            
            if ((k[0] - beam[0][0]) ** 2 + (k[1] - beam[0][1]) ** 2) <= (sizex // limit)**2:
                nk = tools.find_local_max(diffraction, [k[0], k[1]], neighbor_size, threshold)
                if len(nk)!=0 and 0 <= nk[0][0] < px and 0 <= nk[0][1] < py:
                    points.append(np.array([nk[0][0], nk[0][1]]))
                else: points.append(np.array([k[0], k[1]]))
    
    if disk_size =='' or None or disk_size > int(min_distance/2):
        disk_size = int(min_distance/2)
    # Create filters for each candidate point
    filters = np.zeros((px, py))
    if len(points) !=0:   
        for blob in points: 
            built_filter = tools.circle_mask((px, py), blob, (0, disk_size))
            filters += built_filter


    filtered_image = fft_transform * filters.T


    filtered_img = ifft2(ifftshift(filtered_image)) 
    filtered_result = filtered_img[sizex:-sizex, sizey:-sizey]

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), tight_layout =True)
    axs[0,0].imshow(diffraction, cmap='gray')
    axs[0,0].set_title('Fourier transform')
    axs[0,0].scatter(np.asarray(chosen_points)[:, 0], np.asarray(chosen_points)[:, 1])
    axs[0,0].scatter(beam[0][0], beam[0][1])
    axs[0,0].set_xlim(sizex, 2*sizex)
    axs[0,0].set_ylim(sizey, 2*sizey)
    axs[0,1].imshow(diffraction, cmap='gray')
    axs[0,1].scatter(np.asarray(points)[:, 0], np.asarray(points)[:, 1])
    axs[0,1].set_title('Bragg peaks')
    axs[0,1].set_xlim(sizex, 2*sizex)
    axs[0,1].set_ylim(sizey, 2*sizey)
    axs[1,0].imshow(diffraction*filters)
    axs[1,0].set_title('Filterred image')
    axs[1,0].set_xlim(sizex, 2*sizex)
    axs[1,0].set_ylim(sizey, 2*sizey)
    axs[1,1].imshow(np.real(filtered_result))
    axs[1,1].set_title('Virtual image')
    plt.show()

    return np.real(filtered_result)

def elliptical_Moffat_filter(size,sigma1,sigma2,theta, beta):
        """
        Defines an elliptical Moffat distribution, which is a continuous probability distribution 
        based upon the Lorentzian distribution.
        Its particular importance in astrophysics is due to its ability to accurately reconstruct 
        point spread functions, whose wings cannot be accurately portrayed by 
        either a Gaussian or Lorentzian function.

        Inputs:
            sigma1, sigma2 : float  Defines upper frequencies allowed, in pixels - i.e. features
                                smaller than ~n pixels are smoothed - where n1 and n2 are
                                the mimumum feature sizes along the two primary axes.  Thus
                                The ellipticity is epsilon = n1/n2.
                                The cutoff frequency corresponding to the n's are set to 3*sigma
            theta :  float      The angle of the x axis
            beta : float 
        """
        if sigma1 == 0 or sigma2 == 0:
            print("ERROR: sigma cannot be zero.")

        nx = ny = size

        qx = qy = np.fft.fftfreq(size)
        kx = qx[:, None]
        ky = qy[None, :]
        theta = np.radians(theta)
        a = (np.cos(theta)**2)/(sigma1**2) + (np.sin(theta)**2)/(sigma2**2)
        b = (np.sin(theta)*np.cos(theta))*(1/(sigma1**2) - 1/(sigma2**2))
        c = (np.sin(theta)**2)/(sigma1**2) + (np.cos(theta)**2)/(sigma2**2)
        denominator = 1/(1 + (a*kx**2 + 2*b*kx*ky + c*ky**2) )**beta
        mask_fourierspace = np.fft.fftshift(denominator)
        return mask_fourierspace

def elliptical_Fourier_filter(size,sigma1,sigma2,theta):
        """
        Defines an elliptical Gaussian Fourier space mask 

        Inputs:
            sigma1, sigma2 : float  Defines upper frequencies allowed, in pixels - i.e. features
                                smaller than ~n pixels are smoothed - where n1 and n2 are
                                the mimumum feature sizes along the two primary axes.  Thus
                                The ellipticity is epsilon = n1/n2.
                                The cutoff frequency corresponding to the n's are set to 3*sigma
            theta :  float      The angle of the x axis
        """
        if sigma1 == 0 or sigma2 == 0:
            print("ERROR: sigma cannot be zero.")

        nx = ny = size

        qx = qy = np.fft.fftfreq(size)
        kx = qx[:, None]
        ky = qy[None, :]
        theta = np.radians(theta)
        a = (np.cos(theta)**2)/(2*sigma1**2) + (np.sin(theta)**2)/(2*sigma2**2)
        b = (np.sin(theta)*np.cos(theta))*(1/(2*sigma1**2) - 1/(2*sigma2**2))
        c = (np.sin(theta)**2)/(2*sigma1**2) + (np.cos(theta)**2)/(2*sigma2**2)

        mask_fourierspace = np.fft.fftshift(np.exp( -(a*kx**2 + 2*b*kx*ky + c*ky**2) ))
        return mask_fourierspace

def gaussian_bandpass(img, space='real', mode="low", cutoff_ratio=0.1):
    """
    same as "scipy.ndimage.fourier_gaussian(img, sigma=cutoff_ratio)"
    Employing a Gaussian filter on an image in its Fourier space.

    Args:
        img: image array to be filtered, must be square
        cutoff_ratio: cutoff ratio in frequency domain, equal with standard deviation
        mode: string, "low" or "high", representing low-band pass or high-band pass filter
    """
    if mode is None:
        mode = "low"
    sizex, sizey = img.shape
    r = cartesian_to_polar([sizex, sizey], [sizex//2, sizey//2])

    cutoff = sizex * cutoff_ratio
    if mode == "low":
        gaussian_fr = np.exp(- (r**2) / (2 * (cutoff**2)))
    else:
       gaussian_fr = 1 - np.exp(- (r**2) / (2 * (cutoff**2)))
    if space =='real':
        img_fft = fftshift(fft2(img))
        filtered_fshift = img_fft * gaussian_fr
    else:
        filtered_fshift = img * gaussian_fr *sizex*sizey

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout =True)
    axes[0].imshow(np.log(np.abs(gaussian_fr)+.1))
    axes[0].set_title(mode+"_pass Gaussian filter")
    axes[1].imshow(np.log(np.abs(filtered_fshift)+.1))
    axes[1].set_title("After applying Gaussian filter")
    plt.show()
    return filtered_fshift

def gaussian_filter(
       size, cut_ratio, bandpass ='low'
    ):
    """
    It is to build a 2D array Gaussian filter for image processing in the Fourier space.

    Args:
        size: Tuple, determing the shape of the built filter.
        cut_ratio: float, the cutoff frequency
        bandpass: str, 'low' or 'high'

    Return:
        filter: np.ndarray
    """
    if size[0] != size[1]:
        nd = max(size)
    else: nd = size[0]
    qx = np.fft.fftshift(np.fft.fftfreq(nd))
    qy = np.fft.fftshift(np.fft.fftfreq(nd))
    max_freq = max(qx)
    qya, qxa = np.meshgrid(qy, qx)
    qra = np.sqrt(qxa**2 + qya**2)

    env = np.ones_like(qra)
    if bandpass == 'high':
        q_highpass = max_freq * (1 - cut_ratio)
        env *= 1 -np.exp(- (qra**2) / (2 * (q_highpass**2)))
    elif bandpass == 'low':
        q_lowpass = max_freq * cut_ratio
        env *= np.exp(- (qra**2) / (2 * (q_lowpass**2)))
    else:
        raise ValueError("Choosing the type of bandpass: 'low' or 'high'.")

    return env

def butterworth_filter(
       size, cut_ratio, bandpass ='low', order =2, squared_butterworth = False
           ):
    """
    It is to build a 2D array Butterworth filter for image processing in the Fourier space.

    Args:
        size: Tuple, determing the shape of the built filter.
        cut_ratio: float, the cutoff frequency
        bandpass: str, 'low' or 'high'
        order: int, the order of Butterworth filter
        squared_butterworth: bool, if True, a squared filter will be return.

    Return:
        filter: np.ndarray

    """
    if size[0] != size[1]:
        nd = max(size)
    else: nd = size[0]
    qx = np.fft.fftshift(np.fft.fftfreq(nd))
    qy = np.fft.fftshift(np.fft.fftfreq(nd))
    max_freq = max(qx)
    qya, qxa = np.meshgrid(qy, qx)
    qra = np.sqrt(qxa**2 + qya**2)

    env = np.ones_like(qra)
    if bandpass == 'high':
        q_highpass = max_freq * (1 - cut_ratio)
        env *= 1 - 1 / (1 + (qra / q_highpass) ** (2 * order))
    elif bandpass == 'low':
        q_lowpass = max_freq * cut_ratio
        env *= 1 / (1 + (qra / q_lowpass) ** (2 * order))
    else:
        raise ValueError("Choosing the type of bandpass: 'low' or 'high'.")
    if squared_butterworth:
        env = np.sqrt(env)
    return env

def pure_bandpass(mask_size, cutoff, center=None):
    """
    It is a pure bandpass mask, which only enables the frequency to band pass by.
    Args:
        mask_size: turple, specify the size of the whole mask, e.g. [128,128]
        cutoff: turple, specify the range of frequency passed by the mask, e.g.[0.1, 0.8], indicating the frequency within 10% to 80% of the maximum can pass by.
        center: turple, e.g. [64,64], representing the center of 2D filter in the built mask.
    """
    if center is None:
        center = (mask_size[0]//2, mask_size[1]//2)
    if center[0] > mask_size[0]:
        center[0] = mask_size[0]
    if center[1] > mask_size[1]:
        center[1] = mask_size[1]
    
    radius = np.sqrt(mask_size[0]**2/4+mask_size[1]**2/4)
    inner = cutoff[0] * radius
    outer = cutoff[1] * radius
    mask = np.ones(mask_size)
    y, x = np.ogrid[:mask_size[0], :mask_size[1]]
    distance_from_center = (x - center[0])**2 + (y - center[1])**2
    mask[distance_from_center < inner**2] = 0
    mask[distance_from_center > outer**2] = 0
    return mask

def butterworth_bandpass(img, space='real',mode='low', order=2, cutoff_ratio=0.1):
    """
    Employing a Butterworth lowpass filter on the image's FFT.
    𝜔 is the angular frequency in radians per second and 
    𝑛 is the number of poles in the filter—equal to the number of reactive elements in a passive filter. 
    Its cutoff frequency (the half-power point of approximately −3 dB or a voltage gain of 1/√2 ≈ 0.7071) is normalized to 𝜔 = 1 radian per second. 

    Args:
        img: FFT image array to be filtered, must be square
        order: Butterworth order, controlling how shaply the transition from the pass-band to the stop-band
        cutoff_ratio: cutoff ratio in frequency domain
    """
    rows, cols = img.shape
    if rows != cols:
        max_size = max(rows, cols)
        padded_image = np.zeros((max_size, max_size))
        padded_image[:rows, :cols] = img
        image = padded_image
    else: image = img

    sizex, sizey = image.shape

    r = cartesian_to_polar([sizex, sizey], [sizex//2, sizey//2])
    if mode=='low':
        bw = 1/(1+0.414*(r/(cutoff_ratio * sizex))**(2*order))
    else:
        bw = 1 - 1/(1+0.414*(r/(cutoff_ratio * sizex))**(2*order))
    
    if space =='real':
        img_fft = fftshift(fft2(image))
        filtered_image = img_fft * bw
    else:
        filtered_image = image * bw 

    filtered_image = filtered_image[:rows, :cols]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), tight_layout =True)
    axes[0].imshow(bw)
    axes[0].set_title("LowPass Butterworth filter")
    axes[1].imshow(np.log(np.abs(filtered_image)+.1))
    axes[1].set_title("After applying Butterworth filter")
    plt.show()
    return filtered_image

def average_background(img, delta=5):
    """
    Function to get an averaged background of the reciprocal sapce of HR image.
    Args:
        img: 2D array of reciprocal space HR image data
        
        delta: a threshold for background averaging
    Return:
        bkg: np.ndarray, in Fourier space
    """
    sizex, sizey = img.shape
    r = cartesian_to_polar([sizex, sizey], [sizex//2, sizey//2])
    r = r.astype(int)
    
    # Get a Butterworth filter on image to remove the edge effect
    noedgebw = 1/(1+0.414*(r/(0.4 * r.shape[0]))**(2*12))
 
    f_noedge = img * noedgebw
    
    # Light filter the FFT for processing
    f_img = median_filter(np.abs(f_noedge), size=5)
    
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

def avg_background(img, space ='real', delta=5):
    """
    Function to get an averaged background from a real-space HR image.
    Args:
        img: 2D array of real-space HR image data
        space: the image is in real space or Fourier space
        delta: a threashold for background averaging
    Return:
        bkg: np.ndarray, in Fourier space
    """
    sizex, sizey = img.shape
    r = cartesian_to_polar([sizex, sizey], [sizex//2, sizey//2])
    r = r.astype(int)
    
    # Get a Butterworth filter on image to remove the edge effect
    noedgebw = 1/(1+0.414*(r/(0.4 * r.shape[0]))**(2*12))
    if space=='real':
        noedgeimg = img * noedgebw
        f_noedge = fftshift(fft2(noedgeimg))
    else: f_noedge= img*noedgebw
    # Light filter the FFT for processing
    f_img = median_filter(np.abs(f_noedge),size=5)

    # Convert the FFT data into flattened real magnitude data
    f_mag = f_img.flatten()
    # Flatten the r array
    r_mag = r.flatten()
    df = pd.DataFrame({'r_mag': r_mag, 'f_mag': f_mag})    
    grouped = df.groupby('r_mag')['f_mag'].apply(list)   
    r_bin = np.array(grouped.index)
    f_bin = np.array(grouped.values) 

    f_mean0 = [np.mean(bin) for bin in f_bin]

    # For each r bin, replace the pixels > mean with mean0, and take the new mean1.
    # Compare the mean1 and mean0, if the difference is < threashold%, stop
    # This needs to be done through all the bins
    # Make a copy of the original mean list

    f_mean = f_mean0[:]
    for i in range(len(f_bin)):
        bin = np.array(f_bin[i])
        mean = np.mean(bin) * (100 + delta) * 0.01 # Overshoot by the threashold 
        diff_pc = np.inf  # Initialize difference as infinity
        while diff_pc > delta and mean > 0: # While the percentage difference between mean values is greater than the threshold
            bin[bin > mean] = mean    # Replace any data > mean 
            new_mean = np.mean(bin) * (100 + delta) * 0.01    # Recalculate mean           
            diff_pc = np.abs((new_mean - mean) / mean) * 100   # Calculate the percentage difference             
            mean = new_mean  # Update the mean value
        if mean < 0:
            mean = 0        
        f_mean[i] = mean # Update the mean data

    f_avg_r = dict(zip(r_bin, f_mean)) # Construct the background array
    f_avg = np.zeros(f_img.shape)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            f_avg[i, j] = f_avg_r[r[i, j]]
                                    
    return f_avg



# Wiener filter function
def wiener_filter(img, delta=5, lowpass=True, lowpass_cutoff=0.3, lowpass_order=2):
    """
    Wiener filter for HRTEM images.

    Args:
        img: the image data array
        delta: a threashold for background averaging
        lowpass: also apply a lowpass filter after filtering
        lowpass_cutoff: a cutoff ratio in frequency domain for the lowpass
        lowpass_order: order for the Butterworth filter; smaller int retults more tapered cutoff

    Return: 
        img_wf, img_diff: filtered image array and difference

    """

    pad_x, pad_y = 16,16
    
    padded_img = np.pad(img, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant', constant_values=0)
    px, py = padded_img.shape
    f_img = fftshift(fft2(padded_img))

    fu = np.abs(f_img)
    inverse_fu = np.reciprocal(fu, where = fu !=0)
    inverse_fu[inverse_fu==np.inf] = 1
    fa = avg_background(padded_img, 'real', delta)
    wf = (fu**2 - fa**2)*inverse_fu**2
    wf[wf<0] = 0
    if lowpass_cutoff !=0:
        mask_size = lowpass_cutoff * px/img.shape[0]
        mask = tools.circle_mask((px, py), (px//2, py//2), radius = (0, 0.5*px*mask_size))
    else:
        mask = 1
        
    f_img_wf = f_img * wf * mask
    
    if lowpass:
        sx, sy = padded_img.shape
        r = cartesian_to_polar([sx, sy], [sx//2, sy//2])
        bw = 1/(1+0.414*(r/(lowpass_cutoff * sx))**(2*lowpass_order))
        filtered_fshift = f_img_wf * bw
        img_wf = ifft2(ifftshift(filtered_fshift))
    else:
        img_wf = ifft2(ifftshift(f_img_wf))
    img_wf = np.real(img_wf)
    img_result = np.single(img_wf[pad_x:-pad_x, pad_y:-pad_y])
    img_diff = img - img_result
    return img_result, img_diff


def abs_filter(img, delta=5, lowpass=True, lowpass_cutoff=0.3, lowpass_order=2):
    """
    Average background subtraction filter function (ABS) filter for HRTEM images

    Args:
        img: the image data array
        delta: a threashold for background averaging
        lowpass: also apply a lowpass filter after filtering
        lowpass_cutoff: a cutoff ratio in frequency domain for the lowpass
        lowpass_order: order for the Butterworth filter; smaller int retults more tapered cutoff
    Return: 
        img_absf, img_diff: filtered image array and difference
    """
    pad_x, pad_y = 16,16
    
    padded_img = np.pad(img, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant', constant_values=0)
    px, py = padded_img.shape
    f_img = fftshift(fft2(padded_img))

    fu = np.abs(f_img)
    fa = avg_background(padded_img, 'real', delta)
    inverse_fu = np.reciprocal(fu, where = fu !=0)
    inverse_fu[inverse_fu==np.inf] = 1
    #absf = (fu - fa)*inverse_fu #the difference with the wiener filter
    absf = (np.abs(fu) - np.abs(fa))*np.abs(inverse_fu)   # modify
    absf[absf<0] = 0
    
    if lowpass_cutoff !=0:
        mask_size = lowpass_cutoff * px/img.shape[0]
        mask = tools.circle_mask((px, py), (px//2, py//2), radius = (0, 0.5*px*mask_size))
    else:
        mask = 1
        
    f_img_absf = f_img * absf * mask
    
    if lowpass:
        sx, sy = padded_img.shape
        r = cartesian_to_polar([sx, sy], [sx//2, sy//2])
        bw = 1/(1+0.414*(r/(lowpass_cutoff * sx))**(2*lowpass_order))
        filtered_fshift = f_img_absf * bw
        img_absf = ifft2(ifftshift(filtered_fshift))
    else: 
        img_absf = ifft2(ifftshift(f_img_absf))
    img_absf = np.real(img_absf)
    img_result = np.single(img_absf[pad_x:-pad_x, pad_y:-pad_y])
    img_diff = img - img_result
    return img_result, img_diff

     
def nonlinear_filter(img, space ='real', N=50, mode='wiener', delta=10, lowpass_cutoff=0.3, lowpass = True, lowpass_order=2):
    """
    Non-linear filter, calling wiener_filter or abs_filter

    Args:
        img: img 2D-array
        N: number of iterations
        mode: choose one filter, ['wiener', 'abs']
        lowpass_cutoff: cutoff of the low pass filter
        lowpass: apply a Butterworth lowpass filter after Wiener filter
        The Butterworth filter will use lowpass_order and lowpass_cutoff

    Return: 
        img_filtered, img_diff: filtered image array and difference
    """
    try: from tqdm.notebook import tqdm
    except ImportError: print("There is no tqdm module!")
    sizex, sizey = img.shape
    r = cartesian_to_polar([sizex, sizey], [sizex//2, sizey//2])
    cutoff = sizex/2 * lowpass_cutoff
    gaussian_f = np.exp(- (r**2) / (2 * (cutoff**2)))
    x_in = img
    i=0
    if mode is None:
        mode = 'wiener'

    pbar = tqdm(total=N, desc="Building",unit="iteration", bar_format="{l_bar}{bar} [ time left: {remaining} ]")
    while i < N: 
        if space=='real':
            fshift = fftshift(fft2(x_in)) 
            filtered_fshift = fshift *gaussian_f
            
        else: 
            filtered_fshift = x_in*gaussian_f
            x_in_image = ifft2(ifftshift(x_in))

        img_glp = ifft2(ifftshift(filtered_fshift))
        x_lp = np.real(img_glp)
        if space=='real':
            x_diff = x_in - x_lp
        else: x_diff = x_in_image - x_lp
        
        if mode == 'wiener':
            x_diff_wf, _ = wiener_filter(x_diff,                                       
                                     delta=delta,
                                     lowpass=lowpass, 
                                     lowpass_cutoff=lowpass_cutoff, 
                                     lowpass_order=lowpass_order)
        elif mode=='abs':
            x_diff_wf, _ = abs_filter(x_diff, 
                                     delta=delta,
                                     lowpass=lowpass, 
                                     lowpass_cutoff=lowpass_cutoff, 
                                     lowpass_order=lowpass_order)
        x_in = x_lp + x_diff_wf
        i = i+1
        pbar.update(1)
    pbar.close()
    img_filtered = np.single(x_in) # Convert to 32 bit float

    img_diff = img - img_filtered

    return img_filtered, img_diff







