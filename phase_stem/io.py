from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import zarr
import dask.array as da
import mrcfile
from PIL import Image, ImageDraw, ImageFont
from phase_STEM import tools
import hyperspy.api as hs 
from matplotlib_scalebar.scalebar import ScaleBar

def find_files(root, pattern):
    """
    This function is to search files with an extension 'pattern' from a folder 'root', as well as the subfolders
    
    Args:
        root: string, the address of the target folder
        pattern: string, the format of the file, e.g. '.emd', '.dm3'
    
    Return: 
    a list with the files searched from the folder
    """
    matches = []
  
    for path, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(pattern):
                file_path = os.path.join(path, filename)
                matches.append(file_path)    
    return matches

def get_image_intensity(image_path, crop_size = [False, (64,64), 128]):
    """
    Calculating the intensity of an image/frame
    """
    image = Image.open(image_path)
    if image.mode == "RGBA" or image.mode == "RGB":
        image = image.convert('L')  # Ensure the image is in grayscale
    image_array = np.array(image)

    if crop_size[0]:
        mask = tools.circle_mask(image_array.shape, (crop_size[1]), (0, crop_size[2]))
        intensity = np.sum(image_array*mask)
    else: 
        intensity = np.sum(image_array)
    return intensity

def compute_images_intensities(image_path, file_extension='.tiff', crop_size = [False, (64,64), 128], num_workers=8):
    """
    It is employed to calculate the intensities of frames stored in a folder, plotting the intensities as the frame sequence.
    It is suggested using "matplotlib qt5" to show the plot

    Args:
        image_path: string, the path of folder containing images
        file_extension: string, the format of images for calculation
        num_workers: int, the number of threads

    Return:
        files: list, names of found images
        intensities: np.ndarray, 1-D array
    """
    files = find_files(image_path, file_extension)
    num = len(files)
    intensities = np.zeros(num)
    pbar = tqdm(total=num, desc="Computing", unit="frame", bar_format="{l_bar}{bar} [ time left: {remaining} ]")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(get_image_intensity, file, crop_size): (idx, file)
            for idx, file in enumerate(files)
                    }
      
        for future in as_completed(future_to_index):
            idx, file = future_to_index[future]
            try:                    
                intensities[idx] = future.result()/12 #gainfactor = 12

            except Exception as exc:
                print(f"Failure to calculate frame {file}: {exc}")
            pbar.update(1)
        pbar.close()

    plt.plot(np.arange(len(intensities)), intensities, 'o-')
    plt.xlabel('Frame Index (No.)')
    plt.ylabel('Total Intensity (counts)')
    plt.title('Frame Intensities')
    plt.show()
    return files, intensities

def delete_image_if_needed(image_path, allowed_format, delete, threshold):
    """
     Deletes images with total intensities lower or higher than "threshold" from a given folder.

    Args:
        image_path: the path of targeted folder.
        allowed_format: the format of images, e.g., '.tiff'.
        delete: string, "up" means deleting the frames whose intensities are more than "threshold", while "down" is inverse.
        threshold: float, if the total intensity of one image is less than this value, then this image will be deleted.
    Return:
        image_path
        intensity
    """
    if image_path.lower().endswith(allowed_format):
        intensity = get_image_intensity(image_path)
        if (intensity < threshold and delete == "down") or (intensity > threshold and delete == "up"):
            os.remove(image_path)
            return image_path, intensity
    return None, None

def delete_images_from_folder(folder_path, allowed_format='.png', delete="down", threshold=100, num_images=100):
    """
    Deletes images with total intensities lower or higher than "threshold" from a given folder.

    Args:
        folder_path: the address of targeted folder.
        allowed_format: the format of images, e.g., '.tiff'.
        delete: string, "up" means deleting the frames whose intensities are more than "threshold", while "down" is inverse.
        threshold: float, if the total intensity of one image is less than this value, then this image will be deleted.
        num_images: int, the maximum images for deleting from targeted folder.
    """
    #image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(allowed_format) and os.path.isfile(os.path.join(folder_path, f))]
    image_files = find_files(folder_path, allowed_format)
    total = len(image_files)
    print(f"{total} {allowed_format} images are found in {folder_path}")

    if num_images is None or num_images == 0:
        num_images = total
    
    if total > 0:
        print(f"Starting deleting images whose intensities are {'lower' if delete == 'down' else 'higher'} than {threshold}...")

        deleted_count = 0

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(delete_image_if_needed, filename, allowed_format, delete, threshold): filename for filename in image_files}
            for future in tqdm(as_completed(futures), total=min(total, num_images), desc="Processing images"):
                image_path, intensity = future.result()
                if image_path:
                    print(f"Deleted {os.path.basename(image_path)} with intensity: {intensity}")
                    deleted_count += 1
                    if deleted_count >= num_images:
                        break

        print(f"\n{deleted_count} images are deleted from the folder {folder_path}")
        print(f"{total - deleted_count} images are kept!")



def test_shape_of_array(num_images, array=(128,128)):
    """
    Checking the sequence of selected images are correct or not.
    
    Args:
        num_images: int, the number of images presumed to be aranged in an array.
        array: the shape of the array
    """
    if array is None or array[0] ==0:
        row = int(np.sqrt(num_images))
    else: row = int(array[0])
    if array is None or array[1]==0:
        column = int(num_images/row)
    else: column = int(array[1])
    #skipping how many image every row
    skip = (num_images-row*column)//(row-1)
    
    selected_num = list(range(0, num_images))      
    selected_images = []
    if skip >0:  
        index = 0
        num_row = 0
        while index < num_images and num_row < row:
            selected_images.extend(selected_num[index:index+row])
            index += row + skip
            num_row +=1
    else: selected_images = selected_num
    nm = len(selected_images)
    print(f"There are {nm} picked out from {num_images} images for rearanging as following array...")
    print(f"Rows = {row}, Columns = {column}, Number of Skip = {skip}")
    print("selected images as sequences...")
    for i in range(0, nm, array[0]):        
        print(f"Row {i//array[0]}:")
        print(*selected_images[i:i+array[0]])
        print('\n')

def get_neighbors_avg(image_array, i, j, z):
    """
    judging a pixel [i, j] with zero intensity is dead point or not based on its surrounding pixels, located "z" pixels away
    """
    rows, cols = image_array.shape
    neighbors = image_array[max(0, i-z):min(rows, i+1+z), max(0, j-z):min(cols, j+1+z)].flatten()

    # Exclude the center pixel
    center_index = (neighbors.size - 1) // 2
    neighbors = np.delete(neighbors, center_index)
    
    # Get non-zero neighbors
    zero_neighbors = neighbors[neighbors == 0]
    
    # Check if more than half of the neighbors are non-zero
    if len(zero_neighbors) == 0: #condition for the assignment of intensity in zero_location
        return neighbors.mean()
    else:
        return 0

def cheeTahIntensity_correction(image_array, grad=5, column=(255, 256), row=(255, 256), zero_correction=3):
    """
    Corrects the intensity of images acquired by CheeTah camera, which has a cross in the image.
    The crossing looks like:
                       column
                      255  256
                         ++
                         ++
                         ++
       rows              ++
        255    ++++++++++++++++++++++
        256    ++++++++++++++++++++++
                         ++
                         ++
                         ++
                         ++
    Args:
        image_array: np.ndarray
        grad: int, the difference in intensity
        column, row: the cross feature in the camera
        zero_correction: tuple, the first value decides "True" or "False", the second one is an int, means the neighbor size
    """
    # Create a copy of the image to avoid modifying the original one
    modified_image = (image_array.copy()).astype(np.int16) #cheeTah camera's pixel depth is 12 bits.
    
    # Get the dimensions of the image
    rows, cols = image_array.shape
    
    rng = np.random.default_rng()  # Random number generator for consistent random values
    
    # Process columns 255 and 256
    for j in range(cols):
        for i in row:
            neighbors = image_array[max(0, i-2):min(rows, i+3), j].flatten()
            neighbors = np.delete(neighbors, [2, 3])  # Exclude center pixels for each row
            average_neighbors = np.mean(neighbors)
            if image_array[i, j] > average_neighbors + grad:
                modified_image[i, j] = average_neighbors * (0.5 + rng.random())

    # Process rows 255 and 256
    for i in range(rows):
        for j in column:
            neighbors = image_array[i, max(0, j-2):min(cols, j+3)].flatten()
            neighbors = np.delete(neighbors, [2, 3])  # Exclude center pixels for each column
            average_neighbors = np.mean(neighbors)
            if image_array[i, j] > average_neighbors + grad:
                modified_image[i, j] = average_neighbors * (0.5 + rng.random())

    # Process the central 4 pixels
    centres = image_array[max(0, row[0]-1):min(rows, row[1]+2), max(0, column[0]-1):min(cols, column[1]+2)]
    centres = np.delete(centres, [2, 3, 5, 6]).flatten()  # Exclude center 4 pixels
    average_intensity = np.mean(centres)
    for i in row:
        for j in column:
            if image_array[i, j] > average_intensity + grad:
                modified_image[i, j] = average_intensity * (0.5 + rng.random())
    
    # Replace zero intensity pixels with the average of their neighbors
    #this operation consumes so much time, be careful
    if zero_correction is None or zero_correction==0:
        z = 1
    else: z = zero_correction
        
    if z!=1:
        zero_pixels = np.argwhere(image_array == 0)
        count = 0        
        for r, c in zero_pixels:
            correct = get_neighbors_avg(modified_image, r, c, z)
            if correct > 0:
                modified_image[r, c] = np.round(correct, 2) 
                count += 1
                print(f"The dark point : ({r}, {c}) given with an intensity : {modified_image[r, c]}")
        print(f"{count} dark pixels are intensity-corrected")

    return modified_image

def process_image(image_file, 
                  crop = {"crop image":False, "crop centre":(128,128), "crop size":(256,256)}, 
                  cheetah= {"intensity correction":True, "intensity difference": 5, 
                            "cross in column":(255, 256), 
                            "cross in row":(255, 256), 
                            "dead pixels correction":0}):
    """
    It is utilized to pre-process the frame acquired by CheeTah camera.
    
    Args:
        image_file: string, the path of the frame stored.
        
        crop: dictionary, whether the output frame is cropped from its raw data.
                        e.g. crop = {"crop image":True, "crop centre":(128,128), "crop size":(256,256)}
                        
        cheetah: dictionary, which contains the parameters for correcting the intensity of frames, including the cross and dead pxiels
                        e.g. cheetah= {"intensity correction":True, 
                            "intensity difference": 5, 
                            "cross in column":(255, 256), 
                            "cross in row":(255, 256), 
                            "dead pixels correction":3}
                            
                    where the "dead pixel correction" refers to the process of identifying 
                    and correcting pixels that have zero intensity while their surrounding pixels, 
                    located three pixels away, have non-zero intensity. 
                    These zero-intensity pixels are considered dead pixels.
    """                 
    img = Image.open(image_file) #path of image
    # Ensure the image is in grayscale             
    if img.mode == "RGBA" or img.mode == "RGB":
        img = img.convert('L')  
    raw_array = np.array(img, dtype=np.int16)
    name = os.path.basename(image_file)
    if cheetah["intensity correction"]:
        raw_array = cheeTahIntensity_correction(raw_array, cheetah["intensity difference"], 
                                                cheetah["cross in column"], 
                                                cheetah["cross in row"], 
                                                cheetah["dead pixels correction"])
    if crop["crop image"]: 
        img_array = tools.crop_matrix(raw_array, crop["crop centre"], crop["crop size"])        
    else: img_array = raw_array
        
    return img_array, name

def pack_images_to_hdf5(image_folder, file_name='packed_images', 
                        storage_mode="zarr",
                        array=(128,128), 
                        crop = {"crop image":True, "crop centre":(128,128), "crop size":(256,256)}, 
                        cheetah= {"intensity correction":True, "intensity difference": 5, 
                            "cross in column":(255, 256), 
                            "cross in row":(255, 256), 
                            "dead pixels correction":0},
                        num_workers=32):
    """
    It can restore images from a folder to a .h5 file using h5py with a shape of (R, C, N, M),
    which R is the row number, C is the column number, (N, M) represent the pixelsize of single frame
    
    The default formats of images are ('.png', '.jpeg', '.jpg', '.tiff', '.gif', '.bmp')
    
    Args: 
        image_folder: string, the folder path
        file_name: a given name for the hdf5 file
        storage_mode: string, "zarr", "h5py", "dask"
        crop: list, the first one means whether cropping the frame or not
                    the second (e.g.(128, 128)) gives the center coordinate of cropped frame
                    the third (e.g. (256, 256)) which much be given represents the shape of stored frames
        array: int, the stack frames will be arranged in hdf5 file with a shape of (R, C), where R is the number of row and the (N, M) is the shape of single frame
        
        num_workers: int, the number of threads utilized for this code
    """

    beginning = time.time()
    
    # Get a list of all image files in the folder
    allowed_formats = ('.png', '.jpeg', '.jpg', '.tiff', '.gif', '.bmp')
    #image_files = [f for f in os.listdir(image_folder) 
     #              if f.lower().endswith(allowed_formats) and os.path.isfile(os.path.join(image_folder, f))]

    image_files = find_files(image_folder, allowed_formats) 
    # Sort image files to maintain consistent order
    image_files.sort()

    num_images = len(image_files)
    if array is None or array[0] ==0:
        row = int(np.sqrt(num_images))
    else: row = int(array[0])
    if array is None or array[1]==0:
        column = int(num_images/row)
    else: column = int(array[1])
    #skipping how many image every row
    skip = (num_images-row*column)//(row-1)

    if skip <0:
        print(f"Oh, no! The presumed array shape {shape} is incorrect!")
    selected_num = list(range(0, num_images))      
    selected_images = []
    if skip >0:  
        index = 0
        num_row = 0
        while index < num_images and num_row < row:
            selected_images.extend(selected_num[index:index+row])
            index += row + skip
            num_row +=1
    else: selected_images = selected_num
    nm = len(selected_images)
    print(f"There are {num_images} images, which {nm} images will be picked out , converted and restored using {storage_mode}...")
    print(f"Rows = {row}, Columns = {column}, Number of Skip = {skip}")
    
    counting = 0
    # Create a new HDF5 file
    if crop["crop image"]:
        shape = crop["crop size"]
    else: 
        img_test = Image.open(image_files[0])
        shape = img_test.size
        
    # Initialize the progress bar
    pbar = tqdm(total=len(selected_images), desc="Converting", unit="frame", bar_format="{l_bar}{bar} [ time left: {remaining} ]")
    if storage_mode=="h5py":
        h5_path = os.path.join(image_folder, file_name + '.h5')
        hdf5_file = h5py.File(h5_path, 'w')
        # Create a dataset in the HDF5 file to store the images
        grp = hdf5_file.create_group("Raw datasets")
        dataset = grp.create_dataset('raw frames', shape=(nm, *shape), dtype=np.int16)
        img_names = grp.create_dataset('frame names', (nm,), dtype=h5py.string_dtype(encoding="utf-8"))
        
         # Open the thread pool executor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Dictionary to map futures to their indices and frames
            future_to_index = {
                executor.submit(process_image, image_files[frame], crop, cheetah): (idx, frame)
                for idx, frame in enumerate(selected_images)
                    }
      
            for future in as_completed(future_to_index):
                idx, frame = future_to_index[future]
                try:                   
                    img_array, img_name = future.result()                
                    # Store the processed image data and name in the appropriate structures
                    img_names[idx] = f'<{idx}>{img_name}'
                    dataset[idx, :, :] = img_array
                    counting += 1
                except Exception as exc:
                    print(f"Failure to store frame {image_files[frame]}: {exc}")
                pbar.update(1)

    elif storage_mode=="zarr":
        zarr_path = os.path.join(image_folder, file_name + '.zarr')
        zarr_group = zarr.open(zarr_path, 'w')
        grp = zarr_group.create_group("Raw datasets")
        dataset = grp.create_dataset('raw frames', shape=(nm, *shape), dtype=np.int16, chunks=(1, *shape))
        img_names = grp.create_dataset('frame names', shape=(nm,), dtype=str, chunks=(row, ))
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Dictionary to map futures to their indices and frames
            future_to_index = {
                executor.submit(process_image, image_files[frame], crop, cheetah): (idx, frame)
                for idx, frame in enumerate(selected_images)
                    }
      
            for future in as_completed(future_to_index):
                idx, frame = future_to_index[future]
                try:                   
                    img_array, img_name = future.result()                
                    # Store the processed image data and name in the appropriate structures
                    img_names[idx] = f'<{idx}>{img_name}'
                    dataset[idx, :, :] = img_array
                    counting += 1
                except Exception as exc:
                    print(f"Failure to store frame {image_files[frame]}: {exc}")
                pbar.update(1)

    else: 
        arrays = []
        img_names = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Dictionary to map futures to their indices and frames
            future_to_index = {
                executor.submit(process_image, image_files[frame], crop, cheetah): (idx, frame)
                for idx, frame in enumerate(selected_images)
                    }
      
            for future in as_completed(future_to_index):
                idx, frame = future_to_index[future]
                try:                   
                    img_array, img_name = future.result()                
                    arrays.append(da.from_array(img_array, chunks=(shape)))
                    img_names.append(img_name)
                    counting += 1
                except Exception as exc:
                    print(f"Failure to store frame {image_files[frame]}: {exc}")
                pbar.update(1)        
        stack = da.stack(arrays, axis=0)
        h5_path = os.path.join(image_folder, file_name + '.h5')
        # Save both the image data and the names
        with h5py.File(h5_path, 'w') as hdf5_file:
            hdf5_file.create_dataset('Raw datasets/raw frames', data=stack)
            hdf5_file.create_dataset('Raw datasets/frame names', data=np.array(img_names, dtype=h5py.string_dtype(encoding="utf-8")))

    pbar.close()
    print("\n")
    print(f"{counting} images restored successfully using {storage_mode}!")

    spend_time = time.time() - beginning
    if spend_time > 1:
        print(f"The whole procedure takes {round(spend_time, 1)} seconds.")
    elif 0.001 < spend_time < 1:
        print(f"The whole procedure takes {round(spend_time * 1000)} milliseconds.")
    else:
        print(f"The whole procedure takes {round(spend_time * 10**6)} microseconds.")


def batch_save(matches, path):
    """
    This function is to convert images with formats '.dm3', '.emd' files into '.tif' format.

    Args:

    matches: a list storing the dataset, it is achieved using the function 'find_files'

    path: a string, the file location to store the converted images
    """
    
    scalebar = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    for i in range (len(matches)):
        file = matches[i]
        name = os.path.splitext(os.path.basename(file))[0]
        image = hs.load(file)
        
        mode = tools.image_mode(image)
        parameters = tools.parameters_image(image, mode)
        resolution = parameters[0]
        pixelsize = parameters[1]
        unit = parameters[2]
        
        length_raw = round(0.4 + (resolution*pixelsize)/10)
        
        bar = []
        for scale in scalebar:
            bar.append(abs(length_raw - scale))
        length = scalebar[bar.index(min(bar))]  
        
        if len(image)==1:
                
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            save_with_scale_bar(mode, image, path, name+'.tiff', length, resolution, unit)
            
        elif len(image)>1:
            
            for j in range (len(image)):
                img_name = image[j].metadata.General.title
                if not isinstance(image[j], np.ndarray):
                    image[j] = np.array(image[j])
                    save_with_scale_bar(mode, image[j], path, name+"{:}".format(img_name)+'.tiff', length, resolution, unit)
                
        else: print(f'There is no image in {name}' )

def save_with_scale_bar(mode, image, path, name, length, resolution, unit):
    """
    Save images with scale bars.

    Args:
        mode (str): "Diffraction" or "Imaging".
        image (np.ndarray): The image data as a NumPy array.
        path (str): The location for storing the image.
        name (str): The name for the stored image.
        length (int): The length of the scale bar to be added to the image.
        resolution (float): The real length per pixel in the image.
        unit (str): The unit of the scale bar.
    """
    # Set default mode if not provided
    if mode is None:
        mode = "Imaging"
    
    # Process the image based on the mode
    if mode == "Diffraction":
        im = Image.fromarray(np.interp(image, (0, 80 * np.log(image.max())), (0, 65535)).astype(np.uint16))
    else:
        im = Image.fromarray(np.interp(image, (image.min(), image.max()), (0, 65535)).astype(np.uint16))
    
    draw = ImageDraw.Draw(im)
    image_height = image.shape[0]
    
    # Calculate scale bar dimensions
    scale_length = int(length / resolution)  # Length of the scale bar in pixels
    scale_width = int(image_height * 0.01)   # Width of the scale bar
    font_size = int(image_height * 0.08)     # Font size for the scale bar label
    
    # Position for the scale bar
    left = image_height * 0.05
    top = image_height * 0.95
    
    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", size=font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw scale bar and label
    draw.line((left, top, left + scale_length, top), fill=255, width=scale_width)
    draw.text((left - 20, top - image_height * 0.085), f"{length} {unit}", font=font, fill=255)
    
    # Save the image
    full_path = os.path.join(path, name)
    im.save(full_path, dpi=(600, 600))


def save_as_mrc(ndarray, resolution, unit, save_path, name):
    """
    Saving 2d-array into mrc file using the module "mrcfile"
    The unit of the pixel size is angstron (Å)
    Args:
    ndarray: matrix with 2D shape
    
    resolution: the pixe size in nm
    
    save_path: the path of mrc saved
    
    name: the file name, ending with '.mrc'
    """
    saving = os.path.join(save_path, name) 
        
    if type(ndarray[0,0])==np.float64:
        with mrcfile.new(saving, overwrite=True) as mrc:
            mrc.set_data(ndarray.astype(np.float32)) 
            vsize = mrc.voxel_size.copy()
            vsize.x = resolution
            vsize.y = resolution
            mrc.voxel_size = vsize
            mrc.add_label("Unit of scale bar: {unit}")
    else: 
        with mrcfile.new(saving, overwrite=True) as mrc:
            mrc.set_data(ndarray.astype(np.complex64)) 
            vsize = mrc.voxel_size.copy()
            vsize.x = resolution
            vsize.y = resolution
            mrc.voxel_size = vsize
            mrc.add_label("Unit of scale bar: {unit}")

def file_save(path_save, file_name):
    """
    check whether the file name is already existed or not
    """
    file_path = os.path.join(path_save, file_name)  
    if os.path.exists(file_path):
        counter = 1
        base_name, extension = os.path.splitext(file_name)
        while True:
            new_name = f"{base_name}{counter}{extension}"
            new_path = os.path.join(path_save, new_name)
            if not os.path.exists(new_path):
                print(f"The new file name is: {new_name} \n")
                return new_path
                break
            counter += 1              
    else: 
        return file_path

def saving_data_as_h5(OptimumFilters, filename, path_save, parameters, ab):
  """
  It is for saving the reconstructed filters into h5 files using h5py.

  Args:

  OptimumFilters: ndarrays, constructed by alogrithms

  filename and path_save: giving a name for the file and a targeted path for the saving.

  parameters and ab: dictionaries, recording the key informations about the filters.
  """
  data = h5py.File(file_save(path_save, filename), 'a')
  filters = data.create_group('frequency_filters')
  info = data.create_group("Experimental_parameters")
  string_dtype = h5py.special_dtype(vlen=str)  # Using special_dtype for variable-length strings
  for i in range(len(parameters)):
      element = (list(parameters.values())[i])
      if type(element)==str:
          info_input = info.create_dataset(list(parameters.keys())[i], shape=(1,), dtype = string_dtype)
          info_input[0] = element
      else:
          info.create_dataset(list(parameters.keys())[i], data = element, dtype = 'f')

  for i in range(len(ab)):
      element = (list(ab.values())[i])
      if type(element)==str:
          info_input = info.create_dataset(list(ab.keys())[i], shape=(1,), dtype = string_dtype)
          info_input[0] = element
      else:
          info.create_dataset(list(ab.keys())[i], data = element)

  for j in range (len(OptimumFilters)):
      subgrp_name = f"optimum filter {j+1}"
      filters.create_dataset(subgrp_name , data=OptimumFilters[j])

  data.close()

def transfer_data_into_h5(s, path, filename, close = True):
    """
    Saving the .emd datasets into .h5py file
    if the built h5py is not closed after data conversion, then the it returns an opened "data"
    """
    conditions = tools.information_data(s)
    
    data = h5py.File(file_save(path, filename), 'a')
    raw = data.create_group('raw datasets')
    info = data.create_group("Experimental_conditions")

    string_dtype = h5py.special_dtype(vlen=str)  # Using special_dtype for variable-length strings
    for k in range(len(conditions)):
        element = (list(conditions.values())[k])
        if type(element)==str:              
            info_input = info.create_dataset(list(conditions.keys())[k], shape=(1,), dtype = string_dtype)
            info_input[0] = element
        else:
            info.create_dataset(list(conditions.keys())[k], data = element, dtype = 'f')
            
    if len(s)==0:
        print(f'There is no image found !!!')
    elif len(s)==1:
        subgrp_name = s.metadata.General.title
        subgrp = raw.create_dataset(subgrp_name, data = s.data)
    else: 
        for i in range (len(s)):
            subgrp_name = s[i].metadata.General.title
            subgrp = raw.create_group(subgrp_name)
            if len(s[i])>1:
                for j in range(len(s[i].data)):
                    subgrp.create_dataset('num'+str(j+1), data=(s[i].data)[j])
            else:subgrp.create_dataset('num', data=(s[i].data))
    if close:
        data.close()
    else: return data