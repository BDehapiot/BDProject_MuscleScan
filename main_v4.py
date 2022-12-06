#%% Imports

import cv2
import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
from aicsimageio.readers import CziReader
from joblib import Parallel, delayed

from functions import get_mask, process_gabor
from tools.idx import rwhere
from tools.conn import pixconn
from tools.nan import nanfilt, nanreplace
from tools.dtype import ranged_uint8

#%% Hyperstack name

# hstack_name = '2022.10.31_22hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_24hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_26hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
hstack_name = '2022.10.31_28hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_30hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_32hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_48hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'

#%% Parameters

ridge_channel = 1 # channel to consider for ridge detection (first channel = 0)
thresh_coeff = 1
n_kernels = 32
sigma = 3

#%% Open czi file

hstack_path = Path(Path(Path.cwd(), 'data'), hstack_name)
hstack_czi = CziReader(hstack_path)
hstack = hstack_czi.data.squeeze()

# Get voxel size (µm)
vox_size = (
    hstack_czi.physical_pixel_sizes.Y,
    hstack_czi.physical_pixel_sizes.X,
    hstack_czi.physical_pixel_sizes.Z
    )

#%% 

zstack = hstack[ridge_channel,...]
img = zstack[zstack.shape[0]//2,...]

#%%

from skimage.filters import threshold_li
from skimage.morphology import remove_small_holes, remove_small_objects

start = time.time()
print('get_mask')

thresh = threshold_li(img, tolerance=1)
mask = img > thresh*thresh_coeff
mask = remove_small_holes(mask, area_threshold=512)
mask = remove_small_objects(mask, min_size=512)

end = time.time()
print(f'  {(end-start):5.3f} s')  

#%%

from skimage.morphology import disk
from skimage.transform import rescale
from skimage.filters import scharr_h, scharr_v, rank

rescale_factor = 0.5
min_magnitude = 25
smooth_method = 'median'
smooth_kernel_size = 64

start = time.time()
print('get_orientations')

# Rescale img & mask
img_rescale = ranged_uint8(img, percent_low=0.1, percent_high=99.9)
img_rescale = rescale(img_rescale, rescale_factor, preserve_range=True)
mask_rescale = rescale(mask, rescale_factor, order=0)

# Extract local gradient
gradient_h = scharr_h(img_rescale); gradient_v = scharr_v(img_rescale)
magnitudes = np.sqrt((gradient_h ** 2) + (gradient_v ** 2))
orientations = np.arctan2(gradient_h, gradient_v) * (180/np.pi) % 180

# Remove low magnitude orientations
orientations[magnitudes<min_magnitude] = np.nan
orientations = nanreplace(orientations, 3, 'median', mask=mask_rescale)
orientations = nanreplace(orientations, 3, 'median', mask=mask_rescale)
orientations = nanreplace(orientations, 3, 'median', mask=mask_rescale)

# Smooth orientations map    
smooth_function = {
    'mean': rank.mean, 
    'median': rank.median, 
    'modal': rank.modal, 
    } 
orientations = smooth_function[smooth_method](
    orientations.astype('uint8'), 
    footprint=disk(smooth_kernel_size*rescale_factor),
    mask=mask_rescale,
    ).astype('float')

magnitudes[mask_rescale==0] = np.nan
orientations[mask_rescale==0] = np.nan

orientations = rescale(orientations, 1/rescale_factor, preserve_range=True)

end = time.time()
print(f'  {(end-start):5.3f} s')  

#%%

def get_gabor(img, mask, orientations, n_kernels=16, sigma=3):
           
    # Set gabor parameters (according to sigma)
    kernel_size = sigma*10
    lmbda = sigma*3
    gamma = 0.5
    psi = 0
    
    # Design gabor filters
    kernels = [] 
    thetas = np.arange(0, np.pi, np.pi/n_kernels)
    for theta in thetas:        
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size), 
            sigma, theta, lmbda, gamma, psi, 
            ktype=cv2.CV_64F
            )
        kernels.append(kernel)
    
    # Apply all gabor filters
    all_filters = np.zeros((n_kernels, img.shape[0], img.shape[1]))
    for k, kernel in enumerate(kernels):
        all_filters[k,...] = cv2.filter2D(img, -1, kernel)
        
    # Find best filter (according to orientations)
    best_filter = np.zeros((
        len(thetas), 
        orientations.shape[0],
        orientations.shape[1]
        ))
    for i in range(len(thetas)):
        best_filter[i,...] = np.abs(orientations - thetas[i]*180/np.pi) 
    best_filter = np.argmin(best_filter, axis=0)
    
    # Project best filter only
    idx = rwhere(mask, 1)
    idx = tuple(best_filter[idx]) + idx 
    gabor = np.zeros_like(all_filters)
    gabor[idx] = all_filters[idx]
    gabor = np.max(gabor, axis=0)
    
    return gabor

start = time.time()
print('get_gabor')

gabor = get_gabor(img, mask, orientations,
    n_kernels=n_kernels, sigma=sigma)
gabor = get_gabor(gabor, mask, orientations,
    n_kernels=n_kernels, sigma=sigma+1)
gabor = get_gabor(gabor, mask, orientations,
    n_kernels=n_kernels, sigma=sigma+2)

end = time.time()
print(f'  {(end-start):5.3f} s') 

#%%

viewer = napari.Viewer()
viewer.add_image(img)
# viewer.add_image(mask)
# viewer.add_image(orientations)
viewer.add_image(gabor)