#%% Imports

import cv2
import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path

from skimage.morphology import disk
from skimage.filters.rank import gradient

from functions import ranged_uint8

#%% Get raw name

raw_name = 'C1-2022.07.05_Luminy_22hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'

#%% Get path and open data

data_path = Path(Path.cwd(), 'data')
raw_path = Path(data_path, raw_name)
raw = io.imread(raw_path)
img = raw[9,...]

#%%

n_filters=16
kernel_size=25
sigma=3
lmbda=10
gamma=0.5
psi=0

start = time.time()
print('Create filters')

# Create filters   
filters = []    
thetas = np.arange(0, np.pi, np.pi/n_filters)
for theta in thetas:        
    kernel = cv2.getGaborKernel(
        (kernel_size, kernel_size), 
        sigma, theta, lmbda, gamma, psi, 
        ktype=cv2.CV_64F
        )
    filters.append(kernel)

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('Apply filters')

# Apply filters and get orientations
img_filt = np.zeros((n_filters, img.shape[0], img.shape[1]))
for i, kernel in enumerate(filters):
    img_filt[i,...] = cv2.filter2D(img, -1, kernel)  

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

from skimage.filters import threshold_li
thresh_coeff = 0.5
thresh = threshold_li(img) * thresh_coeff

start = time.time()
print('tile image')

roi_size = 20
patched_local_mean = np.zeros_like(img)
patched_max_sd = np.zeros_like(img)
for yi in np.arange(0, img.shape[0], roi_size):
    for xi in np.arange(0, img.shape[1], roi_size):
        
        # Get idx
        idx = (
            np.repeat(np.arange(yi,yi+roi_size), roi_size),
            np.tile(np.arange(xi,xi+roi_size), roi_size)
            )
        
        # idx = np.ravel_multi_index((
        #     np.repeat(np.arange(yi,yi+roi_size), roi_size),
        #     np.tile(np.arange(xi,xi+roi_size), roi_size), 
        #     ), (img.shape))            
        
        # Get local mean (img)
        patch = img[yi:yi+roi_size,xi:xi+roi_size]  
        local_mean = np.mean(patch)
        
        # Get max sd (img_filt)
        patch_filt = img_filt[:,yi:yi+roi_size,xi:xi+roi_size]        
        max_sd = np.argmax(np.std(patch_filt, axis=(1,2)))
        
        # Create output images
        patched_local_mean[yi:yi+roi_size,xi:xi+roi_size] = local_mean
        patched_max_sd[yi:yi+roi_size,xi:xi+roi_size] = max_sd

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------


# viewer = napari.Viewer()
# viewer.add_image(patch_mean)
# viewer.add_image(patch_filt_sd)
# viewer.add_image(np.array(tiles))
# viewer.add_image(np.array(filters))
# viewer.add_image(img_filt)
# viewer.add_image(img_grad)
# viewer.add_image(img_grad_max)
# viewer.add_image(img_untiled)

#%%

# io.imsave(
#     Path(data_path, raw_name.replace('.tif', '_tiles.tif')), 
#     np.array(tiles), check_contrast=False)

# io.imsave(
#     Path(data_path, raw_name.replace('.tif', '_filt.tif')), 
#     img_filt.astype('uint16'), check_contrast=False)


#%%
