#%% Imports

import cv2
import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
from joblib import Parallel, delayed

#%% Get raw name

stack_name = 'C1-2022.07.05_Luminy_22hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_24hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_26hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_2_2.tif'

#%% Get path and open data

data_path = Path(Path.cwd(), 'data')
stack_path = Path(data_path, stack_name)
stack = io.imread(stack_path)

#%% Parameters

# Gabor filters
n_filters=16
kernel_size=25
sigma=3
lmbda=10
gamma=0.25
psi=0
parallel=True

# Patch
patch_size = 20

#%% Functions

def gfilt(stack, n_filters, kernel_size, sigma, lmbda, gamma, psi, patch_size, parallel=False):
    
    # Nested functions --------------------------------------------------------
    
    def _gfilt(img):
        
        # Apply filters
        img_filt = np.zeros((n_filters, img.shape[0], img.shape[1]))
        for i, kernel in enumerate(filters):
            img_filt[i,...] = cv2.filter2D(img, -1, kernel)  

        return img_filt
    
    # Run ---------------------------------------------------------------------
    
    # Create filters  
    filters = []    
    thetas = np.arange(0, np.pi, np.pi/n_filters)
    for theta in thetas:        
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size), 
            sigma, theta, lmbda, gamma, psi, 
            ktype=cv2.CV_64F
            )
        kernel /= 1.0 * kernel.sum() # Brightness normalization
        filters.append(kernel)
        
    # Add one dimension (if ndim == 2)
    ndim = (stack.ndim)        
    if ndim == 2:
        stack = stack.reshape((1, stack.shape[0], stack.shape[1]))  
            
    if parallel:

        # Run parallel
        output_list = Parallel(n_jobs=-1)(
            delayed(_gfilt)(
                img,
                )
            for img in stack
            )
        
    else:
        
        # Run serial
        output_list = [_gfilt(
                img,
                )
            for img in stack
            ]
    
    return output_list
        
#%%

start = time.time()
print('gfilt')
    
output_list = gfilt(
    stack,
    n_filters,
    kernel_size,
    sigma,
    lmbda,
    gamma,
    psi,
    parallel
    )

end = time.time()
print(f'  {(end-start):5.3f} s')  

viewer = napari.Viewer()
viewer.add_image(output_list[0])  

#%%

img = stack[9,...]

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
    kernel /= 1.0 * kernel.sum() # Brightness normalization
    filters.append(kernel)

# Create filters   
filters = []    
thetas = np.arange(0, np.pi, np.pi/n_filters)
for theta in thetas:        
    kernel = cv2.getGaborKernel(
        (kernel_size, kernel_size), 
        sigma, theta, lmbda, gamma, psi, 
        ktype=cv2.CV_64F
        )
    kernel /= 1.0 * kernel.sum() # Brightness normalization
    kernel = kernel.reshape((1, kernel.shape[0], kernel.shape[1]))
    kernel = np.repeat(kernel, n_filters, axis=0)
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

thresh_coeff = 0.75
thresh = threshold_li(img) * thresh_coeff

# -----------------------------------------------------------------------------

start = time.time()
print('tile image')
           
patch_size = 20
n_patch = np.square(img.shape[0]//patch_size)
patched_local_mean = np.zeros_like(img)
patched_max_sd = np.zeros_like(img)
vectors = []
for yi in np.arange(0, img.shape[0], patch_size):
    for xi in np.arange(0, img.shape[1], patch_size):

        # Get local mean (img)
        patch = img[yi:yi+patch_size,xi:xi+patch_size]  
        local_mean = np.mean(patch)
             
        # Get max sd (img_filt)
        patch_filt = img_filt[:,yi:yi+patch_size,xi:xi+patch_size]        
        max_sd = np.argmax(np.std(patch_filt, axis=(1,2)))
        
        # Create output images
        patched_local_mean[yi:yi+patch_size,xi:xi+patch_size] = local_mean
        patched_max_sd[yi:yi+patch_size,xi:xi+patch_size] = max_sd
                
        # Create vectors
        temp = np.array([[yi, xi], [ 
            np.sin(thetas[max_sd]-np.pi/2),
            np.cos(thetas[max_sd]-np.pi/2)            
            ]])
        vectors.append(temp)

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

# viewer = napari.Viewer()
# viewer.add_image(img)
# viewer.add_image(img_filt)
# viewer.add_image(patched_local_mean)
# viewer.add_image(patched_max_sd)
# viewer.add_vectors(vectors, length=patch_size//2)

#%%

# io.imsave(
#     Path(data_path, raw_name.replace('.tif', '_tiles.tif')), 
#     np.array(tiles), check_contrast=False)

# io.imsave(
#     Path(data_path, raw_name.replace('.tif', '_filt.tif')), 
#     img_filt.astype('uint16'), check_contrast=False)