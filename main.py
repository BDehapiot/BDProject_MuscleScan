#%% Imports

import cv2
import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
from joblib import Parallel, delayed
from skimage.transform import resize
from skimage.filters import threshold_li
from skimage.morphology import skeletonize

from tools.nan import nanfilt

#%% Get raw name

# stack_name = 'C1-2022.07.05_Luminy_22hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
stack_name = 'C1-2022.07.05_Luminy_24hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_26hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_2_2.tif'

#%% Get path and open data

data_path = Path(Path.cwd(), 'data')
stack_path = Path(data_path, stack_name)
stack = io.imread(stack_path)

stack = stack[9,...]

#%% Parameters

# Gabor filtering
n_filters=16
kernel_size=100
sigma=4
lmbda=10
gamma=0.25
psi=0
parallel=False

# Patch
patch_size = 20
thresh_coeff = 2

#%% Functions: gabor_filtering

def gabor_filtering(stack, n_filters, kernel_size, sigma, lmbda, gamma, psi, parallel=False):
    
    # Nested functions --------------------------------------------------------
    
    def _gabor_filtering(img):
        
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
            delayed(_gabor_filtering)(
                img,
                )
            for img in stack
            )
        
    else:
        
        # Run serial
        output_list = [_gabor_filtering(
                img,
                )
            for img in stack
            ]
    
    # Extract outputs
    stack_filt = np.stack([data for data in output_list], axis=0).squeeze()
    
    return stack_filt

#%% Functions: patch

def patch_analysis(stack, stack_filt, patch_size, thresh_coeff):
    
    # Nested functions --------------------------------------------------------
    
    def _patch_analysis(img, img_filt):
        
        local_mean = np.zeros((img.shape[0]//patch_size, img.shape[1]//patch_size))
        local_argmax = np.zeros((img.shape[0]//patch_size, img.shape[1]//patch_size))                 
        
        for y, yi in enumerate(np.arange(0, img.shape[0], patch_size)):
            for x, xi in enumerate(np.arange(0, img.shape[1], patch_size)):
                
                # Get local mean (img)
                patch = img[yi:yi+patch_size,xi:xi+patch_size]  
                local_mean[y,x] = np.mean(patch)
                     
                # Get local angle (img_filt)
                patch_filt = img_filt[:,yi:yi+patch_size,xi:xi+patch_size]        
                local_argmax[y,x] = np.argmax(np.std(patch_filt, axis=(1,2)))
                
        # Process local angle         
        thresh = threshold_li(stack) * thresh_coeff
        local_argmax[local_mean<thresh] = np.nan
        local_argmax = nanfilt(local_argmax.astype('float'), 3, 'median')
        
        # Get vectors
        
                             
        return local_mean, local_argmax
    
    # Run ---------------------------------------------------------------------

    # Add one dimension (if ndim == 2)
    ndim = (stack.ndim)        
    if ndim == 2:
        stack = stack.reshape((1, stack.shape[0], stack.shape[1]))  
        stack_filt = stack_filt.reshape((1, stack_filt.shape[0], stack_filt.shape[1])) 

    if parallel:

        # Run parallel
        output_list = Parallel(n_jobs=-1)(
            delayed(_patch_analysis)(
                img,
                img_filt,
                )
            for img, img_filt in zip(stack, stack_filt)
            )
        
    else:
        
        # Run serial
        output_list = [_patch_analysis(
                img,
                img_filt,
                )
            for img, img_filt in zip(stack, stack_filt)
            ]
        
    # Extract outputs
    local_mean = np.stack([data[0] for data in output_list], axis=0).squeeze()    
    local_argmax = np.stack([data[1] for data in output_list], axis=0).squeeze()     
    
    return local_mean, local_argmax
        
#%%

start = time.time()
print('gabor_filtering')
    
stack_filt = gabor_filtering(
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

# -----------------------------------------------------------------------------

# start = time.time()
# print('patch_analysis')
    
# local_mean, local_argmax = patch_analysis(
#     stack,
#     stack_filt,
#     patch_size,
#     thresh_coeff,
#     )

# end = time.time()
# print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

# viewer = napari.Viewer()
# viewer.add_image(local_mean)
# viewer.add_image(local_argmax)    

#%%

thresh_coeff = 0.75
thresh = threshold_li(stack) * thresh_coeff

# -----------------------------------------------------------------------------

thetas = np.arange(0, np.pi, np.pi/n_filters)
img = stack
img_filt = stack_filt

start = time.time()
print('patch_analysis')
          
local_mean = np.zeros((img.shape[0]//patch_size, img.shape[1]//patch_size))
local_argmax = np.zeros((img.shape[0]//patch_size, img.shape[1]//patch_size))                 
for y, yi in enumerate(np.arange(0, img.shape[0], patch_size)):
    for x, xi in enumerate(np.arange(0, img.shape[1], patch_size)):
        
        # Get local mean (img)
        patch = img[yi:yi+patch_size,xi:xi+patch_size]  
        local_mean[y,x] = np.mean(patch)
             
        # Get max sd (img_filt)
        patch_filt = img_filt[:,yi:yi+patch_size,xi:xi+patch_size]        
        local_argmax[y,x] = np.argmax(np.std(patch_filt, axis=(1,2)))
        
# Process local angle         
thresh = threshold_li(stack) * thresh_coeff
local_argmax[local_mean<thresh] = np.nan
local_argmax = nanfilt(local_argmax.astype('float'), 5, 'mean')

# Get vectors 
vectors = []
for y, yi in enumerate(np.arange(0, img.shape[0], patch_size)):
    for x, xi in enumerate(np.arange(0, img.shape[1], patch_size)):
        
        if local_mean[y,x] > thresh:
            vectors.append(np.array([[yi, xi], [
                np.sin(thetas[int(local_argmax[y,x])]-np.pi/2),
                np.cos(thetas[int(local_argmax[y,x])]-np.pi/2)            
                ]]))
        else:
            vectors.append(np.array([[yi, xi], [0,0]]))      

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('??? Proj')
        
proj_filt = np.zeros((img.shape[0], img.shape[1]))
for y, yi in enumerate(np.arange(0, img.shape[0], patch_size)):
    for x, xi in enumerate(np.arange(0, img.shape[1], patch_size)):
        
        if not np.isnan(local_argmax[y,x]):
        
            z1 = int(np.floor(local_argmax[y,x]))
            z2 = int(np.ceil(local_argmax[y,x]))
            z1_coeff = np.abs(z2-local_argmax[y,x])
            z2_coeff = np.abs(z1-local_argmax[y,x])
        
            if z1==z2:
                
                proj_filt[yi:yi+patch_size,xi:xi+patch_size] = (
                    img_filt[z1,yi:yi+patch_size,xi:xi+patch_size]
                    )
                
            else:
                    
                proj_filt[yi:yi+patch_size,xi:xi+patch_size] = (
                    img_filt[z1,yi:yi+patch_size,xi:xi+patch_size]*z1_coeff +
                    img_filt[z2,yi:yi+patch_size,xi:xi+patch_size]*z2_coeff
                    )
        
        else:
            
            proj_filt[yi:yi+patch_size,xi:xi+patch_size] = 0

        
end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('??? Proj')

# Get mask
thresh = threshold_li(proj_filt, tolerance=1)
mask = proj_filt > thresh*3
skel = skeletonize(mask)

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

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


viewer = napari.Viewer()
# viewer.add_image(np.array(filters))
viewer.add_image(img)
# viewer.add_image(mask)
# viewer.add_image(skel, blending='additive')
viewer.add_image(proj_filt)
# viewer.add_image(img_filt)
# viewer.add_image(local_mean)
# viewer.add_image(local_argmax)
# viewer.add_vectors(vectors, length=patch_size//2)

#%%

# io.imsave(
#     Path(data_path, raw_name.replace('.tif', '_tiles.tif')), 
#     np.array(tiles), check_contrast=False)

# io.imsave(
#     Path(data_path, raw_name.replace('.tif', '_filt.tif')), 
#     img_filt.astype('uint16'), check_contrast=False)