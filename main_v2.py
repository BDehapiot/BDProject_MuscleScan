#%% Imports

import cv2
import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
from joblib import Parallel, delayed


from tools.nan import nanfilt

#%% Get raw name

stack_name = 'C1-2022.07.05_Luminy_22hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_24hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_26hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_2_2.tif'
# stack_name = 'C1-2022.07.05_Luminy_28hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_30hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_32hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_1_1.tif'

#%% Get path and open data

data_path = Path(Path.cwd(), 'data')
stack_path = Path(data_path, stack_name)
stack = io.imread(stack_path)

img = stack[9,...]

#%% Parameters

# General
patch_size = 20
thresh_coeff = 0.75

# Gabor kernels
n_kernels = 16
kernel_size = patch_size
sigma = 4
lmbda = 8
gamma = 0.5
psi = 0

#%%

# -----------------------------------------------------------------------------

from skimage.filters import threshold_li

start = time.time()
print('get mask')

thresh = threshold_li(img, tolerance=1)
mask = img > thresh*thresh_coeff

end = time.time()
print(f'  {(end-start):5.3f} s') 

# -----------------------------------------------------------------------------

start = time.time()
print('gabor_filtering')

# Design gabor kernels
kernels = []    
thetas = np.arange(0, np.pi, np.pi/n_kernels)
for theta in thetas:        
    kernel = cv2.getGaborKernel(
        (kernel_size, kernel_size), 
        sigma, theta, lmbda, gamma, psi, 
        ktype=cv2.CV_64F
        )
    kernels.append(kernel)
    
# Apply gfilters
gfilt = np.zeros((n_kernels, img.shape[0], img.shape[1]))
for k, kernel in enumerate(kernels):
    gfilt[k,...] = cv2.filter2D(img, -1, kernel)
    
end = time.time()
print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

from skimage.morphology import skeletonize

start = time.time()
print('??? #1')
 
argmax = np.zeros((img.shape[0]//patch_size, img.shape[1]//patch_size))                 
for y, yi in enumerate(np.arange(0, img.shape[0], patch_size)):
    for x, xi in enumerate(np.arange(0, img.shape[1], patch_size)):
        
        patch_mask = mask[yi:yi+patch_size,xi:xi+patch_size]
        
        if np.mean(patch_mask) > 0.25:   
            patch_gfilt = gfilt[:,yi:yi+patch_size,xi:xi+patch_size]        
            argmax[y,x] = np.argmax(np.std(patch_gfilt, axis=(1,2)))            
        else:            
            argmax[y,x] = np.nan           

argmax = nanfilt(argmax, 5, 'mean')

gproj = np.zeros_like(img)
for y, yi in enumerate(np.arange(0, img.shape[0], patch_size)):
    for x, xi in enumerate(np.arange(0, img.shape[1], patch_size)):
        
        if not np.isnan(argmax[y,x]):
            
            k1 = int(np.floor(argmax[y,x]))
            k2 = int(np.ceil(argmax[y,x]))
            k1_coeff = np.abs(k2-argmax[y,x])
            k2_coeff = np.abs(k1-argmax[y,x])

            if k1==k2:
                
                gproj[yi:yi+patch_size,xi:xi+patch_size] = (
                    gfilt[k1,yi:yi+patch_size,xi:xi+patch_size]
                    )
                
            else:
                    
                gproj[yi:yi+patch_size,xi:xi+patch_size] = (
                    gfilt[k1,yi:yi+patch_size,xi:xi+patch_size]*k1_coeff +
                    gfilt[k2,yi:yi+patch_size,xi:xi+patch_size]*k2_coeff
                    )

        else:
            
            gproj[yi:yi+patch_size,xi:xi+patch_size] = 0
                
end = time.time()
print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

from skimage.morphology import remove_small_holes, remove_small_objects

start = time.time()
print('??? #2')

gthresh = threshold_li(gproj, tolerance=1)
gmask = gproj > gthresh*1.75 # adjust !!!

gmask = remove_small_holes(gmask, area_threshold=128)
gmask = remove_small_objects(gmask, min_size=128)

gskel = skeletonize(gmask)

end = time.time()
print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(img)
# viewer.add_image(mask)
# viewer.add_image(gfilt)
# viewer.add_image(argmax)
# viewer.add_image(gproj)  
viewer.add_image(gmask) 
viewer.add_image(gskel) 
# viewer.add_image(all_patches) 
# viewer.add_image(np.array(kernels))  
    