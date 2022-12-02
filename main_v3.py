#%% Imports

import cv2
import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
from joblib import Parallel, delayed

from tools.nan import nanfilt, nanoutliers
from tools.idx import rwhere

#%% Get raw name

# stack_name = 'C1-2022.07.05_Luminy_22hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
stack_name = 'C1-2022.07.05_Luminy_24hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_26hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_2_2.tif'
# stack_name = 'C1-2022.07.05_Luminy_28hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_30hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_32hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_1_1.tif'

#%% Get path and open data

data_path = Path(Path.cwd(), 'data')
stack_path = Path(data_path, stack_name)
stack = io.imread(stack_path)

#%% Parameters

# General
patch_size = 20
thresh_coeff = 0.75

# Gabor kernels
n_kernels = 32
kernel_size = patch_size
sigma = 4
lmbda = 8
gamma = 0.5
psi = 0

#%%

img = stack[9,...] # extract img


# -----------------------------------------------------------------------------

from skimage.filters import threshold_li

start = time.time()
print('Get mask')

thresh = threshold_li(img, tolerance=1)
mask = img > thresh*thresh_coeff

end = time.time()
print(f'  {(end-start):5.3f} s') 

# -----------------------------------------------------------------------------

from skimage.filters import scharr_h, scharr_v, rank
from skimage.morphology import disk

start = time.time()
print('Get local angles')

# Get local angles (using scharr filter)
scharr_h = scharr_h(img); scharr_v = scharr_v(img)
angles = np.arctan2(scharr_h, scharr_v) * (180/np.pi) % 180
angles = rank.median(
    angles.astype('uint8'), 
    footprint=disk(patch_size),
    mask=mask,
    ).astype('float')
angles[mask==0] = np.nan

end = time.time()
print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

start = time.time()
print('Define gabor filters')

# Define gabor filters
kernels = []    
thetas = np.arange(0, np.pi, np.pi/n_kernels)
for theta in thetas:        
    kernel = cv2.getGaborKernel(
        (kernel_size, kernel_size), 
        sigma, theta, lmbda, gamma, psi, 
        ktype=cv2.CV_64F
        )
    kernels.append(kernel)

end = time.time()
print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

start = time.time()
print('Apply gabor filters')

def apply_gfilt(img, kernels, thetas, ):
   
    # Apply gabor filters
    gfilt = np.zeros((n_kernels, img.shape[0], img.shape[1]))
    for k, kernel in enumerate(kernels):
        gfilt[k,...] = cv2.filter2D(img, -1, kernel)
        
    # Find best angle    
    best_angle = np.zeros((len(thetas), angles.shape[0], angles.shape[1]))
    for i in range(len(thetas)):
        best_angle[i,...] = np.abs(angles - thetas[i]*180/np.pi) 
    argmin = np.argmin(best_angle, axis=0)
    
    # Project best angle 
    idx = rwhere(mask, 1)
    idx = tuple(argmin[idx]) + idx 
    proj = np.zeros_like(gfilt)
    proj[idx] = gfilt[idx]
    proj = np.max(proj, axis=0)
    
    return 
    
end = time.time()
print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(img)
# viewer.add_image(gfilt)
viewer.add_image(angles)
# viewer.add_image(best_angle)
viewer.add_image(proj)

#%%

# from patchify import patchify, unpatchify
# from skimage.transform import resize

# start = time.time()
# print('temp#1')

# ny = img.shape[0]//patch_size
# nx = img.shape[1]//patch_size

# patch_angles = np.reshape(
#     patchify(angles, patch_size, patch_size), 
#     (ny*nx, patch_size, patch_size)
#     )

# angles_med = np.nanmedian(patch_angles, axis=(1,2))
# angles_med = np.reshape(angles_med, (ny,nx))
# angles_med = nanfilt(angles_med, 5, 'median')
# angles_med = resize(angles_med, (img.shape), order=0)

# end = time.time()
# print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------