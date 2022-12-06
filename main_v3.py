#%% Imports

import cv2
import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
from joblib import Parallel, delayed

from tools.idx import rwhere
from tools.conn import pixconn

#%% Get raw name

# stack_name = 'C1-2022.07.05_Luminy_22hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_24hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_26hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_2_2.tif'
stack_name = 'C1-2022.07.05_Luminy_28hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_1_1.tif'
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
sigma = 3

#%%

img = stack[9,...] # extract img

# -----------------------------------------------------------------------------

from skimage.filters import threshold_li
from skimage.morphology import remove_small_holes, remove_small_objects

start = time.time()
print('Get mask')

thresh = threshold_li(img, tolerance=1)
mask = img > thresh*thresh_coeff
mask = remove_small_holes(mask, area_threshold=512)
mask = remove_small_objects(mask, min_size=512)

end = time.time()
print(f'  {(end-start):5.3f} s') 

# -----------------------------------------------------------------------------

from skimage.filters import scharr_h, scharr_v, rank
from skimage.morphology import disk

start = time.time()
print('Get local orientation')

def local_gradient(img, mask): 
    
    gradient_h = scharr_h(img); gradient_v = scharr_v(img)
    magnitude = np.sqrt((gradient_h ** 2) + (gradient_v ** 2))
    orientation = np.arctan2(gradient_h, gradient_v) * (180/np.pi) % 180
    orientation = rank.median(
        orientation.astype('uint8'), 
        footprint=disk(patch_size*2),
        mask=mask,
        ).astype('float')
    orientation[mask==0] = np.nan
    
    return magnitude, orientation

end = time.time()
print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

start = time.time()
print('Apply gabor filters')

def apply_gabor_filters(img, orientation, n_kernels=32, sigma=4):
           
    # Get gabor parameters
    kernel_size = sigma*10
    lmbda = sigma*3
    gamma = 0.5
    psi = 0
    
    # Get gabor filters
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
    all_filt = np.zeros((n_kernels, img.shape[0], img.shape[1]))
    for k, kernel in enumerate(kernels):
        all_filt[k,...] = cv2.filter2D(img, -1, kernel)
        
    # Find best orientation 
    thetas = np.arange(0, np.pi, np.pi/len(kernels)) 
    best_orientation = np.zeros((len(thetas), orientation.shape[0], orientation.shape[1]))
    for i in range(len(thetas)):
        best_orientation[i,...] = np.abs(orientation - thetas[i]*180/np.pi) 
    argmin = np.argmin(best_orientation, axis=0)
    
    # Project best orientation 
    idx = rwhere(mask, 1)
    idx = tuple(argmin[idx]) + idx 
    gfilt = np.zeros_like(all_filt)
    gfilt[idx] = all_filt[idx]
    gfilt = np.max(gfilt, axis=0)
    
    return gfilt

end = time.time()
print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

magnitude, orientation = local_gradient(img, mask)
gfilt1 = apply_gabor_filters(img, orientation, n_kernels=n_kernels, sigma=sigma)

# magnitude, orientation = local_gradient(gfilt1, mask)
gfilt2 = apply_gabor_filters(gfilt1, orientation, n_kernels=n_kernels, sigma=sigma+1)

# magnitude, orientation = local_gradient(gfilt2, mask)
gfilt3 = apply_gabor_filters(gfilt2, orientation, n_kernels=n_kernels, sigma=sigma+2)

#%% ---------------------------------------------------------------------------

from skimage.morphology import skeletonize, label

start = time.time()
print('-1-')

gthresh = threshold_li(gfilt3, tolerance=1)
gmask = gfilt3 > gthresh*3 # adjust !!!
gmask = remove_small_holes(gmask, area_threshold=128)
gmask = remove_small_objects(gmask, min_size=128)
skel = skeletonize(gmask)
skel = skel ^ (pixconn(skel, conn=2) > 2)
skel = remove_small_objects(skel, min_size=32, connectivity=2)
labels = label(skel, connectivity=2)

end = time.time()
print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

start = time.time()
print('-2-')

# Initialize
connect_kernel = np.array([[1, 2, 3],
                  [8, 0, 4],
                  [7, 6, 5]])

y_shift = [-1, -1, -1, 0, 1, 1, 1, 0]
x_shift = [-1, 0, 1, 1, 1, 0, -1, -1]

# Pad img & labels
img = np.pad(img, 
    pad_width=((1, 1), (1, 1)),
    mode='constant', constant_values=0)   
labels = np.pad(labels, 
    pad_width=((1, 1), (1, 1)),
    mode='constant', constant_values=0)   

# Extract seeds (top left pixel for every ridge) 
y,x = np.where(pixconn(labels>0, conn=2)==1)
l = labels[y,x]; endpoints = np.column_stack((l,y,x))
idx = np.lexsort((endpoints[:,2],endpoints[:,1],endpoints[:,0]))  
seeds = np.take(endpoints[idx], np.arange(0, len(endpoints), 2), axis=0)

ridge_data = []
for i in range(np.max(labels)):
        
    # Isolate ridge and coordinates
    ridge = (labels==i+1).astype('uint8')
    y = seeds[i,1]; x = seeds[i,2]

    # Ridge intensity/distance profiles
    distance = np.zeros((np.sum(ridge>0)))
    intensity = np.zeros((np.sum(ridge>0)))
    for j in range(np.sum(ridge>0)):
        
        intensity[j] = img[y,x]
        
        connect = np.sum(ridge[y-1:y+2,x-1:x+2]*connect_kernel)        
        if j == 0:          
            distance[j] = 0            
        else:            
            if connect % 2 == 0:              
                distance[j] = distance[j-1] + 1            
            else:                
                distance[j] = distance[j-1] + np.sqrt(2)
        
        ridge[y,x] = 0
        y = y + y_shift[connect-1] 
        x = x + x_shift[connect-1]   
        
    # Append ridge data
    ridge_data.append((distance, intensity))

end = time.time()
print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

# viewer = napari.Viewer()
# viewer.add_image(img, contrast_limits=(0, 7500))
# viewer.add_image(mask)
# viewer.add_image(magnitude)
# viewer.add_image(orientation)
# viewer.add_image(np.array(kernels))
# viewer.add_image(gfilt1) 
# viewer.add_image(gfilt2)
# viewer.add_image(gfilt3)
# viewer.add_image(gmask)
# viewer.add_image(skel, blending='additive')
