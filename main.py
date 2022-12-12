#%% Imports

import cv2
import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
from aicsimageio.readers import CziReader
from joblib import Parallel, delayed

from dtype import ranged_conversion
from nan import nanfilt, nanreplace
from skel import pixconn

#%% Hyperstack name

hstack_name = '2022.10.31_22hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_24hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_26hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_28hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_30hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_32hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'
# hstack_name = '2022.10.31_48hrsAPF_Luminy_phallo568_405nano2_488_nano42_647nano62_100Xz2.5_1_1.czi'

#%% Parameters

# Get orientations
ridge_channel = 1
thresh_coeff = 0.75
rsize_factor = 0.5
min_magnitude = 0.05
smooth_size = 32

# Get gabor
n_kernels = 16
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

# Extract image for running tests
zstack = hstack[ridge_channel,...]
img = zstack[zstack.shape[0]//2,...]

#%%

from skimage.morphology import disk
from skimage.transform import rescale
from skimage.morphology import binary_dilation
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.filters import threshold_li, scharr_h, scharr_v, gaussian, rank

start = time.time()
print('get_orientations')

# Initialize
smooth_size = int(smooth_size*rsize_factor)

# Resize & process img
rsize = rescale(img, rsize_factor, preserve_range=True)
rsize = ranged_conversion(
    rsize, intensity_range=(5,95), spread=3, dtype='float')

# Get mask
mask = rsize > threshold_li(rsize, tolerance=1)*thresh_coeff
mask = remove_small_holes(mask, area_threshold=rsize.shape[0])
mask = remove_small_objects(mask, min_size=rsize.shape[0])

# Extract local gradient
gradient_h = scharr_h(rsize); gradient_v = scharr_v(rsize)
magnitudes = np.sqrt((gradient_h ** 2) + (gradient_v ** 2))
orientations = np.arctan2(gradient_h, gradient_v) * (180/np.pi) % 180

# Remove low magnitude orientations
orientations[magnitudes < min_magnitude] = np.nan
orientations[mask == False] = np.nan
orientations = nanreplace(
    orientations, kernel_size=smooth_size, method='median', mask=mask)

# Smooth orientations map   
orientations = rank.median(
    orientations.astype('uint8'), 
    footprint=disk(smooth_size),
    mask=mask,
    ).astype('float')
orientations[mask == False] = np.nan

# Rescale and remove out of mask signal
magnitudes = rescale(magnitudes, 1/rsize_factor, preserve_range=True)
orientations = rescale(orientations, 1/rsize_factor, preserve_range=True)

end = time.time()
print(f'  {(end-start):5.3f} s')  

#%%

def get_gabor(img, orientations, n_kernels=16, sigma=3):
           
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
    mask = ~np.isnan(orientations)
    idx = np.where(mask == True)
    idx = (best_filter[idx], idx[0], idx[1]) 
    gabor = np.zeros_like(all_filters)
    gabor[idx] = all_filters[idx]
    gabor = np.max(gabor, axis=0)
    
    return gabor

start = time.time()
print('get_gabor')

gabor = get_gabor(img, orientations,
    n_kernels=n_kernels, sigma=sigma)
gabor = get_gabor(gabor, orientations,
    n_kernels=n_kernels, sigma=sigma+1)
gabor = get_gabor(gabor, orientations,
    n_kernels=n_kernels, sigma=sigma+2)

end = time.time()
print(f'  {(end-start):5.3f} s') 

#%%

from skimage.morphology import skeletonize, label

start = time.time()
print('get skeleton')

gthresh = threshold_li(gabor, tolerance=1)
gmask = gabor > gthresh*3 # adjust !!!
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
print('Measure along skeleton')

# Initialize
connect_kernel = np.array(
    [[1, 2, 3],
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

#%%

viewer = napari.Viewer()
viewer.add_image(img)
# viewer.add_image(rsize)
# viewer.add_image(mask)
# viewer.add_image(magnitudes)
# viewer.add_image(orientations)
viewer.add_image(gabor)
viewer.add_image(skel)
viewer.add_image(labels)
