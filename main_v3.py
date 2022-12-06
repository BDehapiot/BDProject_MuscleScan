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

from skimage.morphology import square

start = time.time()
print('-2-')

selem = np.array([[1, 2, 3],
                  [8, 0, 4],
                  [7, 6, 5]])
shift = [
    (-1,-1), # 1
    (-1,0),  # 2
    (-1,1),  # 3
    (0,1),   # 4  
    (1,1),   # 5
    (1,0),   # 6
    (1,-1),  # 7
    (0,-1),  # 8
    ]

# Pad img & labels
img = np.pad(img, 
    pad_width=((1, 1), (1, 1)),
    mode='constant', constant_values=0)   
labels = np.pad(labels, 
    pad_width=((1, 1), (1, 1)),
    mode='constant', constant_values=0)   

#
conn2 = pixconn(labels>0, conn=2)
y_ends, x_ends = np.where(pixconn(labels>0, conn=2)==1)
l_ends = labels[y_ends,x_ends]; sort = np.argsort(l_ends)
y_ends = y_ends[sort]; x_ends = x_ends[sort]; l_ends = l_ends[sort]
y_seeds = np.take(y_ends, np.arange(0, len(y_ends), 2))
x_seeds = np.take(x_ends, np.arange(0, len(x_ends), 2))
l_seeds = np.take(l_ends, np.arange(0, len(l_ends), 2))


y,x = np.where(pixconn(labels>0, conn=2)==1)
l = labels[y,x]; endpoints = np.column_stack((l,y,x))
idx = np.lexsort((endpoints[:,2],endpoints[:,1],endpoints[:,0]))  
endpoints = endpoints[idx]
seeds = np.take(endpoints, np.arange(0, len(y_ends), 2), axis=0)


test_ridge = np.zeros_like(labels)

for i, label in enumerate(l_seeds):
        
    ridge = (labels==label).astype('uint8')
    y = y_seeds[i]; x = x_seeds[i]
    
    intensity = np.zeros((np.sum(ridge>0)))
    distance = np.zeros((np.sum(ridge>0)))
    for j in range(np.sum(ridge>0)):
        
        intensity[j] = img[y,x]
        
        connect = np.sum(ridge[y-1:y+2,x-1:x+2]*selem)
        
        if j == 0:          
            distance[j] = 0            
        else:            
            if connect % 2 == 0:              
                distance[j] = distance[j-1] + 1            
            else:                
                distance[j] = distance[j-1] + np.sqrt(2)
        
        test_ridge[y,x] = distance[j]
        
        ridge[y,x] = 0
        y = y + shift[connect-1][0]
        x = x + shift[connect-1][1]

        pass
        
        
    

# sort = np.argsort(idx)
# y = y[sort]
# x = x[sort]
# idx = idx[sort]
# y1 = np.take(y, np.arange(0, len(y), 2))
# x1 = np.take(x, np.arange(0, len(x), 2))
# idx1 = np.take(idx, np.arange(0, len(idx), 2))
# y2 = np.take(y, np.arange(1, len(y), 2))
# x2 = np.take(x, np.arange(1, len(x), 2))
# idx2 = np.take(idx, np.arange(1, len(idx), 2))

# # Project best orientation 
# idx = rwhere(mask, 1)
# idx = tuple(argmin[idx]) + idx 

# for label in range(1, np.max(labels)):

#     ridge = labels==label
    
    # rconn1 = pixconn(ridge, conn=1)
    # rconn2 = pixconn(ridge, conn=2)
    # endpoints_y, endpoints_x = np.where(rconn2==1)
    # rconn2[endpoints_y[1],endpoints_x[1]] = 2
    
    # rint = np.zeros(np.sum(ridge))
    # rconn = np.zeros(np.sum(ridge))
    # for i in range(np.sum(ridge)):
    
    #     y,x = np.where(rconn2==1)
    #     y = int(y)
    #     x = int(x)
        
    #     rint[i] = img[y,x]
    #     if rconn1[y,x] == 1:        
    #         rconn[i] = 1
    #     else:
    #         rconn[i] = 2
        
    #     rconn2[y-1:y+2,x-1:x+2] -= 1
        
    #     pass

# rseed_y = rseed_y[0] # Select first endpoint y coordinates
# rseed_x = rseed_x[0] # Select first endpoint x coordinates



# endpoints = np.where(ridge_conn==1)
# seed = (endpoints[0][0], endpoints[1][0])

end = time.time()
print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(img, contrast_limits=(0, 7500))
# viewer.add_image(mask)
# viewer.add_image(magnitude)
# viewer.add_image(orientation)
# viewer.add_image(np.array(kernels))
# viewer.add_image(gfilt1) 
# viewer.add_image(gfilt2)
# viewer.add_image(gfilt3)
# viewer.add_image(gmask)
# viewer.add_image(skel, blending='additive')

viewer.add_image(test_ridge)
# viewer.add_image(labels)
# viewer.add_image(conn2)

#%%

# from patchify import patchify, unpatchify
# from skimage.transform import resize

# start = time.time()
# print('temp#1')

# ny = img.shape[0]//patch_size
# nx = img.shape[1]//patch_size

# patch_orientation = np.reshape(
#     patchify(orientation, patch_size, patch_size), 
#     (ny*nx, patch_size, patch_size)
#     )

# orientation_med = np.nanmedian(patch_orientation, axis=(1,2))
# orientation_med = np.reshape(orientation_med, (ny,nx))
# orientation_med = nanfilt(orientation_med, 5, 'median')
# orientation_med = resize(orientation_med, (img.shape), order=0)

# end = time.time()
# print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------