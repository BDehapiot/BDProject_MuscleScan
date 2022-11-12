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

img = raw[9,...]
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
# img_grad = np.zeros((n_filters, img.shape[0], img.shape[1]))
for i, kernel in enumerate(filters):
    img_filt[i,...] = cv2.filter2D(img, -1, kernel)
#     img_grad[i,...] = gradient(
#         ranged_uint8(img_filt[i,...], 0.1, 99.9), disk(lmbda))    
# img_grad_max = np.argmax(img_grad, axis=0)       

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('tile image')

img_tiled = []
for temp in np.split(img, 20, axis=0):
    for temp in np.split(temp, 20, axis=1):
        img_tiled.append(temp)
        
img_tiled = np.array(img_tiled)        

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

start = time.time()
print('untile image')

img_untitled = np.zeros_like(img)
for temp in 

img_untiled = np.reshape(img_tiled, (img.shape))  

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

viewer = napari.Viewer()
# viewer.add_image(np.array(tiles))
# viewer.add_image(np.array(filters))
# viewer.add_image(img_filt)
# viewer.add_image(img_grad)
# viewer.add_image(img_grad_max)
viewer.add_image(img_untiled)

#%%

# io.imsave(
#     Path(data_path, raw_name.replace('.tif', '_tiles.tif')), 
#     np.array(tiles), check_contrast=False)

# io.imsave(
#     Path(data_path, raw_name.replace('.tif', '_filt.tif')), 
#     img_filt.astype('uint16'), check_contrast=False)


#%%
