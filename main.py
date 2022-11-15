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

raw_name = 'C1-2022.07.05_Luminy_24hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'

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

thresh_coeff = 0.75
thresh = threshold_li(img) * thresh_coeff

# -----------------------------------------------------------------------------

start = time.time()
print('tile image')

roi_size = 10
n_rois = np.square(img.shape[0]//roi_size)
patched_local_mean = np.zeros_like(img)
patched_max_sd = np.zeros_like(img)
vectors = np.empty((n_rois, 2, 2))

count = -1

for yi in np.arange(0, img.shape[0], roi_size):
    for xi in np.arange(0, img.shape[1], roi_size):
        
        count += 1 
                
        # Get local mean (img)
        patch = img[yi:yi+roi_size,xi:xi+roi_size]  
        local_mean = np.mean(patch)
        
        if local_mean > thresh:
        
            # Get max sd (img_filt)
            patch_filt = img_filt[:,yi:yi+roi_size,xi:xi+roi_size]        
            max_sd = np.argmax(np.std(patch_filt, axis=(1,2)))
            
            # Create output images
            patched_local_mean[yi:yi+roi_size,xi:xi+roi_size] = local_mean
            patched_max_sd[yi:yi+roi_size,xi:xi+roi_size] = max_sd
            
            #
            vectors[count,0,0] = yi # y position
            vectors[count,0,1] = xi # x position
            vectors[count,1,0] = np.sin(thetas[max_sd] - np.pi/2) # x-y projection
            vectors[count,1,1] = np.cos(thetas[max_sd] - np.pi/2) # x-y projection      

end = time.time()
print(f'  {(end-start):5.3f} s')

# -----------------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(img)
viewer.add_image(patched_local_mean)
viewer.add_image(patched_max_sd)
viewer.add_vectors(vectors, length=roi_size//2)

# -----------------------------------------------------------------------------

# viewer = napari.Viewer()
# vector_data = [
# [[0, 10, 11],  # position of v0
#  [0,  1,  2]],  # projection of v0
# [[1, 20, 10],  # position of v1
#  [1,  3,  2]],  # projection of v1
# ]
# viewer.add_vectors(vector_data)

#%%

# io.imsave(
#     Path(data_path, raw_name.replace('.tif', '_tiles.tif')), 
#     np.array(tiles), check_contrast=False)

# io.imsave(
#     Path(data_path, raw_name.replace('.tif', '_filt.tif')), 
#     img_filt.astype('uint16'), check_contrast=False)


#%%

# import napari
# import numpy as np


# # create the viewer and window
# viewer = napari.Viewer()

# n = 20
# m = 40

# image = 0.2 * np.random.random((n, m)) + 0.5
# layer = viewer.add_image(image, contrast_limits=[0, 1], name='background')

# # sample vector image-like data
# # n x m grid of slanted lines
# # random data on the open interval (-1, 1)
# pos = np.zeros(shape=(n, m, 2), dtype=np.float32)
# rand1 = 2 * (np.random.random_sample(n * m) - 0.5)
# rand2 = 2 * (np.random.random_sample(n * m) - 0.5)

# # assign projections for each vector
# pos[:, :, 0] = rand1.reshape((n, m))
# pos[:, :, 1] = rand2.reshape((n, m))

# # add the vectors
# vect = viewer.add_vectors(pos, edge_width=0.2, length=2.5)

# print(image.shape, pos.shape)

# if __name__ == '__main__':
#     napari.run()
