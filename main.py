#%% Imports

import napari
import numpy as np
from skimage import io 
from pathlib import Path

from functions import ranged_uint8

#%% Get raw name

raw_name = 'C1-2022.07.05_Luminy_22hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'

#%% Get path and open data

raw_path = Path(Path.cwd(), 'data', raw_name)
raw = io.imread(raw_path)
test = raw[9,...]
# test = ranged_uint8(raw[9,...], 0.1, 99.9)

#%%

import cv2

def gabor_filt(
        img,
        n_filters=16,
        kernel_size=15,
        sigma=3,
        lmbda=10,
        gamma=0.5,
        psi=0,
        ):

    # Create filters   
    filters = []    
    thetas = np.arange(0, np.pi, np.pi/n_filters)
    for theta in thetas:        
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size), 
            sigma, theta, lmbda, gamma, psi, 
            ktype=cv2.CV_64F
            )
        
        # kernel /= 1.0 * kernel.sum() # Brightness normalization
        filters.append(kernel)
        
    # Apply filters
    img_filt = np.zeros((n_filters, img.shape[0], img.shape[1]))
    for i, kernel in enumerate(filters):
        img_filt[i,...] = cv2.filter2D(img, -1, kernel)
    
    return filters, img_filt

filters, img_filt = gabor_filt(test)

viewer = napari.Viewer()
# viewer.add_image(np.array(filters))
viewer.add_image(img_filt)

#%%
# def apply_filter(img, filters):
# # This general function is designed to apply filters to our image
     
#     # First create a numpy array the same size as our input image
#     newimage = np.zeros_like(img)
     
#     # Starting with a blank image, we loop through the images and apply our Gabor Filter
#     # On each iteration, we take the highest value (super impose), until we have the max value across all filters
#     # The final image is returned
#     depth = -1 # remain depth same as original image
     
#     for kern in filters:  # Loop through the kernels in our GaborFilter
#         image_filter = cv2.filter2D(img, depth, kern)  #Apply filter to image
         
#         # Using Numpy.maximum to compare our filter and cumulative image, taking the higher value (max)
#         np.maximum(newimage, image_filter, newimage)
#     return newimage

# # We create our gabor filters, and then apply them to our image
# gfilters = create_gaborfilter()
# test_filt = apply_filter(test, gfilters)

#%%
# viewer = napari.Viewer()
# viewer.add_image(test)
# viewer.add_image(test_filt)
# viewer.grid.enabled = True