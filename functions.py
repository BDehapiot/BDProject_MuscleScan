#%% Imports

import cv2
import numpy as np
from skimage.morphology import disk
from skimage.transform import rescale
from skimage.filters import threshold_li
from skimage.filters import scharr_h, scharr_v, rank
from skimage.morphology import remove_small_holes, remove_small_objects

from BDTools import dtype, nan, skel

#%% functions

def get_mask(img, thresh_coeff):

    thresh = threshold_li(img, tolerance=1)
    mask = img > thresh*thresh_coeff
    mask = remove_small_holes(mask, area_threshold=512)
    mask = remove_small_objects(mask, min_size=512)
    
    return mask

def get_orientations(
        img, 
        mask, 
        rescale_factor = 0.5,
        min_magnitude = 25,
        smooth_method='median', 
        smooth_kernel_size=64): 

    # Rescale img & mask
    img = rescale(img, rescale_factor, preserve_range=True)
    mask = rescale(mask, rescale_factor, order=0)
    
    # Extract local gradient
    gradient_h = scharr_h(img); gradient_v = scharr_v(img)
    magnitudes = np.sqrt((gradient_h ** 2) + (gradient_v ** 2))
    orientations = np.arctan2(gradient_h, gradient_v) * (180/np.pi) % 180
    
    # Remove low magnitude orientations
    orientations[magnitudes<min_magnitude] = np.nan
    orientations = nan.replace(orientations, 3, 'median', mask=mask)
    orientations = nan.replace(orientations, 3, 'median', mask=mask)
    orientations = nan.replace(orientations, 3, 'median', mask=mask)
    
    # Smooth orientations map    
    smooth_function = {
        'mean': rank.mean, 
        'median': rank.median, 
        'modal': rank.modal, 
        } 
    orientations = smooth_function[smooth_method](
        orientations.astype('uint8'), 
        footprint=disk(smooth_kernel_size*rescale_factor),
        mask=mask,
        ).astype('float')
    
    magnitudes[mask==0] = np.nan
    orientations[mask==0] = np.nan
    
    orientations = rescale(orientations, 1/rescale_factor, preserve_range=True)
    
    return orientations

def get_gabor(img, mask, orientations, n_kernels=16, sigma=3):
           
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
    idx = np.where(mask == 1)
    idx = tuple(best_filter[idx]) + idx 
    gabor = np.zeros_like(all_filters)
    gabor[idx] = all_filters[idx]
    gabor = np.max(gabor, axis=0)
    
    return gabor

#%% tasks

def process_gabor(img, thresh_coeff, n_kernels=16, sigma=3):
    
    # Get mask
    mask = get_mask(
        img, thresh_coeff
        )
    
    # Get orientations
    orientations = get_orientations(
        img, mask, 
        rescale_factor = 0.5,
        min_magnitude = 25,
        smooth_method='median', 
        smooth_kernel_size=32
        )
    
    # Gabor filtering (3 consecutive rounds with increasing sigmas)
    gabor = get_gabor(img, mask, orientations,
        n_kernels=n_kernels, sigma=sigma)
    gabor = get_gabor(gabor, mask, orientations,
        n_kernels=n_kernels, sigma=sigma+1)
    gabor = get_gabor(gabor, mask, orientations,
        n_kernels=n_kernels, sigma=sigma+2)
    
    return mask, orientations, gabor
    