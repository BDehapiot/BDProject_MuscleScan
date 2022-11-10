#%% Imports

import numpy as np

#%% Functions

def ranged_uint8(img, percent_low=1, percent_high=99):

    """ 
    Convert image to uint8 using a percentile range.
    
    Parameters
    ----------
    img : ndarray
        Image to be converted.
        
    percent_low : float
        Percentile to discard low values.
        Between 0 and 100 inclusive.
        
    percent_high : float
        Percentile to discard high values.
        Between 0 and 100 inclusive.
    
    Returns
    -------  
    img : ndarray
        Converted image.
    
    """

    # Get data type 
    data_type = (img.dtype).name
    
    if data_type == 'uint8':
        
        raise ValueError('Input image is already uint8') 
        
    else:
        
        # Get data intensity range
        int_min = np.percentile(img, percent_low)
        int_max = np.percentile(img, percent_high) 
        
        # Rescale data
        img[img<int_min] = int_min
        img[img>int_max] = int_max 
        img = (img - int_min)/(int_max - int_min)
        img = (img*255).astype('uint8')
    
    return img