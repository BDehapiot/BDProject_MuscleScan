#%% Imports

import cv2
import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
from joblib import Parallel, delayed

#%% Get raw name

# stack_name = 'C1-2022.07.05_Luminy_22hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
stack_name = 'C1-2022.07.05_Luminy_24hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_26hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_2_2.tif'

#%% Get path and open data

data_path = Path(Path.cwd(), 'data')
stack_path = Path(data_path, stack_name)
stack = io.imread(stack_path)

img = stack[9,...]

#%% Parameters

# Gabor kernels
n_kernels = 16
kernel_size = 25
sigma = 6
lmbda = 10
gamma = 0.75
psi = 0 # 0 to 6 

#%%

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
    # kernel /= 1.0 * kernel.sum() # Brightness normalization
    kernels.append(kernel)
    
# Apply filters
img_filt = np.zeros((n_kernels, img.shape[0], img.shape[1]))
for k, kernel in enumerate(kernels):
    img_filt[k,...] = cv2.filter2D(img, -1, kernel) 
    
end = time.time()
print(f'  {(end-start):5.3f} s')  

# -----------------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(np.array(kernels)) 
# viewer.add_image(img) 
# viewer.add_image(img_filt)   
    
    