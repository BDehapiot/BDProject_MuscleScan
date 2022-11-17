#%% Imports

import cv2
import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
from joblib import Parallel, delayed

#%% Get raw name

stack_name = 'C1-2022.07.05_Luminy_22hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_24hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_1_1.tif'
# stack_name = 'C1-2022.07.05_Luminy_26hrsAPF_Phallo568_aAct488_405nano2_100Xz2.5_AS_488LP4_2_2.tif'

#%% Get path and open data

data_path = Path(Path.cwd(), 'data')
stack_path = Path(data_path, stack_name)
stack = io.imread(stack_path)

stack = stack[9,...]

#%% Parameters

# Gabor kernels
n_kernels = 16
kernel_size = 50
theta = 0
sigma = 3
lmbda = 20 # 
gamma = 0.25 # 0 to 1
psi = 0 # 0 to 6          

#%%

# sigma test for gabor kernels
sigma_kernels = [] 
for sigma in np.arange(20):
    kernel = cv2.getGaborKernel(
        (kernel_size, kernel_size), 
        sigma, theta, lmbda, gamma, psi, 
        ktype=cv2.CV_64F
        )
    # kernel /= 1.0 * kernel.sum() # Brightness normalization
    sigma_kernels.append(kernel)
               
viewer = napari.Viewer()
viewer.add_image(np.array(sigma_kernels))            
    
#%%

# # lmbda test for gabor kernels
# lmbda_kernels = [] 
# for lmbda in np.arange(20):
#     kernel = cv2.getGaborKernel(
#         (kernel_size, kernel_size), 
#         sigma, theta, lmbda, gamma, psi, 
#         ktype=cv2.CV_64F
#         )
#     # kernel /= 1.0 * kernel.sum() # Brightness normalization
#     lmbda_kernels.append(kernel)

# viewer = napari.Viewer()
# viewer.add_image(np.array(lmbda_kernels))     

    
#%%

# # gamma test for gabor kernels
# gamma_kernels = [] 
# for gamma in np.arange(0,1,0.1):
#     kernel = cv2.getGaborKernel(
#         (kernel_size, kernel_size), 
#         sigma, theta, lmbda, gamma, psi, 
#         ktype=cv2.CV_64F
#         )
#     # kernel /= 1.0 * kernel.sum() # Brightness normalization
#     gamma_kernels.append(kernel)

# viewer = napari.Viewer()
# viewer.add_image(np.array(gamma_kernels))       

#%%

# # psi test for gabor kernels
# psi_kernels = [] 
# for psi in np.arange(0,7,1):
#     kernel = cv2.getGaborKernel(
#         (kernel_size, kernel_size), 
#         sigma, theta, lmbda, gamma, psi, 
#         ktype=cv2.CV_64F
#         )
#     # kernel /= 1.0 * kernel.sum() # Brightness normalization
#     psi_kernels.append(kernel)

# viewer = napari.Viewer()
# viewer.add_image(np.array(psi_kernels))  

