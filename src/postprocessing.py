import os
from glob import glob
import numpy as np
import cv2
from tifffile import imread, imwrite

def adaptative_hist_equalization(folder_path: str, input_filename: str):
    
    '''
    
    This function applies adaptative histogram equalization to the image in the folder_path.
    
    Parameters:
    - folder_path: str, path to the folder containing the image to be processed.
    - input_filename: str, base name of the image to be processed.
    
    Returns:
    - None
        
    '''
    
    image = imread(glob(os.path.join(folder_path, input_filename + "*.tiff"))[-1])
    
    lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = lab_img[:,:,0], lab_img[:,:,1], lab_img[:,:,2]
    clahe = cv2.createCLAHE(clipLimit = 5, tileGridSize = (2,2))
    l = clahe.apply(l)
    lab_img = np.dstack((l,a,b))
    rgb = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)        

    cv2.imwrite(os.path.join(folder_path, input_filename + '_postprocessed.tiff'), rgb)
         