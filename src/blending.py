import os
import sys
from glob import glob
import warnings
import numpy as np
from edt import edt
import cv2
from tifffile import imread, imwrite

def registration(folder_path: str):
    
    '''
    
    This function reads the images from the input_folder and applies a known affine transformations to register them.
    
    Parameters:
    - folder_path: str, path to the folder containing the images to be registered.

    Returns:
    - warpedImgs: list, list of the registered images.
    
    '''
    
    imgs = []
    affineTransforms = []
    
    if folder_path.find("case01"):
        case = 1
    elif folder_path.find("case02"):
        case = 2
    else:
        print("Error: folder path must contain 'case01' or 'case02'")
        sys.exit()

    if case == 1:
        imgs.append(imread(glob(os.path.join(folder_path, "*0.tiff"))[-1]))
        imgs.append(imread(glob(os.path.join(folder_path, "*1.tiff"))[-1]))
        affineTransforms.append(np.array([[1, 0, 614],
                                        [0, 1, 765]], np.float32))
        affineTransforms.append(np.array([[0.958, 0.297, 2.43e-05],
                                        [-0.297, 0.958, 822]], np.float32))
        mosaicWidth = 3383
        mosaicHeight = 2606
    elif case == 2:
        imgs.append(imread(glob(os.path.join(folder_path, "*0.tiff"))[-1]))
        imgs.append(imread(glob(os.path.join(folder_path, "*1.tiff"))[-1]))
        imgs.append(imread(glob(os.path.join(folder_path, "*2.tiff"))[-1]))
        imgs.append(imread(glob(os.path.join(folder_path, "*3.tiff"))[-1]))
        imgs.append(imread(glob(os.path.join(folder_path, "*4.tiff"))[-1]))
        affineTransforms.append(np.array([[1, 0, 1823],
                                        [0, 1, 0]], np.float32))
        affineTransforms.append(np.array([[1, -0.052, 1673],
                                        [0.052, 1, 454]], np.float32))
        affineTransforms.append(np.array([[0.994, -0.052, 1402],
                                        [0.052, 0.994, 1157]], np.float32))
        affineTransforms.append(np.array([[0.985, 0.039, 1030],
                                        [-0.039, 0.985, 2108.]], np.float32))
        affineTransforms.append(np.array([[0.994, 0.093, -2.71e-05],
                                        [-0.093, 0.994, 2957]], np.float32))
        mosaicWidth = 4598
        mosaicHeight = 4786
    else:
        print("Error: case number must be 1 or 2")
        sys.exit()

    warpedImgs = []
    for i in range(len(imgs)):
        warpedImgs.append(cv2.warpAffine(imgs[i], affineTransforms[i],
                                         (mosaicWidth, mosaicHeight), flags=cv2.INTER_LINEAR))
        
    return warpedImgs

def raw_blending(warpedImgs: list):
    
    '''
    
    This function creates a mosaic by overlapping registered images between each other.
    
    Parameters:
    - warpedImgs: list, list of the registered images.
    
    Returns:
    - mosaic: np.array, the mosaic image.
    
    '''
        
    mosaic = warpedImgs[-1]
    for i in reversed(range(len(warpedImgs)-1)):
        mosaic = np.where(mosaic != 0, mosaic, warpedImgs[i])
        
    return mosaic

def weighted_blending(warpedImgs: list):
    
    '''
    
    This function creates a mosaic by overlapping registered images between each other.
    Averaged contributions are defined in overlapping regions.
        
    Parameters:
    - warpedImgs: list, list of the registered images.
    
    Returns:
    - mosaic: np.array, the mosaic image.
    
    '''
    
    # Define alpha map
    warnings.filterwarnings("ignore")
    overlapping_count_mask = np.zeros(warpedImgs[0].shape[:2])
    mosaic = warpedImgs[-1]
    for i in range(len(warpedImgs)):
        overlapping_count_mask[np.sum(warpedImgs[i], axis=-1) > 0] += 1
    alpha_maps = np.divide(1,overlapping_count_mask)
    alpha_maps[alpha_maps > 1] = 0
    alpha_maps = np.repeat(np.expand_dims(alpha_maps, axis=-1), 3, axis=-1)

    # Image blending to create the mosaic
    for channel in range(3):
        stacked_images = np.array([warpedImgs[i]*alpha_maps for i in range(len(warpedImgs))])
    mosaic = np.sum(stacked_images, axis=0).astype(np.uint8)

    return mosaic

def nonuniform_blending(warpedImgs: list):
    
    '''
    
    This function creates a mosaic by overlapping registered images between each other.
    Nonuniform weighted contributions are defined in overlapping regions.
    Weights are estimated based on the distance from image centers: the further the image, the lower the weight.
    
    Parameters:
    - warpedImgs: list, list of the registered images.
    
    Returns:
    - mosaic: np.array, the mosaic image.
    
    '''
    
    # Estimate distance from centered images for nonuniform weighting in overlapping regions
    warnings.filterwarnings("ignore")
    stacked_edt = []
    for i in range(len(warpedImgs)):
        stacked_edt.append(edt(np.sum(warpedImgs[i], axis=-1) > 0))
    normalized_edt = np.nan_to_num(stacked_edt/np.sum(np.array(stacked_edt), axis=0))
    normalized_edt = np.repeat(np.expand_dims(normalized_edt, axis=-1), 3, axis=-1)

    # Image blending to create the mosaic
    for channel in range(3):
        stacked_images = np.array([warpedImgs[i]*normalized_edt[i] for i in range(len(warpedImgs))])
    mosaic = np.sum(stacked_images, axis=0).astype(np.uint8)

    return mosaic