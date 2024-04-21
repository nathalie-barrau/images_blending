import os
from glob import glob
import numpy as np
import cv2
import SimpleITK as sitk

def n4_bias_estimation(image: np.ndarray):
    
    '''
    
    This function estimates the bias field of an image using the N4 bias field correction algorithm.
    
    Parameters:
    - image: np.array, image to be processed.
    
    Returns:
    - global_bias_field: np.array, estimated global bias field on the overall image.
    - borders_bias_field: np.array, finetuned bias estimation on the borders of the image.
    
    '''
    
    #Global N4 bias estimation of illumination
    inputImage = sitk.Cast(sitk.GetImageFromArray(image), sitk.sitkFloat32)
    inputImageLowRes = sitk.Shrink(inputImage, [4] * inputImage.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([int(32)])
    _ = corrector.Execute(inputImageLowRes)
    log_bf = corrector.GetLogBiasFieldAsImage(inputImage)
    global_bias_field = sitk.GetArrayFromImage(sitk.Exp(log_bf))
    
    #Fine tuning on borders bias estimation
    mask = np.ones(image.shape)
    bounds = (np.array(mask.shape)/6).astype(int)
    mask[bounds[0]:-bounds[0], bounds[1]:-bounds[1]] = 0
    inputImage = sitk.Cast(sitk.GetImageFromArray(np.divide(image, global_bias_field)), sitk.sitkFloat32)
    inputImageLowRes = sitk.Shrink(inputImage, [4] * inputImage.GetDimension())
    inputMask = sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt8)
    maskLowRes = sitk.Shrink(inputMask, [4] * inputMask.GetDimension())
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([int(128)])
    _ = corrector.Execute(inputImageLowRes, maskLowRes)
    log_bf = corrector.GetLogBiasFieldAsImage(inputImage)
    borders_bias_field = sitk.GetArrayFromImage(sitk.Exp(log_bf))
    
    return global_bias_field, borders_bias_field

def bias_field_correction(folder_path: str):
    
    '''
    
    This function applies bias field correction to all images in the folder_path.
    
    Parameters:
    - folder_path: str, path to the folder containing the images to be processed.
    
    Returns:
    - None
   
    '''
    
    # Apply on all images of the data
    images = [cv2.imread(img_path) for img_path in glob(os.path.join(folder_path, '*.tiff'))]
    
    # Apply bias corrections on all images
    for idx, image in enumerate(images):
        
        input = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #cv2.COLOR_BGR2HSV
        output = np.copy(input)
 
        for channel in [1,2]: # Only on S and V component
            global_bias_field, borders_bias_field = n4_bias_estimation(input[:,:,channel])
            unbiased_channel = np.divide(np.divide(input[:,:,channel], global_bias_field), borders_bias_field)
            unbiased_channel = np.round((unbiased_channel-unbiased_channel.min() + input[:,:,channel].min())/\
                unbiased_channel.max()*input[:,:,channel].max())  
            output[:,:,channel] = unbiased_channel.astype(np.uint8)
            
        rgb_output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
        
        cv2.imwrite(os.path.join(folder_path, '..', 'preprocessed', '{:.0f}.tiff'.format(idx)), rgb_output)

def match_histograms(folder_path: str):
    
    '''
    
    This function applies histogram matching to all images in the folder_path.
    
    Parameters:
    - folder_path: str, path to the folder containing the images to be processed.
    
    Returns:
    - None
        
    '''
    
    # Load images
    images = [cv2.imread(img_path) for img_path in glob(os.path.join(folder_path, '*.tiff'))]

    # Compute reference histograms for each channel
    reference_hists = [np.zeros((256, 1), dtype=np.float32) for _ in range(3)]
    for image in images:
        for i in range(3):
            channel_hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            reference_hists[i] += channel_hist
    for i in range(3):
        reference_hists[i] /= len(images)

    # Match histograms of all images to the reference histograms
    for idx, image in enumerate(images):
        matched_channels = []
        
        # Match each channel to the reference histogram
        for i in range(3):
            channel = image[:, :, i]
            channel_cdf = np.cumsum(cv2.calcHist([channel], [0], None, [256], [0, 256]))
            reference_cdf = np.cumsum(reference_hists[i])
            mapping = np.interp(channel_cdf, reference_cdf, range(256))
            matched_channels.append(cv2.LUT(channel, mapping.astype('uint8'))) # Apply mapping to channel
        matched_image = cv2.merge(matched_channels)

        # Save image
        cv2.imwrite(os.path.join(folder_path, '..', 'preprocessed', '{:.0f}.tiff'.format(idx)), matched_image) 
        

def adaptative_hist_equalization(folder_path: str):
    
    '''
    
    This function applies adaptative histogram equalization to all images in the folder_path.
    
    Parameters:
    - folder_path: str, path to the folder containing the images to be processed.
    
    Returns:
    - None
        
    '''
    
    # Load images
    images = [cv2.imread(img_path) for img_path in glob(os.path.join(folder_path, '*.tiff'))]
    
    # Apply CLAHE on all images
    for idx, image in enumerate(images):
        
        # Convert to LAB color space
        lab_img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = lab_img[:,:,0], lab_img[:,:,1], lab_img[:,:,2]
        
        #Apply CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit = 5, tileGridSize = (2,2))
        l = clahe.apply(l)
        lab_img = np.dstack((l,a,b))
        
        # Convert back to RGB
        rgb = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)        

        cv2.imwrite(os.path.join(folder_path, '..', 'preprocessed', '{:.0f}.tiff'.format(idx)), rgb)