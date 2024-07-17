import os
from glob import glob
import argparse
from src.preprocessing import *
from src.blending import *

def main(case, preprocessing_flag, blending_type):

    folder_path_raw = os.path.join("data", "case0" + str(case), "raw")

    #Define preprocessing steps
    if preprocessing_flag:
        
        folder_path_preprocessed = os.path.join("data", "case0" + str(case), "preprocessed")
        
        bias_field_correction(folder_path_raw)
        match_histograms(folder_path_preprocessed)

        #Register images 
        warpedImgs = registration(folder_path_preprocessed)
        
    else:
        warpedImgs = registration(folder_path_raw)
    
    #Define blending
    if blending_type == "raw":
        mosaic = raw_blending(warpedImgs)
    elif blending_type == "weighted":
        mosaic = weighted_blending(warpedImgs)
    elif blending_type == "nonuniform":
        mosaic = nonuniform_blending(warpedImgs)
    else:
        print("Error: blending type must be 'raw', 'weighted' or 'nonuniform'")
        sys.exit()
    
    #Save mosaic
    imwrite(os.path.join("data", "case0" + str(case), "mosaic", "mosaic.tiff"), mosaic)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run preprocessing and blending steps")
    parser.add_argument("--case", type=int, choices=[1, 2], help="Case number (1 or 2)")
    parser.add_argument("--preprocessing", action="store_true", help="Whether to perform preprocessing")
    parser.add_argument("--blending_type", choices=["raw", "weighted", "nonuniform"], help="Type of blending")

    args = parser.parse_args()

    main(args.case, args.preprocessing, args.blending_type)
