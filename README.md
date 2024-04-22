# Image mosaics

This project present a simple proof of concept to optimized skin images blending.

## Project structure

    ```                      
    ├── data/              
    │   ├── case01/                  
    │   │   ├── raw/
    │   │   │   ├── `0.tiff`     
    │   │   │   └── `1.tiff`  
    │   │   ├── preprocessed/       
    │   │   └── mosaic/               
    │   └── case02/                   
    │       ├── raw/
    │       │   ├── `0.tiff`
    │       │   ├── `1.tiff`
    │       │   ├── `2.tiff`
    │       │   ├── `3.tiff`     
    │       │   └── `4.tiff`                   
    │       ├── preprocessed/          
    │       └── mosaic/
    ├── src/              
    │   ├── `preprocessing.py`: Contains functions for different image preprocessing steps                   
    │   ├── `blending.py`: Contains functions for registration and images blending    
    │   └── `postprocessing.py`: Contains a function for mosaic postprocessing
    ├── `main.py`: Entry point script to run preprocessing and blending steps producing mosaics
    ├── `requirements.txt`: Lists project dependencies
    └── `README.md`: Describe the project and provides information to use the code
    ```

## Installation

Install Python 3.8 (more recent Python versions might work but have not been tested)

Clone the repository

(Optional) Create a virtual environment with Python 3.8
    
    ```
    python3.8 -m venv env
    source env/bin/activate
    ```

Install dependencies:

    ```
    python -m pip install -r requirements.txt
    ```

## Usage

Run the main script `main.py` with appropriate arguments:

    `--case`: Specify the case number (1 or 2).
    `--preprocessing`: Flag to perform preprocessing steps.
    `--blending_type`: Specify the type of blending (raw, weighted, nonuniform).

The following command run preprocessing and perform nonuniform blending for case 1:

    ```
    python main.py --case 1 --preprocessing --blending_type nonuniform
    ```

The following command perform raw blending for case 2 without preprocessing:

    ```
    python main.py --case 2 --blending_type raw
    ```

## Results

Resulting mosaics are saved whithin "mosaic" folders for each case.