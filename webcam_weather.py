import pandas as pd
import numpy as np
import sys
from skimage.io import imread_collection
import glob
import os

def main(csv_directory, img_directory):
    
    allFiles = glob.glob(csv_directory + "/*.csv")
    df = pd.concat((pd.read_csv(f, parse_dates=['Date/Time'], header=14) for f in allFiles), ignore_index=True)
    # TODO: clean weather data
    
    #print(df)
    imgs = imread_collection(os.path.join(img_directory,"*.jpg"))
    print(imgs) #imgs[0] etc
    

if __name__=='__main__':
    csv_directory = sys.argv[1]
    img_directory = sys.argv[2]
    main(csv_directory, img_directory)