import pandas as pd
import numpy as np

def main(csv_directory, img_directory):
    weather = pd.read_csv(csv_directory, parse_dates=['Date/Time'], header=14)
    

if __name__=='__main__':
    csv_directory = sys.argv[1]
    img_directory = sys.argv[2]
    main(csv_directory, img_directory)