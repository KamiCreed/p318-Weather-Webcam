import pandas as pd
import numpy as np
import sys
from skimage.io import imread_collection
from skimage.io import ImageCollection
import glob
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from PIL import Image
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import image
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from skimage import exposure
from skimage import filters
import re
from skimage import io


# Clean the weather labels into several defined distinct labels.
# Also, allows for multiple labels per row.
def np_clean_labels(weather_labels):
    # https://stackoverflow.com/questions/42254384/pandas-extractall-is-not-extracting-all-cases-given-a-regex
    r = re.compile(r'\b(?:Cloudy|Rain|Clear|Fog|Drizzle|Ice Pellets|Hail|Thunderstorms|Snow)')
    
    length = len(weather_labels)
    all_labels = []
    for i in range(0,length):
        sub_len = len(weather_labels[i])
        s = r.findall(weather_labels[i])
        num_matches = len(s)
        string_out = ''
        for j in range(0,num_matches):
            string_out += s[j]
            if j < num_matches-1:
                string_out += ','
        all_labels.append(string_out)
        
    
    all_labels = pd.Series(all_labels)
    return all_labels
    
# Loads the images from an array of paths into a numpy array.
# Each image is ravelled into one long array.
def load_imgs(X_path):
    # From https://stackoverflow.com/questions/27841554/array-of-images-in-python
    arr = np.array([np.array(io.imread(img).ravel()) for img in X_path])
    return arr

# Return the hour from a time string as an int
def hourstring_to_int(hour_str):
    return int(hour_str[0:2])

# Interprets a string with the time (hour) as a time of day.
# Note Katkam images are only taken during daylight hours (usually 6-18 or 6AM to 6PM)
def hour_to_timeofday(hour_str):
    hour = hourstring_to_int(hour_str)
    if hour > 6:
        return 'Early morning'
    elif hour >= 6 and hour <= 12:
        return 'Morning'
    elif hour > 12 and hour < 18:
        return 'Afternoon'
    elif hour >= 19 and hour <= 24:
        return 'Evening'

def main():
    if len(sys.argv) < 4:
        print('Please input parameters in this order: CSV folder, Image folder, Column ID')
        return
    
    csv_directory = sys.argv[1]
    img_directory = sys.argv[2]
    column_id = int(sys.argv[3])
    
    # Load in CSV files from a folder
    # Where the data actually starts is hard-coded in `header=14`.
    allFiles = glob.glob(csv_directory + "/*.csv")
    df = pd.concat((pd.read_csv(f, parse_dates=['Date/Time'], header=14) for f in allFiles), ignore_index=True)
    
    paths = []
    names = []

    # Get the image paths in the folder    
    # Adapted from https://stackoverflow.com/questions/34976595/using-train-test-split-with-images-from-my-local-directory
    for path, subdirs, files in os.walk(img_directory):
        for name in files:
            img_path = os.path.join(path,name)
            paths.append(img_path)
            names.append(name)
    
    # Join the image paths and weather data in terms of date.
    # This makes sure the X and the y arrays are the same size.
    labels = pd.DataFrame(names, columns=['filename'])
    labels['paths'] = pd.Series(paths)
    labels['string'] = labels.filename.str.extract('(\d+)', expand=True).astype(str)
    labels['DateTime'] = pd.to_datetime(labels.string)
    joined = labels.set_index('DateTime').join(df.set_index('Date/Time'))
    
    # Target == weather
    if column_id == 0:
        joined = joined.dropna(subset=['Weather'])
        y = np_clean_labels(joined['Weather']).values
        y = MultiLabelBinarizer().fit_transform(y)
    # Target == Time of day
    elif column_id == 1:
        joined['time_of_day'] = joined['Time'].apply(hour_to_timeofday)
        y = joined['time_of_day'].values
    # Target == Specific Hour
    elif column_id == 2:
        y = joined['Time'].apply(hourstring_to_int).values
    # Target == The next hour's weather
    elif column_id == 3:
        # Shift the weather column one day to get the next hour's weather
        joined['Next_Hour_Weather'] = joined['Weather'].shift(-1)
        joined = joined.dropna(subset=['Next_Hour_Weather'])
        y = np_clean_labels(joined['Next_Hour_Weather']).values
        y = MultiLabelBinarizer().fit_transform(y)
    else:
        print("Invalid column ID.")
        return
    
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(joined['paths'].values, y)
    
    print("Loading images...")
    X_train = load_imgs(X_train_paths)
    
    print("Start Training")
    model = make_pipeline(
                PCA(50),
                KNeighborsClassifier(n_neighbors=13)
                )
    model.fit(X_train, y_train)
    
    X_test = load_imgs(X_test_paths)
    
    print(model.score(X_test, y_test))

if __name__=='__main__':
    main()