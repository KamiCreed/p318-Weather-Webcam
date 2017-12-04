import pandas as pd
import numpy as np
import sys
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import re
from skimage import io
import gc
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB

weather_classes = ["Cloudy", "Rain", "Clear", "Fog", "Drizzle", "Ice Pellets",
                   "Hail", "Thunderstorms", "Snow"]

# Clean the weather labels into several defined distinct labels.
# Also, allows for multiple labels per row.
def np_clean_labels(weather_labels):
    weather_reg = '|'.join(weather_classes)
    # https://stackoverflow.com/questions/42254384/pandas-extractall-is-not-extracting-all-cases-given-a-regex
    r = re.compile(r'\b(?:' + weather_reg + ')')

    length = len(weather_labels)
    all_labels = []
    for i in range(0,length):
        s = r.findall(weather_labels[i])
        num_matches = len(s)
        l = []
        for j in range(0,num_matches):
            l.append(s[j])
        all_labels.append(l)


    all_labels = pd.Series(all_labels)
    return all_labels

# Loads the images from an array of paths into a numpy array.
# Each image is ravelled into one long array.
def load_imgs(X_path):
    # Assuming they are all the same size
    shape = np.shape(io.imread(X_path[0]))
    # From https://stackoverflow.com/questions/27841554/array-of-images-in-python
    arr = np.array([np.array(io.imread(img).ravel()) for img in X_path])
    return arr, shape

# Return the hour from a time string as an int
def hourstring_to_int(hour_str):
    return int(hour_str[0:2])

# Interprets a string with the time (hour) as a time of day.
# Note Katkam images are only taken during daylight hours (usually 6-18 or 6AM to 6PM)
def hour_to_timeofday(hour_str):
    hour = hourstring_to_int(hour_str)
    if hour < 6:
        return 'Early morning'
    elif hour >= 6 and hour <= 12:
        return 'Morning'
    elif hour > 12 and hour < 18:
        return 'Afternoon'
    elif hour >= 18 and hour <= 24:
        return 'Evening'

def main():
    if len(sys.argv) < 5:
        print('Please input parameters in this order: CSV folder, Image folder, Target ID, Output folder')
        return

    csv_directory = sys.argv[1]
    img_directory = sys.argv[2]
    target_id = int(sys.argv[3])
    output_dir = sys.argv[4]
    try:
        os.makedirs(output_dir)
    except OSError as e:
        pass

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

    mlb = MultiLabelBinarizer()
    # Target == weather
    if target_id == 0:
        joined = joined.dropna(subset=['Weather'])
        y = np_clean_labels(joined['Weather']).values
        y = mlb.fit_transform(y)
    # Target == Time of day
    elif target_id == 1:
        joined['time_of_day'] = joined['Time'].apply(hour_to_timeofday)
        y = joined['time_of_day'].values

    # Target == Specific Hour
    elif target_id == 2:
        y = joined['Time'].apply(hourstring_to_int).values

    # Target == The next hour's weather
    elif target_id == 3:
        # Shift the weather column one day to get the next hour's weather
        joined['Next_Hour_Weather'] = joined['Weather'].shift(-1)
        joined = joined.dropna(subset=['Next_Hour_Weather'])
        y = np_clean_labels(joined['Next_Hour_Weather']).values
        y = mlb.fit_transform(y)
    else:
        print("Invalid column ID.")
        return

    X_train_paths, X_test_paths, y_train, y_test = train_test_split(joined['paths'].values, y)

    # Use MLP Classifier and PCA for Time of Day
    if target_id == 1 or target_id == 2:
        # Partial fitting allows for loading the images in two batches
        # ...instead of all at once which causes memory usage problems
        half = int(len(X_train_paths) / 2)
        X_path1 = X_train_paths[:half]
        X_path2 = X_train_paths[half:]
        y_train1 = y_train[:half]
        y_train2 = y_train[half:]
        classes = np.unique(y_train)

        print("Loading first half of images...")
        X_train1, _ = load_imgs(X_path1)

        if target_id == 1:
            print("Partial training first half (BernoulliNB)...")
            model = BernoulliNB(alpha=1.0,binarize=1.0)
        else: # target_id == 2
            print("Partial training first half (SGD)...")
            model = SGDClassifier()

        model.partial_fit(X_train1, y_train1, classes)
        del X_train1
        gc.collect()

        print("Loading second half of images...")
        X_train2, _ = load_imgs(X_path2)
        print("Partial training second half...")
        model.partial_fit(X_train2, y_train2)
        del X_train2
        gc.collect()

    # Use KNeighbors for weather predictions
    else:
        print("Loading images...")
        X_train, _ = load_imgs(X_train_paths)
        print("Start Training")
        model = make_pipeline(
                PCA(50),
                KNeighborsClassifier(n_neighbors=13)
               )
        model.fit(X_train, y_train)

    X_test, shape_test = load_imgs(X_test_paths)
    print(model.score(X_test, y_test))

    num_samples = 6
    if num_samples > X_test.shape[0]:
        num_samples = X_test.shape[0]

    sample = X_test[np.random.randint(0,X_test.shape[0],num_samples)]
    y_predict = model.predict(sample)
    if target_id == 0 or target_id == 3:
        y_predict = mlb.inverse_transform(y_predict)
        y_predict = [','.join(words) for words in y_predict]

    fig, axarr = plt.subplots(int(num_samples/3), 3)
    fig.suptitle('Predictions')
    for i in range(0,num_samples):
        ax = axarr[int(i/3),i%3]
        img = np.reshape(sample[i], shape_test) # Reshape ravelled back to RGB image
        ax.imshow(img)
        text = y_predict[i]
        if text == '':
            text = 'N/A'
        ax.set_title(text)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Predictions'))
    plt.close('all')

if __name__=='__main__':
    main()
