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
from PIL import Image
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
    
def load_imgs(X_path):
    l = []
    for img_path in X_path:
        img2d = io.imread(img_path)
        l.append(img2d.ravel())
    return np.asarray(l)

def load_imgs_processing(X_path):
    l = []
    for img_path in X_path:
        img2d = io.imread(img_path)
#        misc.imshow(img2d)
#        input("Press enter for processing...")
#        img2d = exposure.equalize_hist(img2d)
#        misc.imshow(img2d)
#        input("Press enter to continue...")
        img2d = filters.gaussian(img2d, sigma=8, multichannel=True)
        l.append(img2d.ravel())
    return np.asarray(l)

def main():
    if len(sys.argv) < 4:
        print('Please input parameters in this order: CSV folder, Image folder, Column ID')
        return
    
    csv_directory = sys.argv[1]
    img_directory = sys.argv[2]
    column_id = int(sys.argv[3])
    
    allFiles = glob.glob(csv_directory + "/*.csv")
    df = pd.concat((pd.read_csv(f, parse_dates=['Date/Time'], header=14) for f in allFiles), ignore_index=True)
#    print(df)
    
#    imgs = imread_collection(os.path.join(img_directory,"*.jpg"))
#    imgs = ImageCollection(os.path.join(img_directory,"*.jpg"))
#    print(imgs.files)
#    dataset_size = len(imgs)
#    imgs = imgs.reshape(dataset_size,-1)
    
    paths = []
    names = []
    
    # From https://stackoverflow.com/questions/34976595/using-train-test-split-with-images-from-my-local-directory
    for path, subdirs, files in os.walk(img_directory):
        for name in files:
            img_path = os.path.join(path,name)
            paths.append(img_path)
            names.append(name)
    
    labels = pd.DataFrame(names, columns=['filename'])
    labels['paths'] = pd.Series(paths)
    labels['string'] = labels.filename.str.extract('(\d+)', expand=True).astype(str)
    labels['Date/Time'] = pd.to_datetime(labels.string)
    
    joined = labels.set_index('Date/Time').join(df.set_index('Date/Time'))
    
    if column_id == 0:
        joined = joined.dropna(subset=['Weather'])
        y = np_clean_labels(joined['Weather']).values
        y = MultiLabelBinarizer().fit_transform(y)
    elif column_id == 1:
        y = joined.index.hour
    else:
        print("No training of column.")
        return
    
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(joined['paths'].values, y)
    
    print("Loading images...")
    X_train = load_imgs_processing(X_train_paths)
    
    print("Start Training")
    #X_train, X_test, y_train, y_test = train_test_split(X, joined.Weather.values)
    #X_train, X_test, y_train, y_test = train_test_split(X, MultiLabelBinarizer().fit_transform(joined['Weather']))
    # got 0.00542 with MLP, no OneVsRest, MultiLabelBinarizer
    # got 0.00542 with MLP, OneVsRest, MultiLabelBinarizer
    #X_train, X_test, y_train, y_test = train_test_split(X, joined['clean'].values) #doesn't work because multilabel

    model = make_pipeline(
                PCA(40000),
#                MLPClassifier(solver='lbfgs', hidden_layer_sizes=(100),
#                      activation='logistic')
                KNeighborsClassifier(n_neighbors=20)
                )
    
    model.fit(X_train, y_train)
    
    X_test = load_imgs(X_test_paths)
    
    print(model.score(X_test, y_test))

if __name__=='__main__':
    main()