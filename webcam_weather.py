import pandas as pd
import numpy as np
import sys
from skimage.io import imread_collection
from skimage.io import ImageCollection
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

def main(csv_directory, img_directory):
    
    allFiles = glob.glob(csv_directory + "/*.csv")
    df = pd.concat((pd.read_csv(f, parse_dates=['Date/Time'], header=14) for f in allFiles), ignore_index=True)
    
    df = df.dropna(subset=['Weather'])
#    print(df)
    
#    imgs = imread_collection(os.path.join(img_directory,"*.jpg"))
    imgs = ImageCollection(os.path.join(img_directory,"*.jpg"))
    labels = pd.DataFrame([os.path.basename(f) for f in imgs.files], columns=['filename'])
    labels['string'] = labels.filename.str.extract('(\d+)', expand=True).astype(str)
    labels['Date/Time'] = pd.to_datetime(labels.string)
#    print(labels)
    
    joined = labels.set_index('Date/Time').join(df.set_index('Date/Time'))
#    print(joined)
    
    X_train, X_test, y_train, y_test = train_test_split(imgs, joined.Weather.values)
    model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(4, 3),
                      activation='logistic')
    model.fit(X_train, y_train)

if __name__=='__main__':
    csv_directory = sys.argv[1]
    img_directory = sys.argv[2]
    main(csv_directory, img_directory)