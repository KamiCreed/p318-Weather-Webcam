#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:22:43 2017

Helps figure out unique categories for training

@author: masamim
"""

import pandas as pd
import glob
import sys

def main(csv_directory, output):
    allFiles = glob.glob(csv_directory + "/*.csv")
    df = pd.concat((pd.read_csv(f, parse_dates=['Date/Time'], header=14) for f in allFiles), ignore_index=True)
    
    df.drop_duplicates('Weather', inplace=True)
    df['Weather'].to_csv(output)

if __name__=='__main__':
    csv_directory = sys.argv[1]
    output = sys.argv[2]
    main(csv_directory, output)