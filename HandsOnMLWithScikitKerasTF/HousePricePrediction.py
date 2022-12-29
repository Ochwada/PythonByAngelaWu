#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:07:29 2022

Data science training

@author: ochwada
"""
# ----- Libraries ---------
from pathlib import Path
import tarfile
import urllib.request


import pandas as pd
import matplotlib.pyplot as plt


# --------------------------

# Function to fetch and load data --->
def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/Ochwada/Datasets/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path = "datasets")
            
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()

#print(housing[:10])
#print(housing.head())

ocean_proximity_counts = housing["ocean_proximity"].value_counts()

description_housing  = housing.describe()

# Ploting for better visualization ------>
housing.hist(bins = 50, figsize= (12,8))
# plt.show()


# Splitting the data and creating a Test Set