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

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


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
#housing.hist(bins = 50, figsize= (12,8))
# plt.show()


# Splitting the data and creating a Test Set (but not perfet) - with several saves, the model 
#will view the data
# =============================================================================
# 
# def shuffle_create_test_set(data, test_ratio):
#     #shuffled_indices = np.random.seed(46)
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     
#     return data.iloc[train_indices], data.iloc[test_indices]
#     
# train_set, test_set =  shuffle_create_test_set(housing, 0.2)
# 
# print(len(train_set))
# print(len(test_set))
# =============================================================================

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


housing["income_cat"] = pd.cut(housing["median_income"],
                               bins = [0., 1.5, 3., 4.5, 6., np.inf],
                               labels = [1, 2,3,4,5]
                               )
# =============================================================================
# 
# housing["income_cat"].value_counts().sort_index().plot.bar(rot = 0, grid=True)
# 
# plt.xlabel("income Category")
# plt.ylabel("Number of districts")
# 
# plt.show()
# 
# =============================================================================


splitter  = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)


strat_splits = []

for train_index, test_index in splitter.split(housing, housing["income_cat"]):
    strat_train_set_n = housing.iloc[train_index]
    strat_test_set_n = housing.iloc[test_index]
    
    strat_splits.append([strat_train_set_n, strat_test_set_n])
    
#use the 1st split

#strat_train_set_n, strat_test_set_n = strat_splits[0]

strat_train_set, strat_test_set = train_test_split(
    housing,
    test_size = 0.2,
    stratify= housing["income_cat"], random_state=42
    )
    
#print(strat_test_set["income_cat"].value_counts()/ len(strat_test_set))
    
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
    
#Copy of data to be retrived later -->

# housing = strat_test_set.copy()

housing.plot(kind = "scatter", x = "longitude", y = "latitude", grid=True, 
             
             s =housing["population"]/100, label = "population",
             c = "median_house_value", cmap ="jet", colorbar = True,
             legend = True, sharex = False, figsize =(10,7)            
             
             )

plt.show()




corr_matrix = housing.corr()

print(corr_matrix["median_house_value"].sort_values(ascending=False))




# Correltaion between attributes (use scatter_matrix)

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]

scatter_matrix(housing[attributes], figsize=(12,8))


plt.show()













    
    
    
    
    
    
    
    
    
    
    
    
    