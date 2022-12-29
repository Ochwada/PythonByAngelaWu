#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 10:07:29 2022

Data science training

@author: ochwada
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor



# Downloading and preparing data

url = 'https://github.com/Ochwada/Datasets/raw/main/'

lifesat_df = pd.read_csv(url + "lifesat/lifesat.csv")

# Get the values to plot

X = lifesat_df[["GDP per capita (USD)"]].values

y = lifesat_df[["Life satisfaction"]].values



# visualizing the data

lifesat_df.plot(
    kind = "scatter", grid=True,
    x = "GDP per capita (USD)", y = "Life satisfaction"
    )

plt.axis([23_500, 62_500, 4, 9])

plt.show()


#Select a linear model
# model= LinearRegression()
model= KNeighborsRegressor(n_neighbors=3)

#Train the model

model.fit(X, y)

#Make a prediction for Cyprus

X_new = [[37_655.2]] # Cyprus' GDP per Capita in 2020

print(model.predict(X_new))

