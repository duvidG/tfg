#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:55:22 2022

@author: dgiron
"""


import pandas as pd

def mean_frec(data):
    """
    Creates a 2D image from the 3 dimensional images provided (in frecuency 
    dimension) by doing the mean of all of them.

    Parameters
    ----------
    data : list
        list with the data arrays.

    Returns
    -------
    reduced_data : pd.DataFrame
        mean of ll the images.

    """
    reduced_data = data[0]
    for i in range(len(data)-1):
        reduced_data = reduced_data.add(data[i+1], fill_value=0)
    
    reduced_data = reduced_data/len(data)
    return reduced_data