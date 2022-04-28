#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:55:22 2022

@author: dgiron
"""


import pandas as pd

def mean_frec(data):
    reduced_data = data[0]
    for i in range(len(data)-1):
        reduced_data = reduced_data.add(data[i+1], fill_value=0)
    
    reduced_data = reduced_data/len(data)
    return reduced_data