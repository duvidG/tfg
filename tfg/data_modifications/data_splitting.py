#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:00:22 2022

@author: dgiron
"""

import pandas as pd
import math
from errors import DimensionError


def split_into_small_boxes(data, number_of_boxes, catalogue, overlapping=40):
    """
    Splits a 2D datacube into the (number of boxes * number of boxes) given as parameter. It also considers
    overlapping to include galaxies just in the boundary of the small datacubes. The catalogue for the sources is
    also splitted.


    Parameters
    ----------
    data : pd.DataFrame
        2D dataframe with data.
    number_of_boxes : int
        number of boxes to divide the dataset in each direction.
    catalogue: pd.DataFrame
        catalogue of the sources in the full datacube. 'ra' and 'dec' column must be given in pixels
    overlapping : int, optional
        number of extra entries to consider in each minicube, given in pixels. The default is 40.

    Raises
    ------
    DimensionError
        if number of boxes is bigger than the number of entries in any direction.

    Returns
    -------
    data_splitted : list
        list with the small dataframes.
    thruth_cat_splitted : list
        list with catalogues for each of the small datacubes

    """
    
    data_splitted = []
    thruth_cat_splitted = []
    len_height = len(data.iloc[:, 0])
    len_width = len(data.iloc[0, :])

    if number_of_boxes > len_height or number_of_boxes > len_width or overlapping > len_width or overlapping > len_height:
        raise DimensionError('The number of boxes selected is bigger than the height or width of the data table')
    # if len_height != len_width or len_height % 2 != 0:
    #     raise DimensionError('The dataset can not be divided into equal small datasets. Try removing or adding rows or changing the'+
    #                          'number of datacubes in each row')
        
        
    # Prevents round(xxxx.5) values, which causes the last column not to be considered sometimes
    if math.modf(len_height/number_of_boxes)[0] < 0.500001:
        height_small_box = math.floor(len_height/number_of_boxes)
        
    else:
        height_small_box = round(len_height/number_of_boxes)
        
        
    if math.modf(len_width/number_of_boxes)[0] < 0.500001:
        width_small_box = math.floor(len_width/number_of_boxes)
                
    else:
        width_small_box = round(len_width/number_of_boxes)
            
    # Loop over each row
    for i in range(number_of_boxes):
        # Prevents negative pixels
        if i == number_of_boxes-1: # Last minicube without overlapping in the border
            data_mod = data.iloc[max((i*height_small_box)-overlapping, 0):, :]
        # if i == 0:
        #     data_mod = data.iloc[max((i*height_small_box)-overlapping, 0):, :]    
        else:
            data_mod = data.iloc[max((i*height_small_box)-overlapping, 0):((i+1)*height_small_box)+overlapping, :]
        # Loop over each column in a row
        for var in range(number_of_boxes):
            # Prevents negative pixels in the table
            if var == number_of_boxes-1:
                mini_data = data_mod.iloc[:, max((var*width_small_box)-overlapping, 0):]
            else:
                mini_data = data_mod.iloc[:, max((var*width_small_box)-overlapping, 0):((var+1)*width_small_box)+overlapping]
            data_splitted.append(mini_data)
            
            # Catalogue part
            limit_pixels_vert = (mini_data.index.values[0], mini_data.index.values[-1])
            limit_pixels_hor = (mini_data.columns.values[0], mini_data.columns.values[-1])
            # Loop over every source in the catalogue. If it's inside the small cubes it saves the source in a different dataframe
            thruth_cat_small = catalogue.loc[(catalogue['ra']>=limit_pixels_hor[0]) & (catalogue['ra']<= limit_pixels_hor[1]) & 
                                              (catalogue['dec']>=limit_pixels_vert[0]) & (catalogue['dec']<= limit_pixels_vert[1]), :]
            # When the loop is completed, the table is stored                    
            thruth_cat_splitted.append(thruth_cat_small)
    
    return data_splitted, thruth_cat_splitted


# def split_catalogue(data, catalogue):
#     """
#     Divide the thruth catalogue dataset in an array of dataframes, one for each small 
#     datacube

#     Parameters
#     ----------
#     data : pd.DataFrame
#         table with values for each pixel.
#     catalogue : pd.DataFrame
#         table with parameters of the sources.

#     Returns
#     -------
#     thruth_cat_splitted : list of pd.DataFrame
#         list of tables with the sources for each datacube (same order as small_datacubes attribute).

#     """
#     # List to save the dataframes
#     thruth_cat_splitted = []
    
#     # Loop over every small datacube
#     for small_cube in data:
#         thruth_cat_small = pd.DataFrame([], columns=catalogue.columns.values)
#         # Limits of the small cubes (in pixels)
#         limit_pixels_vert = (small_cube.index.values[0], small_cube.index.values[-1])
#         limit_pixels_hor = (small_cube.columns.values[0], small_cube.columns.values[-1])
#         # Loop over every source in the catalogue. If it's inside the small cubes it saves the source in a different dataframe
#         thruth_cat_small = catalogue.loc[(catalogue['ra']>=limit_pixels_hor[0]) & (catalogue['ra']<= limit_pixels_hor[1]) & 
#                                           (catalogue['dec']>=limit_pixels_vert[0]) & (catalogue['dec']<= limit_pixels_vert[1]), :]
#         # When the loop is completed, the table is stored                    
#         thruth_cat_splitted.append(thruth_cat_small)
#     return thruth_cat_splitted             


def test():
    tab = pd.DataFrame([[1, 2, 3, 5, 6, 8, 9, 0],
                        [1, 2, 3, 8, 6, 8, 9, 0],
                        [3, 4, 5, 9, 7, 8, 9, 0],
                        [5, 2, 3, 5, 7, 8, 9, 0],
                        [4, 2, 3, 8, 8, 8, 9, 0],
                        [3, 4, 5, 9, 9, 8, 9, 0]])

    a = 12
    # Original table
    print(tab)
    
    # No overlap
    print(split_into_small_boxes(tab, 4, 0, 0)[0][12])
    # With overlap
    print(split_into_small_boxes(tab, 4, 2, 1)[0][a])
    
#test()