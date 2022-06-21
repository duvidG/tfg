#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:00:22 2022

@author: dgiron
"""

import pandas as pd
import numpy as np
from src.errors import DimensionError
import warnings
import time

def __last_minicube_hor(multi_row, i, pixel_per_box, remaining_pixels, data, 
                        overlapping, catalogue, len_vert):
    """
    Calculates the last row of minicubes. Designed to calculate it with and 
    without overlap.

    Parameters
    ----------
    multi_row : int
        number to multiply the overlap of the down part of the row of minicubes.
    i : int
        number of minicubes - 1.
    pixel_per_box : int
        quotient of number of pixels in one direction over number of boxes.
    remaining_pixels : int
        remainder of the division.
    data : pd.DataFrame
        2D dataframe with data. The labels must be in pixels, starting from zero.
    overlapping : int
        number of pixels that will form the overlapping region.
    catalogue : pd.DataFrame
        catalogue with all the sources.
    len_vert : int
        length of the datacube.

    Returns
    -------
    data_mod_fin : pd.DataFrame
        small datacube with the mirrored part.

    """
    # Rows to fill the last datacubes
    lwr_lim = len_vert - (pixel_per_box - remaining_pixels)
    uppr_lim = len_vert
    
    esp = data.iloc[lwr_lim:uppr_lim, :]
    
    # Remaining rows
    lwr_lim2 = (i)*pixel_per_box-multi_row*overlapping
    data_mod_especial = data.iloc[lwr_lim2:, :] 
    
    # Take mirrored image
    esp = esp.iloc[::-1]

    # Transform the index (pixel number) to be a continuation of the last one
    first_esp_index = data_mod_especial.index.values[-1] + 1 
    new_index = np.arange(first_esp_index, first_esp_index + len(esp.index))
    esp.index = new_index
    
    # Sum the mirrored and the original images
    data_mod_fin = pd.concat([data_mod_especial, esp])
    
    # Catalogue actualization
    
    # Select sources within the limits of the brand new datacube
    cat = catalogue.loc[(catalogue['dec']>=lwr_lim) & (catalogue['dec']<= uppr_lim), :]
    
    # Mirror sources
    cat.loc[:, 'dec'] = - cat['dec'] + (len_vert - 1) * 2
    
    # Add them to the bottom of the original catalogue, with the index being 1 
    # more than the last index
    first_cat_index = catalogue.index.values[-1] + 1 
    new_cat_index = np.arange(first_cat_index, first_cat_index + len(cat.index))
    cat.index = new_cat_index      

    # Remove the duplicated galaxies in the catalogue, reseting the index
    catalogue_fin = pd.concat([catalogue, cat])
    catalogue_fin = catalogue_fin.drop_duplicates(subset=['ra', 'dec'])

    return data_mod_fin, catalogue_fin
    

def __last_minicube_vert(multi_col, var, pixel_per_box, remaining_pixels, data_mod, 
                         overlapping, catalogue, len_vert):
    """
    Calculates the last minicubes in a row. Designed to calculate it with and 
    without overlap.

    Parameters
    ----------
    multi_col : int
        number to multiply the overlap of the left part of the last minicube in 
        a row.
    i : int
        number of minicubes - 1.
    pixel_per_box : int
        quotient of number of pixels in one direction over number of boxes.
    remaining_pixels : int
        remainder of the division.
    data_mod : pd.DataFrame
        row of minicubes.
    overlapping : int
        number of pixels that will form the overlapping region..
    catalogue : pd.DataFrame
        catalogue with all the sources.
    len_vert : int
        length of the datacube.

    Returns
    -------
    data_mod_fin : pd.DataFrame
        small datacube with the mirrored part.

    """
    # Columns to fill the last datacubes
    left_lim =  len_vert - (pixel_per_box - remaining_pixels)
    right_lim = len_vert
    
    esp = data_mod.iloc[:, left_lim:right_lim]
    
    # Remaining columns
    mini_data_especial = data_mod.iloc[:, (var)*pixel_per_box - multi_col*overlapping:]
    
    # Take mirrored image
    esp = esp.iloc[:, ::-1]

    # Transform the index (pixel number) to be a continuation of the last one
    first_esp_index = mini_data_especial.columns.values[-1] + 1
    new_index = np.arange(first_esp_index, first_esp_index + len(esp.columns))
    esp.columns = new_index
    
    # Catalogue part
    
    # Select sources within the limits of the brand new datacube
    cat = catalogue.loc[(catalogue['ra']>=left_lim) & (catalogue['ra']<= right_lim), :]
    
    # Mirror sources
    cat.loc[:, 'ra'] = - cat['ra'] + (len_vert - 1) * 2
    
    # Add them to the bottom of the original catalogue, with the index being 1 
    # more than the last index
    first_cat_index = catalogue.index.values[-1] + 1
    new_index_cat = np.arange(first_cat_index, first_cat_index + len(cat.index))
    cat.index = new_index_cat
    
    # Remove the duplicated galaxies in the catalogue, reseting the index
    cat_fin = pd.concat([catalogue, cat])
    cat_fin = cat_fin.drop_duplicates(subset=['ra', 'dec'])

    return pd.concat([mini_data_especial, esp], axis=1), cat_fin
    
def overlap_bigger_than_remaining(number_of_boxes, pixel_per_box, remaining_pixels, 
                                  data, overlapping, multi_row, multi_col, catalogue, 
                                  len_vert):
    """
    Algorithm to split the full datacube when the number of pixels in the overlapping 
    part is bigger than the number of pixels w/o mini datacube. This remaining 
    pixels will form a mini datacube with the mirrored image of the full datacube 
    (pixels of the mirrored image are taken until the last mini cube is full). 
    All the mini datacubes will have the same dimensions. Overlaps in the borders 
    of the datacube are not considered. Instead a double sized overlap is 
    considered in the borders of the minicube which are not in the border of 
    the full datacube.

    Parameters
    ----------
    number_of_boxes : int
        number of boxes to divide the dataset in each direction..
    pixel_per_box : int
        quotient of number of pixels in one direction over number of boxes.
    remaining_pixels : int
        remainder of the division.
    data : pd.DataFrame
        2D dataframe with data. The labels must be in pixels, starting from zero
    overlapping : int
        number of pixels that will form the overlapping region.
    multi_row : np.ndarray
        array with the numbers to multiply the horizontal overlap. First element 
        multiplies the left overlap and second the right one
    multi_col : np.ndarray
        array with the numbers to multiply the vertical overlap. First element 
        multiplies the overlap above and second the below one.
    catalogue : pd.DataFrame
        catalogue with all the sources.
    len_vert : int
        length of the datacube.
    Returns
    -------
    data_splitted : list
        list containing all the splitted data. Each element of the list, i.e., 
        each small datacube has the same dimensions.

    """
    data_splitted = []
    truth_cat_splitted = []
    # Loop over the rows
    for i in range(number_of_boxes + 1):
        # If the overlap of the last full minicube exceeds the number of pixels 
        # remaining, some pixels of the datacube with a mirrored part must be taken. 
        # So it should be calculated (w/o overlap)
        if i == number_of_boxes - 1: 
            multi_row[:] = [0, 0]
            # Calculate last minicube w/o overlap
            data_mod_fin, catalogue = __last_minicube_hor(multi_row[1], i+1, pixel_per_box, 
                                                          remaining_pixels, data, overlapping,
                                                          catalogue, len_vert)
            
            # Calculate (i+1) th datacube but w/o the overlap that will affect
            # the last minicube, only with the regular one.
            multi_row[:] = [1, 1]
            limit1 = (i * pixel_per_box) - multi_row[0] * overlapping 
            limit2 = (i + 1) * (pixel_per_box)
            data_mod_1 = data.iloc[limit1 : limit2, :]
            
            # The other overlap is taken from the mirrored datacube previously created
            data_mod_2 = data_mod_fin.iloc[:overlapping , :]
            
            # Finally, they are merged into one dataframe
            data_mod = pd.concat([data_mod_1, data_mod_2])
            
        # The remaining pixels are considered, with a mirrored part, to form a 
        # full size minicube. The overlapping is doubled in the regular borders 
        #to compensate the border of the full datacube
        elif i == number_of_boxes:
            multi_row[:] = [2, 0]
            data_mod, catalogue = __last_minicube_hor(multi_row[0], i, pixel_per_box, 
                                                      remaining_pixels, data, overlapping, 
                                                      catalogue, len_vert)
        else: 
            if i != 0:  # Regular rows
                multi_row[:] = [1, 1]
            
            # If it is not the last row it selects the pixels using the number of 
            # pixels per cube parameter
            limit1 = (i * pixel_per_box) - multi_row[0] * overlapping
            limit2 = (i + 1) * pixel_per_box + multi_row[1] * overlapping
            data_mod = data.iloc[limit1 : limit2, :]
        # Same but for columns in the selected row
        for var in range(number_of_boxes + 1):
            if var == number_of_boxes-1:
                multi_col[:] = [0, 0]
                mini_data_fin, catalogue = __last_minicube_vert(multi_col[1], var+1, 
                                                                pixel_per_box, 
                                                                remaining_pixels, 
                                                                data_mod, overlapping, 
                                                                catalogue, len_vert)
                multi_col[:] = [1, 1]
                
                limit1 = (var * pixel_per_box) - multi_col[1] * overlapping
                limit2 = (var + 1) * (pixel_per_box)
                mini_data_1 = data_mod.iloc[:, limit1 : limit2]
                mini_data_2 = mini_data_fin.iloc[: , :overlapping]
                mini_data = pd.concat([mini_data_1, mini_data_2], axis=1)

            elif var == number_of_boxes:
                multi_col[:] = [2, 0]
                mini_data_fin, catalogue = __last_minicube_vert(multi_col[0], var, 
                                                                pixel_per_box, 
                                                                remaining_pixels, 
                                                                data_mod, overlapping, 
                                                                catalogue, len_vert)
                mini_data = mini_data_fin
                
            else: 
                if var == 0:
                    multi_col[:] = [0, 2] 
                else:
                    multi_col[:] = [1, 1] 
                limit1 = var * pixel_per_box - overlapping * multi_col[0]
                limit2 = (var + 1) * pixel_per_box + overlapping * multi_col[1]
                
                mini_data = data_mod.iloc[:, limit1 : limit2]
            data_splitted.append(mini_data)
            
            # Catalogue splitting. Galaxies in the overlapped region are not 
            # considered as they may be cut by the border of the datacube
            limit_pixels_vert = (mini_data.index.values[0] + overlapping, 
                                 mini_data.index.values[-1] - overlapping)
            limit_pixels_hor = (mini_data.columns.values[0] + overlapping, 
                                mini_data.columns.values[-1] - overlapping)
            
            # Loop over every source in the catalogue. If it's inside the small 
            # cubes it saves the source in a different dataframe
            truth_cat_small = catalogue.loc[(catalogue['ra']>=limit_pixels_hor[0]) & 
                                             (catalogue['ra']<= limit_pixels_hor[1]) & 
                                              (catalogue['dec']>=limit_pixels_vert[0]) & 
                                              (catalogue['dec']<= limit_pixels_vert[1]), :]
            # When the loop is completed, the table is stored                    
            truth_cat_splitted.append(truth_cat_small)
    return data_splitted, truth_cat_splitted


def overlap_smaller_than_remaining(number_of_boxes, pixel_per_box, remaining_pixels, data, 
                                   overlapping, multi_row, multi_col, catalogue, len_vert):
    """
    Algorithm to split the full datacube when the number of pixels in the overlapping 
    part is smaller than the number of pixels w/o mini datacube. This remaining 
    pixels will form a mini datacube with the mirrored image of the full datacube 
    (pixels of the mirrored image are taken until the last mini cube is full). 
    All the mini datacubes will have the same dimensions. Overlaps in the borders 
    of the datacube are not considered. Instead a double size overlap is 
    considered in the borders of the minicube which are not in the border 
    of the full datacube.

    Parameters
    ----------
    number_of_boxes : int
        number of boxes to divide the dataset in each direction..
    pixel_per_box : int
        quotient of number of pixels in one direction over number of boxes.
    remaining_pixels : int
        remainder of the division.
    data : pd.DataFrame
        2D dataframe with data. The labels must be in pixels, starting from zero
    overlapping : int
        number of pixels that will form the overlapping region.
    multi_row : np.ndarray
        array with the numbers to multiply the horizontal overlap. First element
        multiplies the left overlap and second the right one
    multi_col : np.ndarray
        array with the numbers to multiply the vertical overlap. First element
        multiplies the overlap above and second the below one.
    catalogue : pd.DataFrame
        catalogue with all the sources.
    len_vert : int
        length of the datacube.
    Returns
    -------
    data_splitted : list
        list containing all the splitted data. Each element of the list, i.e.,
        each small datacube has the same dimensions

    """
    
    data_splitted = []
    truth_cat_splitted = []
    for i in range(number_of_boxes + 1):
        overlapping_borde = overlapping

        if i == number_of_boxes: 
            multi_row[:] = [2, 0]
            data_mod, catalogue = __last_minicube_hor(multi_row[0], i, pixel_per_box, 
                                                      remaining_pixels, data, overlapping, 
                                                      catalogue, len_vert)  
            
        else: 
            if i != 0: 
                multi_row[:] = [1, 1]
            if i == 0:
                overlapping_borde = 0
                
            # If it is not the last row it selects the pixels using the number 
            # of pixels per cube parameter
            limit1 = (i * pixel_per_box) - multi_row[0] * overlapping
            limit2 = (i + 1) * pixel_per_box + multi_row[1] * overlapping
            data_mod = data.iloc[limit1 : limit2, :]
       
        # Loop over each column in a row. Same procedure as before but with columns
        for var in range(number_of_boxes + 1):
            overlapping_borde_col = overlapping
            if var == number_of_boxes:
                multi_col[:] = [2, 0]
                mini_data, catalogue = __last_minicube_vert(multi_col[0], var, 
                                                            pixel_per_box, 
                                                            remaining_pixels, 
                                                            data_mod, overlapping, 
                                                            catalogue, len_vert)
            else:
                if var != 0:
                    multi_col[:] = [1, 1] 
                    
                else:
                    multi_col[:] = [0, 2]
                    
                    overlapping_borde_col = 0
                limit1 = var * pixel_per_box - overlapping * multi_col[0]
                limit2 = (var + 1) * pixel_per_box + overlapping * multi_col[1]
                mini_data = data_mod.iloc[:, limit1 : limit2]
            data_splitted.append(mini_data)
            # Catalogue splitting. Galaxies in the overlapped region are not 
            # considered as they may be cut by the border of the datacube
            limit_pixels_vert = (mini_data.index.values[0] + overlapping_borde, 
                                 mini_data.index.values[-1] - overlapping)
            limit_pixels_hor = (mini_data.columns.values[0] + overlapping_borde_col, 
                                mini_data.columns.values[-1] - overlapping)
            
            # Loop over every source in the catalogue. If it's inside the small 
            # cubes it saves the source in a different dataframe
            truth_cat_small = catalogue.loc[(catalogue['ra']>=limit_pixels_hor[0]) & 
                                             (catalogue['ra']<= limit_pixels_hor[1]+1) & 
                                              (catalogue['dec']>=limit_pixels_vert[0]) &
                                              (catalogue['dec']<= limit_pixels_vert[1]+1), :]
            
            # When the loop is completed, the table is stored                    
            truth_cat_splitted.append(truth_cat_small)
    return data_splitted, truth_cat_splitted
    
def no_remaining_pixels(number_of_boxes, pixel_per_box, remaining_pixels, data, overlapping,
                        multi_row, multi_col, catalogue):
    """
    Algorithm to split the full datacube when the number of pixels in any direction is a multiple of 
    the number of datacubes. All the mini datacubes will have the same dimensions. 
    Overlaps in the borders of the datacube are not considered. Instead a double size overlap is 
    considered in the borders of the minicube which are not in the border of the full datacube.

    Parameters
    ----------
    number_of_boxes : int
        number of boxes to divide the dataset in each direction.
    pixel_per_box : int
        quotient of number of pixels in one direction over number of boxes.
    remaining_pixels : int
        remainder of the division.
    data : pd.DataFrame
        2D dataframe with data. The labels must be in pixels, starting from zero
    overlapping : int
        number of pixels that will form the overlapping region.
    multi_row : np.ndarray
        array with the numbers to multiply the horizontal overlap. First element
        multiplies the left overlap and second the right one
    multi_col : np.ndarray
        array with the numbers to multiply the vertical overlap. First element
        multiplies the overlap above and second the below one.
    catalogue : pd.DataFrame
        catalogue with all the sources.
    len_vert : int
        length of the datacube.
    Returns
    -------
    data_splitted : list
        list containing all the splitted data. Each element of the list, i.e.,
        each small datacube has the same dimensions

    """
    data_splitted = []
    truth_cat_splitted = []
    
    # Loop over all the rows
    for i in range(number_of_boxes):
        # If the last row is considered, the overlapping will be on the inner
        # part of the small datacube. The same is done for the first row
        if i == number_of_boxes - 1:
            multi_row[:] = [2, 0]
        else: 
            if i != 0:
                multi_row[:] = [1, 1]
                
        # Extract the row from the datacube
        data_mod = data.iloc[(i * pixel_per_box) - overlapping * multi_row[0] : 
                             (i + 1) * pixel_per_box + overlapping * multi_row[1], :]
            
        # Same procedure but for columns
        for var in range(number_of_boxes):
            if var == number_of_boxes-1:
                multi_col[:] = [2, 0]
            else: 
                if var != 0:
                    multi_col[:] = [1, 1] 
                else:
                    multi_col[:] = [0, 2]        
            mini_data = data_mod.iloc[:, var * pixel_per_box - overlapping * multi_col[0] : 
                                      (var + 1) * pixel_per_box + overlapping * multi_col[1]]
            data_splitted.append(mini_data)
            
            # Catalogue splitting. Galaxies in the overlapped region are not
            # considered as they may be cut by the border of the datacube
            limit_pixels_vert = (mini_data.index.values[0]+overlapping, 
                                 mini_data.index.values[-1]-overlapping)
            limit_pixels_hor = (mini_data.columns.values[0]+overlapping,
                                mini_data.columns.values[-1]-overlapping)
            
            # Loop over every source in the catalogue. If it's inside the small cubes it saves the source in a different dataframe
            truth_cat_small = catalogue.loc[(catalogue['ra']>=limit_pixels_hor[0]) &
                                             (catalogue['ra']<= limit_pixels_hor[1]) & 
                                              (catalogue['dec']>=limit_pixels_vert[0]) &
                                              (catalogue['dec']<= limit_pixels_vert[1]), :]
            
            # When the loop is completed, the table is saved                    
            truth_cat_splitted.append(truth_cat_small)      
    return data_splitted, truth_cat_splitted
        

def split_into_small_boxes(data, final_pixel_per_box, catalogue, overlapping='auto'):
    """
    Splits a 2D datacube into the (number of boxes * number of boxes) given as parameter. 
    It also considers overlapping to include galaxies just in the boundary of 
    the small datacubes. The catalogue for the sources is also splitted.

    Parameters
    ----------
    data : pd.DataFrame
        2D dataframe with data.
    final_pixel_per_box : int
        size of the small datacube in any direction, including overlap.
    catalogue: pd.DataFrame
        catalogue of the sources in the full datacube. 'ra' and 'dec' column 
        must be given in pixels
    overlapping : int, optional
        number of extra entries to consider in each minicube, given in pixels.
        'auto' option uses an overlap equal to the maximum size of a galaxy in
        the whole datacube. The default is 'auto'.

    Raises
    ------
    DimensionError
        if number of boxes is bigger than the number of entries in any direction.
        if the remaining pixels are bigger than the length of a small datacube
        if overlapping is bigger than the size of a small datacube

    Returns
    -------
    data_splitted : list
        list with the small dataframes.
    truth_cat_splitted : list
        list with catalogues for each of the small datacubes

    """
    pd.set_option('mode.chained_assignment',None)

    len_cube = len(data.iloc[:, 0])
    if overlapping == 'auto':
        overlapping = int(np.max(catalogue['hi_size'])/2)
        
    pixel_per_box = final_pixel_per_box - overlapping * 2
    # Number to multiply the overlapping. First element is start and second is
    # stop in the .iloc[]
    multi_row = np.array([0, 2], dtype=int)
    multi_col = np.array([0, 2], dtype=int)  
    
    number_of_boxes, remaining_pixels = divmod(len_cube, pixel_per_box) 
    
   

    if overlapping > pixel_per_box:
        warnings.warn("The biggest galaxy size is bigger than the pixels per "+
                      'small box, '+
                      'which has a value of {} pixels.'.format(pixel_per_box) +
                      'The program will use the number of pixels per box - 1 as overlapping')

        overlapping = pixel_per_box - 1

    if pixel_per_box <= remaining_pixels:
        raise DimensionError('Too many small datacubes for the lenght of the dataset. '
                             +'Try with a smaller number')

    if number_of_boxes > len_cube or overlapping > len_cube:
        raise DimensionError('The number of boxes selected is bigger than the height '
                             +'or width of the data table')
    
    if overlapping > pixel_per_box:
        raise DimensionError('The overlapping selected exceeds the length of a minicube. '
                             +'Please, try with smaller overlapping or less minicubes per line')
    
    # Chooses one of the three available functions, depending on the number of pixels remaining
    if remaining_pixels == 0: 
        return no_remaining_pixels(number_of_boxes, pixel_per_box, remaining_pixels, 
                                   data, overlapping, multi_row, multi_col, catalogue)    
    
    elif overlapping > remaining_pixels:
        a, b = overlap_bigger_than_remaining(number_of_boxes, pixel_per_box, 
                                             remaining_pixels, data, overlapping, 
                                             multi_row, multi_col, catalogue, len_cube)
        
        return a, b
         
    else:
        a, b = overlap_smaller_than_remaining(number_of_boxes, pixel_per_box, 
                                              remaining_pixels, data, overlapping, 
                                              multi_row, multi_col, catalogue, len_cube)
        
        return a, b




 