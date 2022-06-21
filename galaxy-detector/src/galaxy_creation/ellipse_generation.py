#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 12:49:58 2022

@author: dgiron
"""

import pandas as pd
import numpy as np

from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.modeling import models

import matplotlib.pyplot as plt

def add_masks(img, total_labels):
    """
    Add image to the array storing the sum of images. Special treatment is
    done with galaxies that overlap other galaxies. The border of the galaxies
    is preserved, having a higher priority than the inside part of other galaxies.
    At the end, every pixels' value is contained in the interval [0, 1, 2], meaning 
    (background, inside galaxy, border)

    Parameters
    ----------
    img : pd.DataFrame
        array of the same size of total_labels array. It contains the image of 
        the generated galaxy, with 1 being the interior pixels and 2 the borders
    total_labels : pd.DataFrame
        array to store the addition.

    Returns
    -------
    total_labels : pd.DataFrame
        updated array.

    """
    
    suma = img + total_labels > 2
    # Check if any galaxy overlaps an existent one
    if suma.any(axis=None):
        # Transform borders value to 3, to differentiate it from the overlapping of
        # inside parts of galaxies, whose addition equals 2 as well.
        img = img.mask(img == 2, 3)
        # Change inside parts from 1 to 0.5
        img = img.mask(img == 1, 0.5)
        # Add images
        total_labels = total_labels + img
        # Change 1.5 values (inside + inside) to 1, in order to remain 1 when the np.ceil function 
        # is called
        total_labels = total_labels.mask(total_labels == 1.5, 1)
    else:
        total_labels = total_labels + img
    # Converts all values to [0, 1, 2]
    total_labels = total_labels.apply(np.ceil).clip(upper=2)
    
    return total_labels
    
    
def generate_galaxies(small_datacube, small_catalogue, length, min_major_length=0., min_minor_length=0.):
    """
    Generates an array with the labels to train the NN in a small datacube. Every
    pixel that contains a galaxy is labelled as 1, with the exception of borders 
    (when the sum of the surrounding pixels is less than 8, i.e., one of the eight 
    surrounding pixels is labelled as 0) which are labelled as 2. As small galaxies
    (less than (major semiaxis, minor semiaxis) = (0.7, 0.5)) are not created by 
    astropy.ellipse2d function, a couple of parameters are available to set the 
    minimum size that will be created. If a galaxy is bigger than those, but smaller 
    than the astropy minimum size, its size will be approximated to (0.7, 0.5).

    Parameters
    ----------
    small_datacube : pd.DataFrame
        array with the data.
    small_catalogue : pd.DataFrame
        array with the sources.
    length : int
        length of the original datacube, without the mirrored part.
    min_major_length : float, optional
        minimum major semiaxis length for galaxies to be considered, in pixels. 
        The default is 0..
    min_minor_length : TYPE, optional
        minimum major semiaxis length for galaxies to be considered, in pixels. 
        The default is 0..

    Returns
    -------
    total_labels : pd.DataFrame
        sum of all the images of the generated galaxies. Each pixel which value 
        is bigger than 1 is considered as 1. The rest are set to 0

    """
    total_labels = 0
    ind = small_datacube.index.values
    col = small_datacube.columns.values
    x,y = np.meshgrid(np.arange(col[0], col[-1]+1), np.arange(ind[0], ind[-1]+1))
    
    # If the catalogue is empty, it returns a cube full of zeros
    if len(small_catalogue.index) == 0:
        model = models.Ellipse2D(0, 0, 0, 1, 1, 0)
        img = pd.DataFrame(model(x, y))
        total_labels = img
        return total_labels
    # Loop over all the galaxies in the catalogue
    for i in small_catalogue.index:
        ra = small_catalogue.loc[i, 'ra']
        dec = small_catalogue.loc[i, 'dec']
        inc = small_catalogue.loc[i, 'i']
        theta = small_catalogue.loc[i, 'pa']
        
        # Change orientation if the sources come from the mirrored image
        if length < dec and length < ra:
            theta = theta + np.pi
        elif length < dec or length < ra:
            theta = theta + np.pi/2
            
        alpha = 0.2
        major_semi = small_catalogue.loc[i, 'hi_size']/2
        # Calculate minor semiaxis with inclination
        
        minor_semi = major_semi * np.sqrt((1 - alpha ** 2) * np.cos(inc) ** 2 + alpha**2)
        
        
        # Evaluate the galaxy size with the parameters provided. If it is smaller
        # than them will not be created by the Ellipse2D function.
        if np.abs(major_semi) < 0.75 and np.abs(major_semi) > min_major_length:
            major_semi = 0.75
        
        if np.abs(minor_semi) < 0.5 and np.abs(minor_semi) > min_minor_length:
            minor_semi =  0.51
            
        # Generate the galaxy
        model = models.Ellipse2D(1, ra, dec, major_semi, minor_semi, theta + np.pi/2)
        img = pd.DataFrame(model(x, y), index = y[:, 0], columns = x[0])
        
        # Consider all the values greater than 0 to be 1
        img = img.apply(np.ceil)
        
        #  Narrow down the area to search for borders
        inters = [ra - major_semi, ra + major_semi, dec - major_semi, dec + major_semi]
        img_mod = img.loc[inters[2]:inters[3],  inters[0]:inters[1]]
        
        # Galaxy borders
        border = pd.DataFrame([[2 if img_mod.iloc[ind-1:ind+2, col-1:col+2].values.sum()
                                <= 8 and img_mod.iloc[ind, col] == 1 else 1 
                                if img_mod.iloc[ind, col] == 1 else 0 
                                for col in range(len(img_mod.columns.values))] 
                               for ind in range(len(img_mod.index.values))],
                              index=img_mod.index.values, columns=img_mod.columns.values)
        # Update the original image
        img.loc[inters[2]:inters[3],  inters[0]:inters[1]] = border
        # Add the created image to the variable that stores the total sum of the datacube
        total_labels = add_masks(img, total_labels)
        
    return total_labels
