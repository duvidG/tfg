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

def generate_galaxies(small_datacube, small_catalogue, wcs, length):
    """
    Generates an array with the labels to train the NN in a small datacube. Every
    pixel that contains a galaxy is labelled as 1.

    Parameters
    ----------
    small_datacube : pd.DataFrame
        array with the data.
    small_catalogue : pd.DataFrame
        array with the sources.
    major_semiaxis : float
        defaul major semiaxis, used to generate the galaxy.

    Returns
    -------
    total_labels : pd.DataFrame
        sum of all the images of the generated galaxies. Each pixel which value is bigger
        than 1 is considered as 1. The rest are set to 0

    """
    #generated_galaxies = []
    total_labels = 0
    ind = small_datacube.index.values
    col = small_datacube.columns.values
    x,y = np.meshgrid(np.arange(col[0], col[-1]), np.arange(ind[0], ind[-1]))
    for i in small_catalogue.index:

        ra = small_catalogue.loc[i, 'ra']
        dec = small_catalogue.loc[i, 'dec']
        inc = small_catalogue.loc[i, 'i']
        theta = small_catalogue.loc[i, 'pa']
        if length < dec or length < ra:
            theta = theta + np.pi/2
        if length < dec and length < ra:
            theta = theta + np.pi/2
            

        major_semi = small_catalogue.loc[i, 'hi_size']
        
        minor_semi = major_semi * np.cos(inc)
        # print(ra, dec)
        model = models.Ellipse2D(1, ra, dec, major_semi, minor_semi, theta)
        img = pd.DataFrame(model(x, y))
        #generated_galaxies.append(img) # Guardar imagenes ind
        total_labels = total_labels + img
    # Asign a 1 value to data bigger than 1 and 0 to rest    
    #total_labels = reduce(lambda a, b: a.add(b, fill_value=0), generated_galaxies)
    
    total_labels = total_labels/total_labels.max().max()
    total_labels = total_labels.apply(np.ceil)
    # generated_galaxies, meter en el return para imag galax individual
    return total_labels

def calculate_major_semiaxis(aperture_ang, wcs):
    """
    Transforms the aperture angle, in arcsec, of a galaxy into pixels.

    Parameters
    ----------
    aperture_ang : float
        aperture angle of the galaxies in the telescope.
    wcs : object
        rules to transform sky coordinates to pixels.

    Returns
    -------
    major_semiaxis : (float, float)
        equivalent, in pixels, of the aperture angle, for both directions RA and Dec.

    """
    
    # Transform aperture angle into pixels in both RA and Dec scale --> Coordinates 
    # (in pixels) of point (1, 1) in (ra, dec)
    ra_aperture = Angle(aperture_ang/3600, unit=u.degree) 
    dec_aperture = Angle(aperture_ang/3600, unit=u.degree)
    
    origin = wcs.pixel_to_world(0, 0)
    
    # Check in which direction does the ra increase
    sig_ra = 1
    if wcs.pixel_to_world(1, 0).ra < ra_aperture:
        sig_ra = -1
    
    coords = SkyCoord(origin.ra + sig_ra * ra_aperture, origin.dec+dec_aperture)
    major_semiaxis = wcs.world_to_pixel(coords)
    return major_semiaxis