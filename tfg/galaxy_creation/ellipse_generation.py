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


def generate_galaxies(small_datacube, small_catalogue, major_semiaxis):
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
    generated_galaxies = []
    total_labels = 0
    x,y = np.meshgrid(np.arange(len(small_datacube.iloc[0, :])), np.arange(len(small_datacube.iloc[:, 0])))

    for i in small_catalogue.index:

        ra = small_catalogue.loc[i, 'ra']
        dec = small_catalogue.loc[i, 'dec']
        inc = small_catalogue.loc[i, 'i']
        theta = small_catalogue.loc[i, 'pa']
        
        major_semi = np.sqrt( (major_semiaxis[0] * np.cos(theta) ) ** 2 + (major_semiaxis[1] * np.sin(theta)) ** 2 )
        minor_semi = major_semi * np.cos(inc)
        
        model = models.Ellipse2D(1, ra, dec, major_semi, minor_semi, theta)
        img = pd.DataFrame(model(x, y))
        generated_galaxies.append(img) # Guardar imagenes ind
        total_labels = total_labels + img
    # Asign a 1 value to data bigger than 1 and 0 to rest    
    #total_labels = reduce(lambda a, b: a.add(b, fill_value=0), generated_galaxies)
    
    total_labels = total_labels/total_labels.max().max()
    total_labels = total_labels.apply(np.ceil)
    # generated_galaxies, meter en el return para imag galax individual
    return total_labels, generated_galaxies

def calculate_major_semiaxis(aperture_ang, wcs):
    """
    Transforms the aperture angle, in arcsec, of a default galaxy into pixels.

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






# def generate_galaxies(small_datacube, small_catalogue, major_semiaxis):
#     generated_galaxies = []
#     plt.close('all')
#     x,y = np.meshgrid(np.arange(44), np.arange(44))
#     small_datacube = small_datacube*1000000
#     for i in small_catalogue.index:

#         ra = small_catalogue.loc[i, 'ra']
#         dec = small_catalogue.loc[i, 'dec']
#         inc = small_catalogue.loc[i, 'i']
#         theta = small_catalogue.loc[i, 'pa']

#         i_r0 = small_datacube.loc[int(ra), int(dec)]
#         diff_vert = np.abs(i_r0 - small_datacube.loc[:, int(dec)])
#         diff_hor = np.abs(i_r0 - small_datacube.loc[int(ra), :])

#         r_vert = np.abs(diff_vert.idxmin() - dec)
#         r_hor = np.abs(diff_hor.idxmin() - ra)



#         r_eff = (r_vert + r_hor)/2

#         mod = models.Sersic2D(1, r_eff, 4, ra, dec, 1-np.cos(inc), theta)
#         generated_galaxies.append(mod)

#         img = mod(x, y)
#         log_img = np.log10(img)

#         plt.figure()
#         plt.imshow(log_img, origin='lower', interpolation='nearest')
#         plt.colorbar()
#     return generated_galaxies