#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 16:29:49 2022

@author: dgiron
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
import matplotlib.colors as colors
import pandas as pd
import numpy as np
import os
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle 
from astropy import units as u
from matplotlib.patches import Ellipse

from astropy.io.fits import PrimaryHDU

plt.style.use(astropy_mpl_style)

cmap = plt.cm.winter

def create_mask(pred_mask, idx, cols, thr, sdss_pixelscale, wcs, img_num):
    """
    Creates the final mask from the three predicted ones, with labels [0, 1, 2]

    Parameters
    ----------
    pred_mask : list(pd.DataFrame)
        list with the output of the NN, i.e., the 3 masks, to transform coordinates 
        from the minicube to the full cube.
    idx : list
        list with the index of the image.
    cols : list
        list with the column names of the image, to transform coordinates 
        from the minicube to the full cube.
    thr : float
        minimum value to consider pixels as part of galaxies instead of background.
    sdss_pixelscale : float
        degrees per pixel, using astropy units module.
    wcs : WCS
        header of the FITS file, as an object of WCS class, from astropy.
    img_num : int, optional
        number of the image. The default is 0.

    Returns
    -------
    pd.DataFrame
        predicted mask.
    pd.DataFrame
        predicted catalogue, with the same format as the original.
    pd.DataFrame
        predicted catalogue, with everything in pixels.

    """
    galaxy_pixels = pred_mask[:, :, 1] + pred_mask[:, :, 2]
    galaxy_pixels = pd.DataFrame(galaxy_pixels)
    
    # Pixels with prob bigger than thr are considered galaxy
    galaxy_pixels = galaxy_pixels.mask(galaxy_pixels >= thr, 1)

    # The rest are considered background
    galaxy_pixels = galaxy_pixels.apply(np.floor)

    t = PrimaryHDU(galaxy_pixels.to_numpy())
    t.writeto('src/output/mask'+str(img_num)+'.fits', overwrite=True)
    
    os.system('sex src/output/mask'+str(img_num)+'.fits -c src/output/daofind.sex')
    try:
        mini_catalogue = pd.read_csv('src/output/test.cat', delimiter = ' ', skipinitialspace=True, 
                                     header=None, skiprows=(5), names=['ra', 'dec', 'hi_size', 'i', 'pa'])
    except:
        return galaxy_pixels, pd.DataFrame([], ['ra', 'dec', 'hi_size', 'i', 'pa'])
    else:
        mini_catalogue = mini_catalogue.apply(pd.to_numeric)
        mini_catalogue_pix = mini_catalogue.copy()
        mini_catalogue['ra'] = mini_catalogue['ra'] + cols[0]
        mini_catalogue['dec'] = mini_catalogue['dec'] + idx[0]

        # Transform coordinates in the catalogue to pixels
        truth_pixels = wcs.pixel_to_world(mini_catalogue['ra'], mini_catalogue['dec'])
        
        mini_catalogue['dec'] = truth_pixels.dec.degree
        mini_catalogue['ra'] = truth_pixels.ra.degree
        
        # Transform major_semiaxis in the catalogue to pixels. Assumes same pixel scale for both axis
        mini_catalogue['hi_size'] = [(i * u.pixel).to(u.arcsec, sdss_pixelscale).value for i in mini_catalogue['hi_size']]
        mini_catalogue['i'] = [(i * u.pixel).to(u.arcsec, sdss_pixelscale).value for i in mini_catalogue['i']]
        mini_catalogue['i'] = [np.sqrt(((i/hi_size)**2 - 0.2**2)/(1-0.2**2)) for i, hi_size in zip(mini_catalogue['i'], mini_catalogue['hi_size'])]
        mini_catalogue_pix_2 = mini_catalogue_pix.copy()
        mini_catalogue_pix_2['i'] = [(180/np.pi)*np.arccos(np.sqrt(((i/hi_size)**2 - 0.2**2)/(1-0.2**2))) for i, hi_size in zip(mini_catalogue_pix['i'], mini_catalogue_pix['hi_size'])]

        os.remove("src/output/test.cat")
        if img_num == 115:
            print(mini_catalogue_pix_2.to_latex(escape=False))
            
        return galaxy_pixels, mini_catalogue, mini_catalogue_pix

def display(display_list, img_num, small_cat, new_catalogue, lth=1e-5, cmap=cmap):
    fig, axs = plt.subplots(1, 1)
    
    img = display_list[2]
    im = axs.imshow(img, origin='lower')
    fig.colorbar(im, cmap=cmap)
    for k in new_catalogue.index.values:
        ellipse = Ellipse((new_catalogue.loc[k, 'ra'], new_catalogue.loc[k, 'dec']), 
                          2*new_catalogue.loc[k, 'hi_size'], 2*new_catalogue.loc[k, 'i'], 
                          new_catalogue.loc[k, 'pa'], edgecolor='red', facecolor='none')
        axs.add_patch(ellipse)
    axs.set_title('Predicted mask')
            
        
    
def draw_predictions(predictions, img, real_mask, small_cat, thr, sdss_pixelscale, wcs, img_num=0):
    """
    Creates the mask from the prediction masks and draws a figure comparing the 
    real image and both masks (original and predicted)

    Parameters
    ----------
    predictions : list
        list with the 3 masks.
    img : pd.DataFrame
        original small image.
    real_mask : pd.DataFrame
        mask generated from the catalogue.
    small_cat : pd.DataFrame
        catalogue of the small image.
    thr : float
        minimum value to consider pixels as part of galaxies instead of background.
    sdss_pixelscale : float
        degrees per pixel, using astropy units module.
    wcs : WCS
        header of the FITS file, as an object of WCS class, from astropy.
    img_num : int, optional
        number of the image. The default is 0.

    Returns
    -------
    new_catalogue : pd.DataFrame
        predicted catalogue.

    """
    
    new_mask, new_catalogue, new_cat_pix = create_mask(predictions, img.index.values, img.columns.values,
                                          thr, sdss_pixelscale, wcs, img_num)
        
    
    if img_num+1 == 116:
        display([img, real_mask, new_mask], img_num, small_cat, new_cat_pix)
        plt.savefig('../informe/imgs/resultados_comparacion_3.png')
        plt.figure()
        plt.imshow(predictions[:, :, 0], origin='lower')
        plt.colorbar()
        plt.savefig('../informe/imgs/nada.png')
        
    else:
        plt.figure()
        plt.imshow(new_mask, origin='lower')
        plt.savefig('imgs/prueba'+str(img_num)+'.png')
        plt.close('all')
    return new_catalogue

    # Esta parte va fuera cuando el thr funcione y lo de arriba de bien la mask
    # predictions = predictions[0]
    # plt.figure()
    # plt.imshow(predictions[:, :, 0], origin='lower')
    # plt.colorbar()
    # plt.savefig('imgs/fondo'+str(img_num)+'.png')
    # plt.figure()
    # plt.imshow(predictions[:, :, 1], origin='lower')
    # plt.colorbar()
    # plt.savefig('imgs/interior'+str(img_num)+'.png')
    # plt.figure()
    # plt.imshow(predictions[:, :, 2], origin='lower')
    # plt.colorbar()
    # plt.savefig('imgs/borde'+str(img_num)+'.png')
    
    

            
# def display(display_list, img_num, small_cat, lth=1e-5, cmap=cmap):
#     """
#     Plots a figure with the real image, and both masks, the one generated from
#     the catalogue and the predicted one.

#     Parameters
#     ----------
#     display_list : list
#         list containing the 3 images.
#     img_num : int
#         number of datacube.
#     small_cat : pd.DataFrame
#         catalogue of the small image.
#     lth : float, optional
#         lth parameter of matplotlib normalization. The default is 1e-5.
#     cmap : plt.cmap or srt, optional
#         selected colormap for the plot. The default is cmap.

#     Returns
#     -------
#     None.

#     """
#     plt.figure(figsize=(15, 15))
    
#     title = ['Input Image', 'True Mask', 'Predicted Mask']
    
#     for i in range(len(display_list)):
#         plt.subplot(1, len(display_list), i+1)
#         plt.title(title[i])
#         if  i == 0:
#             img = display_list[i]     
#             ra_min, dec_min = img.columns.values[0], img.index.values[0]

#             plt.imshow(img, cmap=cmap, norm=colors.SymLogNorm(linthresh=lth), origin='lower')
#             plt.plot(small_cat['ra']- ra_min, small_cat['dec']- dec_min, 'rx')

#             plt.colorbar()
#         elif i == 2:
#             img = display_list[i]
#             plt.imshow(img, origin='lower')
#             plt.colorbar()
            
#         else:
#             img = display_list[i]
#             plt.imshow(img[0, :, :], origin='lower') 
#             plt.title('Datacube: {}' .format(img_num))
#             plt.colorbar()    
    