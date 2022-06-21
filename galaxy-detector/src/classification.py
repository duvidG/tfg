#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 11:45:39 2022

@author: dgiron
"""


import pandas as pd
import numpy as np

from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle  
from astropy.visualization import astropy_mpl_style
from astropy.wcs.utils import proj_plane_pixel_scales

import matplotlib.pyplot as plt
import matplotlib.colors as colors

import tensorflow as tf
from tensorflow.keras import layers

from keras.callbacks import EarlyStopping

from src.nn_model.models import *
from src.read_data.save_data import read_data
from src.data_modifications.algorithms import mean_frec
from src.data_modifications.data_splitting import split_into_small_boxes
from src.galaxy_creation.ellipse_generation import generate_galaxies
from src.mask_prediction.draw_mask import draw_predictions

from skimage.transform import resize

# Scoring software

from src.scoring.ska_sdc.sdc2.sdc2_scorer import Sdc2Scorer

plt.style.use(astropy_mpl_style)

cmap = plt.cm.winter

class GalaxyDetector():
    def __init__(self, data_file, cube_dimensions, overlapping, 
                 function_to_reduce_data, truth_cat, data_file_ps, min_major_length, 
                 min_minor_length, flujo_min):
        """
        Creates the object

        Parameters
        ----------
        data_file : str
            route to the continuum emission datafile.
        cube_dimensions : int
            desired size of images, in pixels.
        overlapping : int or 'auto'
            number of pixels to include in the overlapping region. If 'auto' is provided,
            it selects the size of the biggest major semiaxis in the catalogue.
        function_to_reduce_data : str
            function to reduce the dimensions.
        truth_cat : str
            route to the catalogue file.
        data_file_ps : str
            route to HI emission file. Unused in the current version of the code
        min_major_length : float
            minimum major semiaxis to draw in the plot.
        min_minor_length : float
            minimum minor semiaxis to draw in the plot.
        flujo_min : float
            minimum inetgrated flux, in JyHz, of galaxies in order to be considered.

        Returns
        -------
        None.

        """
        self.data, self.header = read_data(data_file)
       
        self.cube_number = cube_dimensions
        
        # Reduce dimensions
        if function_to_reduce_data == 'mean_frec':
            self.reduced_data = mean_frec(self.data)
        self.length = len(self.reduced_data)
        
        # Read catalogue and drop objects whose flux integral is less than a given parameter
        self.truth_cat_original = pd.read_csv(truth_cat, sep=' ', index_col=0)
        
        
        self.overlapping = overlapping
        self.truth_cat_original.index = [i for i in range(len(self.truth_cat_original))]
        
        self.truth_cat = self.truth_cat_original.copy()
        
        
        self.truth_cat = self.truth_cat.loc[self.truth_cat['line_flux_integral'] >= flujo_min]

        # Transform coordinates in the catalogue to pixels
        ra = Angle(self.truth_cat['ra'], unit=u.degree) 
        dec = Angle(self.truth_cat['dec'], unit=u.degree)
        
        coords = SkyCoord(ra, dec)
        self.wcs = WCS(self.header, naxis=2)
        truth_pixels = self.wcs.world_to_pixel(coords)
        
        
        self.truth_cat['dec'] = truth_pixels[1]
        self.truth_cat['ra'] = truth_pixels[0]
        
        
        # Transform major_semiaxis in the catalogue to pixels. Assumes same pixel scale for both axis
        self.sdss_pixelscale = u.pixel_scale(proj_plane_pixel_scales(self.wcs)[0]*u.degree/u.pixel)
        self.truth_cat['hi_size'] = [(i * u.arcsec).to(u.pixel, self.sdss_pixelscale).value for i in self.truth_cat['hi_size']]
        
       
        # Conversion to radians
        self.truth_cat['i'] = self.truth_cat['i'] * np.pi/180
        self.truth_cat['pa'] = self.truth_cat['pa'] * np.pi/180 
        
        # Split the datacube into small cubes
        
        self.small_datacubes, self.truth_cat_small = split_into_small_boxes(self.reduced_data, 
                                                                             self.cube_number,
                                                                             self.truth_cat, 
                                                                             self.overlapping)
        
        ############### Normalizacion (not completed) ########################################
        maximo = np.max([np.max(i) for i in self.small_datacubes])
        self.small_datacubes = [i/maximo for i in self.small_datacubes]
        
        # layer = layers.Normalization()
        # for i in range(len(self.small_datacubes)):
        #     ind, cols = self.small_datacubes[i].index.values, self.small_datacubes[i].columns.values
        #     layer.adapt(self.small_datacubes[i])
        #     normalized_data = layer(self.small_datacubes[i])
        #     self.small_datacubes[i] = pd.DataFrame(normalized_data.numpy(), index=ind, columns=cols)
        # new_image = tf.image.per_image_standardization(image)
        #####################################################################################
        
        self.length_list_minicubes = len(self.small_datacubes)
        
        # Generate the ellipses for each datacube
        
        self.all_total_labels = [generate_galaxies(datacube, catalogue, self.length, 
                                                    min_major_length, min_minor_length)
                                  for datacube, catalogue in zip(self.small_datacubes[:], 
                                                                self.truth_cat_small[:])]
        

    def plot_datacube_label(self, number_of_datacube, new_figure=True):
        """
        Draw the label for a given datacube

        Parameters
        ----------
        number_of_datacube : int
            number of the datacube to plot.

        Returns
        -------
        None.

        """
        
        plt.figure()

        plt.imshow(self.all_total_labels[number_of_datacube], origin='lower') 
        plt.title('Datacube: {}' .format(number_of_datacube))
        plt.colorbar()
        # plt.show()
        
    def plot_flux_histogram(self, bins):
        """
        Plot a histogram with the sizes of the galaxies

        Parameters
        ----------
        bins : int
            number of bins.

        Returns
        -------
        None.

        """
        plt.figure()
        plt.hist(self.truth_cat['line_flux_integral'], log=True, bins=bins, ec='black')
        plt.xlabel('Flujo integrado/Jy Hz')
        plt.ylabel('Número de galaxias')
    
            
    def plot_small_reduced_datacube(self, number_of_datacube=0, lth=7e-6):
        """
        Plot a small datacube from the list

        Parameters
        ----------
        number_of_datacube : int, optional
            number of the desired datacube. The default is 0.
        lth : float, optional
            lth parameter of matplotlib log normalization. The default is 7e-6.

        Returns
        -------
        None.

        """
        
        plt.figure()
        # Parameters to set up-left square as the origin
        ra_min = self.small_datacubes[number_of_datacube].columns.values[0]
        dec_min = self.small_datacubes[number_of_datacube].index.values[0]

        plt.imshow(self.small_datacubes[number_of_datacube], cmap=cmap, norm=colors.SymLogNorm(linthresh=lth), origin='lower')
        
        plt.plot(self.truth_cat_small[number_of_datacube]['ra']- ra_min, self.truth_cat_small[number_of_datacube]['dec']- dec_min, 'rx')
        plt.title('Datacube:'+str(number_of_datacube))
        plt.colorbar()
        plt.show()

        
        
    def plot_full_reduced_datacube(self, coords='pix', lth=1e-5):
        """
        Draw the whole datacube with the catalogue over it

        Parameters
        ----------
        coords : str, optional
            type of coordinates for the axis, pix or equatorial. The default is 'pix'.
        lth : float, optional
            lth parameter of matplotlib log normalization. The default is 1e-5.

        Returns
        -------
        None.

        """
        fig = plt.figure()
        if coords == 'pix':
            fig.add_subplot(111)
        elif coords == 'equatorial':
            fig.add_subplot(111, projection=self.wcs)
            

        plt.imshow(self.reduced_data, cmap=cmap, norm=colors.SymLogNorm(linthresh=lth), origin='lower')
        plt.colorbar()
        
        plt.plot(self.truth_cat['ra'], self.truth_cat['dec'], 'rx')
        plt.savefig('../informe/imgs/resultados_comprobacion_splitting.png', pad_inches=0)
        plt.show()

    def __plot_val_tra_loss(self, model_history):
        """
        Plot the training/validation loss curve

        Parameters
        ----------
        model_history : tf.model
            output of the fit method.

        Returns
        -------
        None.

        """
        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']
        
        plt.figure()
        plt.plot(model_history.epoch, loss, 'r', label='Training loss')
        plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.ylim([0, 1])
        plt.legend()
        plt.show()
        

    def solve_NN(self, fraction_of_training, dim, batch_size, 
                 EPOCHS, filt, architecture='unet_2d', optimizer='adam', 
                 loss='sparse_categorical_crossentropy', metrics='accuracy', 
                 save_scheme=False, patience=10, verbose=1, dil=0):
        """
        Trains the neural network

        Parameters
        ----------
        fraction_of_training : float
            fraction of the whole dataseet to use in training the NN.
        dim : int
            dimensions of the images.
        batch_size : int
            size of the batches used in the training.
        EPOCHS : int
            total number of epochs.
        filt : list
            list with filters for each layer. The list should end at the bottom
            of the 'U'. The filters of the other half are obtained from these. 
        architecture : str, optional
            keyword of the desired architecture. List of the available keywords
            is provided in the keras-unet-collection documentation. The default is 'unet_2d'.
        optimizer : str or Optimizer, optional
            name of the optimizer. The default is 'adam'.
        loss : str, optional
            name of the loss function. The default is 'sparse_categorical_crossentropy'.
        metrics : str, optional
            name of the metrics. The default is 'accuracy'.
        save_scheme : bool, optional
            If true, it saves the scheme of the model in the imgs folder. The default is False.
        patience : int, optional
            patience parameter for the early stopping. The default is 10.
        verbose : int, optional
            verbose parameter for the early stopping. The default is 1.
        dil : float, optional
            dilatation parameter. It is only used in resunet_a_2d architecture. The default is 0.

        Returns
        -------
        None.

        """
        
        datac_num = int(fraction_of_training*len(self.small_datacubes))

        train_images = np.array([i.to_numpy() for j, i in enumerate(self.small_datacubes) if j <= datac_num])
        train_labels = np.array([i.to_numpy() for j, i in enumerate(self.all_total_labels) if j <= datac_num])

        test_images = np.array([i.to_numpy() for j, i in enumerate(self.small_datacubes) if j > datac_num])
        test_labels = np.array([i.to_numpy() for j, i in enumerate(self.all_total_labels) if j > datac_num])
        
        
        length = len(train_images[0][0])

        # self.unet = first_model((length, length, 1), n_classes=3, n_filters=n_filters)
        if architecture == 'att_unet_2d':
            if filt == 'auto':
                filt = [32, 64, 128, 256, 512, 1024]
            self.unet = segmentation_models.att_unet_2d((length, length, 1), 
                                                        filter_num=filt, n_labels=3) # 
        elif architecture == 'unet_2d':
            if filt == 'auto':
                filt = [32, 64, 128, 256, 512, 1024]
            self.unet = segmentation_models.unet_2d((length, length, 1), filter_num=filt, 
                                                    n_labels=3) # 
        elif architecture == 'vnet_2d':
            if filt == 'auto':
                filt = [32, 64, 128, 256]
            self.unet = segmentation_models.vnet_2d((length, length, 1), filter_num=filt,
                                                    n_labels=3) #No funciona 
        elif architecture == 'unet_plus_2d':
            if filt == 'auto':
                filt = [32, 64, 128, 256, 512]
            self.unet = segmentation_models.unet_plus_2d((length, length, 1), filter_num=filt,
                                                         n_labels=3, stack_num_down=2, stack_num_up=2,
                                                         activation='ReLU', output_activation='Softmax', 
                                                         batch_norm=False, pool='max', unpool=False, name='xnet')
 #Funciona pero sale mal 
        elif architecture == 'r2_unet_2d':
            if filt == 'auto':
                filt = [32, 64, 128, 256, 512, 1024]
            self.unet = segmentation_models.r2_unet_2d((length, length, 1), filter_num=filt,
                                                       n_labels=3) 
        elif architecture == 'resunet_a_2d':
            if filt == 'auto':
                filt = [32, 64, 128]
            if dil == 'auto':
                dil = [1, 3, 15]
            self.unet = segmentation_models.resunet_a_2d((length, length, 1), 
                                                         dilation_num=dil, filter_num=filt,
                                                         n_labels=3) # Me llena la memoria  
        elif architecture == 'u2net_2d':
            if filt == 'auto':
                filt = [ 64, 128, 256, 512]
            self.unet = segmentation_models.u2net_2d((length, length, 1), filter_num_down=filt, 
                                                     n_labels=3, output_activation='Softmax') # 
        elif architecture == 'unet_3plus_2d':
            if filt == 'auto':
                filt = [32, 64, 128, 256]
            self.unet = segmentation_models.unet_3plus_2d((length, length, 1), filter_num=filt, n_labels=3, output_activation='softmax')
        elif architecture == 'transunet_2d':
            if filt == 'auto':
                filt = [32, 64, 128, 256]
            self.unet = segmentation_models.transunet_2d((length, length, 1), filter_num=filt, n_labels=3)
        elif architecture == 'swin_unet_2d':
            if filt == 'auto':
                filt = [32, 64, 128, 256]
            self.unet = segmentation_models.swin_unet_2d((length, length, 1), filter_num_begin=32, n_labels=3, depth=4, stack_num_down=2, stack_num_up=2, 
                            patch_size=(2, 2), num_heads=[4, 8, 8, 8], window_size=[4, 2, 2, 2], num_mlp=512, 
                            output_activation='Softmax', shift_window=True, name='swin_unet')
            
        elif architecture == 'first_model':
            if filt == 'auto':
                filt = [32, 64, 128, 256]
            self.unet = first_model((length, length, 1), n_filters=16, n_classes=3)
        else:
            if filt == 'auto':
                filt = [32, 64, 128, 256]
            self.unet = architecture((length, length, 1), filter_num=filt, n_labels=3)
        
        
        self.unet.summary()
        
        if save_scheme:
            img_file = 'imgs/model_scheme.png'
            tf.keras.utils.plot_model(self.unet, to_file=img_file, show_shapes=True, show_layer_names=True)
        
        
        self.unet.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        
        earlystopper = EarlyStopping(patience=patience, verbose=verbose)
        
        model_history = self.unet.fit(train_images, train_labels, batch_size=batch_size,
                                      validation_data=(test_images, test_labels), epochs=EPOCHS, 
                                      callbacks=[earlystopper])
        
        self.__plot_val_tra_loss(model_history)
        
    def prediction(self, dim, img_num, img, real_mask, truth_cat_small, thr):
        """
        

        Parameters
        ----------
        dim : int
            size of the trained image in one direction.
        img_num : int
            number of the image (to plot the title).
        img : pd.DataFrame
            image to evaluate.
        real_mask : pd.DataFrame
            mask obtained from catalogue, to plot next to the predicted mask.
        truth_cat_small : pd.DataFrame
            catalogue of the small image.
        thr : float
            minimum value to consider pixels as part of galaxies instead of background.

        Returns
        -------
        new_catalogue : pdDataFrame
            predicted catalogue.

        """
                
        if dim == 'auto':
            dim = len(img)
            img_tf = tf.expand_dims(img, axis=0)
            real_mask = tf.expand_dims(real_mask.to_numpy(), axis=0)
                        
        else:
            # Not recommended. Better change dimensions when splitting datacube
            print('Resizing images. This is not a recommended operation as accuracy might be reduced')
            img = np.array([resize(img.to_numpy(), (dim, dim), anti_aliasing=True)])
            real_mask = np.abs(np.round(resize(real_mask.to_numpy(), (dim, dim), anti_aliasing=True)))
        
        predictions = self.unet.predict(img_tf)

        predictions = predictions[0, :, :, :]
        new_catalogue = draw_predictions(predictions, img, real_mask, truth_cat_small, thr, self.sdss_pixelscale, self.wcs, img_num)
        
        return new_catalogue
    def score(predicted_catalogue, truth_catalogue):
        """
        Calls the score software, from the developers of the challenge

        Parameters
        ----------
        predicted_catalogue : str
            route to predicted catalogue.
        truth_catalogue : str
            route to truth catalogue.

        Returns
        -------
        result : Sdc2Score object
            object which has the information of the score in its attributes.

        """
        scorer = Sdc2Scorer.from_txt(predicted_catalogue, truth_catalogue)
        result = scorer.run(detail=True)
        return result

                
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        