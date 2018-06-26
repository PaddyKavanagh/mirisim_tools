#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# @author: Patrick Kavanagh (DIAS)
#
"""
Compare the MRS output of two versions of MIRISim

Should use the latest MIRISim/pipeline environment to run this script
"""
import os, shutil, logging, glob, sys
from subprocess import call

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.table import Table

import numpy as np

from jwst import datamodels
from miri.datamodels import miri_illumination_model

class LoadMIRISims:
    """
    Load simulation files.

    Parameters:
    -----------
        path:   path to the simulation directory. Should be top level MIRISim output (i.e., 2017....)

    Examples:
    ---------
        >>> my_sims = LoadMIRISims('/home/me/mirisim/20170603_071718_mirisim')
        >>> my_sims.load()

    """

    def __init__(self, path):
        assert os.path.isdir(path)
        self.path = path

        self.det_images = None
        self.num_det_images = None
        self.illum_models = None
        self.num_illum_models = None
        self.skycubes = None
        self.num_skycubes = None

    def gather_det_images(self, path):
        """
        Load the det_images to list
        """
        self.det_images = sorted(glob.glob(os.path.join(path, 'det_images', 'det_image*.fits')))
        self.num_det_images = len(self.det_images)

        return self.det_images

    def gather_illum_models(self, path):
        """
        load the det_images to list
        """
        self.illum_models = sorted(glob.glob(os.path.join(path, 'illum_models', 'illum_model*.fits')))
        self.num_illum_models = len(self.illum_models)

        return self.illum_models

    def gather_skycubes(self, path):
        """
        load the det_images to list
        """
        self.skycubes = sorted(glob.glob(os.path.join(path, 'skycubes', 'skycube*.fits')))
        self.num_skycubes = len(self.skycubes)

        return self.skycubes

    def load(self):
        """
        Load everything into the object.
        """
        self.gather_det_images(self.path)
        self.gather_illum_models(self.path)

        # skycubes only generated for MRS sims
        with datamodels.open(self.det_images[0]) as dm:
            if 'IFU' in dm.meta.instrument.detector:
                self.gather_skycubes(self.path)


class MIRISimsRegression:
    """
    take to LoadMIRISims instances and compare for regression
    """
    def __init__(self, sim_old_path, sim_new_path, sim_name, out_fig_dir):
        self.sim_old_path = sim_old_path
        self.sim_new_path = sim_new_path
        self.sim_name = sim_name

        # construct the full path and load the sim files
        self.sim_old_full_path = glob.glob(os.path.join(self.sim_old_path, 'mirisim_tools', 'mirisim_functional_tests',
                                                        self.sim_name, '20*'))[0]
        self.sim_new_full_path = glob.glob(os.path.join(self.sim_new_path, 'mirisim_tools', 'mirisim_functional_tests',
                                                        self.sim_name, '20*'))[0]

        print('Testing: \n{}\n{}'.format(self.sim_old_full_path, self.sim_new_full_path))

        try:
            self.my_old_sim_files = LoadMIRISims(self.sim_old_full_path)
            self.my_old_sim_files.load()
            self.my_new_sim_files = LoadMIRISims(self.sim_new_full_path)
            self.my_new_sim_files.load()
        except:
            pass

        self.det_images_reg = None
        self.illum_models_reg = None
        self.skycubes_ratio_frames = None
        self.skycubes_ratio_collapsed = None
        self.out_fig_dir = out_fig_dir

    def _plot_image(self, old, new, ratio, name):
        """
        plot data old, new and ratio
        """
        # set up the plot
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))

        # set up limits of the colour scale for plotting
        if 'det_images' in name:
            vmin = 9.e3
        else:
            vmin = 1.

        # plot
        axs[0].imshow(old, cmap='jet', interpolation='nearest', origin='lower', norm=LogNorm(vmin=vmin))
        axs[0].set_facecolor('black')
        axs[1].imshow(new, cmap='jet', interpolation='nearest', origin='lower', norm=LogNorm(vmin=vmin))
        axs[1].set_facecolor('black')
        im = axs[2].imshow(ratio, cmap='jet', interpolation='nearest', origin='lower', vmin=0.99, vmax=1.01)
        axs[2].set_facecolor('black')
        fig.colorbar(im, ax=axs[2])

        # save output to file
        plt.tight_layout()
        out_fig_name = name + '.pdf'
        fig.savefig(os.path.join(self.out_fig_dir, out_fig_name), dpi=200)#
        plt.clf()

    def compare_skycubes(self):
        """
        compare skycubes
        """

        for n, sc in enumerate(self.my_old_sim_files.skycubes):

            old_sc_hdu = fits.open(sc)
            new_sc_hdu = fits.open(self.my_new_sim_files.skycubes[n])

            skycube_old_collpased = np.sum(old_sc_hdu[0].data, axis=0)
            skycube_new_collpased = np.sum(new_sc_hdu[0].data, axis=0)

            # divide for plotting
            self.skycubes_ratio_collapsed = skycube_old_collpased / skycube_new_collpased

            # make a plot
            plot_name = self.sim_name + '_skycube' + str(n)
            self._plot_image(skycube_old_collpased, skycube_new_collpased, self.skycubes_ratio_collapsed, plot_name)

            try:
                np.testing.assert_array_almost_equal(skycube_old_collpased, skycube_new_collpased, 8)
                # log pass
                testing_logger.info('{} skycubes passed'.format(plot_name))

            except Exception as e:
                testing_logger.warning('{} skycubes FAILED!!!'.format(plot_name))
                testing_logger.warning('  {}: {}'.format(e.__class__.__name__, str(e)))

    def compare_illum(self):
        """
        compare illumination models
        """

        for n, im in enumerate(self.my_old_sim_files.illum_models):

            old_illum_dm = miri_illumination_model.MiriIlluminationModel(im)
            new_illum_dm = miri_illumination_model.MiriIlluminationModel(self.my_new_sim_files.illum_models[n])

            # divide for plotting
            self.illum_ratio = old_illum_dm.intensity / new_illum_dm.intensity

            # make a plot
            plot_name = self.sim_name + '_illum' + str(n)
            self._plot_image(old_illum_dm.intensity, new_illum_dm.intensity, self.illum_ratio, plot_name)

            try:
                np.testing.assert_array_almost_equal(old_illum_dm.intensity, new_illum_dm.intensity, 8)
                # log pass
                testing_logger.info('{} illum models passed'.format(plot_name))

            except Exception as e:
                testing_logger.warning('{} illum models FAILED!!!'.format(plot_name))
                testing_logger.warning('  {}: {}'.format(e.__class__.__name__, str(e)))


    def compare_det_images(self):
        """
        compare det_images
        """

        for n, di in enumerate(self.my_old_sim_files.det_images):

            old_di_dm = datamodels.MIRIRampModel(di)
            new_di_dm = datamodels.MIRIRampModel(self.my_new_sim_files.det_images[n])

            #det_image_old_collpased = np.sum(old_di_dm.data[0], axis=0)
            #det_image_new_collpased = np.sum(new_di_dm.data[0], axis=0)

            # divide for plotting
            self.det_images_ratio_collapsed = old_di_dm.data[0][-1] / new_di_dm.data[0][-1]

            # make a plot
            plot_name = self.sim_name + '_det_images' + str(n)
            self._plot_image(old_di_dm.data[0][-1], new_di_dm.data[0][-1], self.det_images_ratio_collapsed, plot_name)

            try:
                np.testing.assert_array_almost_equal(old_di_dm.data[0][-1], new_di_dm.data[0][-1], 8)
                # log pass
                testing_logger.info('{} det_images passed'.format(plot_name))

            except Exception as e:
                testing_logger.warning('{} det_images FAILED!!!'.format(plot_name))
                testing_logger.warning('  {}: {}'.format(e.__class__.__name__, str(e)))


    def run(self):
        """
        run the tests
        """
        #self.compare_skycubes()
        #self.compare_illum()
        self.compare_det_images()


if __name__ == "__main__":

    #TODO add proper command line parser
    mirisim_old_version_sims = sys.argv[1]
    mirisim_new_version_sims = sys.argv[2]

    # set the output figure directory
    out_fig_dir = 'MIRISim_regression_plots'
    try:
        shutil.rmtree(out_fig_dir)
    except:
        pass
    os.mkdir(out_fig_dir)

    # set up logging
    log_file = 'MIRISim_regression.log'

    # check if the pipeline log file exists
    if os.path.isfile(log_file):
        os.remove(log_file)
    else:
        pass

    testing_logger = logging.getLogger(__name__)
    testing_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    testing_logger.addHandler(handler)


    # get the names of the sims
    sim_names = glob.glob(os.path.join(mirisim_old_version_sims, 'mirisim_tools',
                                       'mirisim_functional_tests', 'MRS*'))

    # do the regression checks
    for s in sim_names:
        sim_name = os.path.split(s)[-1]
        my_regression_checks = MIRISimsRegression(mirisim_old_version_sims, mirisim_new_version_sims, sim_name,
                                                  out_fig_dir=out_fig_dir)
        my_regression_checks.run()
