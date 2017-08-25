#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# @author: Patrick Kavanagh (DIAS)
#
"""

The purpose of this script is to take a set of MIRISim simulations (produced
using break_mirisim.py for example) and run the JWST calibration pipeline on
each level 1B file.

The goal here is to check if MIRISim provides all the necessary keywords for
the JWST pipeline to run without error across the MIRI observational setup range.

Will flag if a pipeline run fails and output the error. Will also plot the input
level 1B files produced by MIRISim and the output level 2B files.

Note that, for now, this only runs to level 2B as level 3 pipelines are unavailable.
These will eventually be included along with association creation and processing.

Build 7.1dev of the pipeline is required.

"""
from __future__ import absolute_import, division, print_function

import os, shutil, logging, glob, sys

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import numpy as np

from jwst import datamodels
from jwst.pipeline import SloperPipeline
from jwst.pipeline import Image2Pipeline
from jwst.pipeline import Spec2Pipeline

def pipeline_mirisim(input_dir):
    """

    """
    # set up logging
    log_file = os.path.join(os.path.abspath(input_dir),'pipeline_MIRISim.log')

    # check if the pipeline log file exists
    if os.path.isfile(log_file): os.remove(log_file)
    else: pass

    testing_logger = logging.getLogger(__name__)
    testing_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    testing_logger.addHandler(handler)

    # check if the simulation folder exists
    if os.path.isdir(input_dir):
        pass
    else:
        print("Simulation folder not found")
        sys.exit(0)

    # go through individual simulations and run through pipeline
    simulations = glob.glob(os.path.join(input_dir,'IMA*','20*'))
    simulations.extend(glob.glob(os.path.join(input_dir,'MRS*','20*')))
    simulations.extend(glob.glob(os.path.join(input_dir,'LRS*','20*')))
    #for simulation in simulations: print(simulation)

    # get the full path of the cwd
    cwd = os.path.abspath(os.getcwd())

    # set the output figure directory
    out_fig_dir = os.path.join(cwd,input_dir,'pipeline_plots')
    try: shutil.rmtree(out_fig_dir)
    except: pass
    os.mkdir(out_fig_dir)

    for simulation in simulations:

        os.chdir(os.path.join(simulation,'det_images'))
        level1b_file = glob.glob('*.fits')[0]

        # isolate the simulation name for logging
        sim_name = simulation.split(os.sep)[1]

        with datamodels.open(level1b_file) as level1b_dm:

            if level1b_dm.meta.exposure.type == 'MIR_IMAGE':
                try:
                    level2a_dm = SloperPipeline.call(level1b_dm, output_file='rate.fits')
                    level2b_dm = Image2Pipeline.call(level2a_dm, output_file='cal.fits')

                    # level2b_dm doesn't contain anything. report on GitHub
                    # for now read from fits file
                    level2b_dm = datamodels.open('cal_cal.fits')

                    # set up output plots
                    fig,axs = plt.subplots(1, 3)
                    fig.set_figwidth(15.0)
                    fig.set_figheight(5.0)
                    axs = axs.ravel()
                    # TODO improve colour scales for plots

                    # plot level 1b, 2a, 2b
                    axs[0].imshow(level1b_dm.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                    axs[0].annotate('level 1B', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')
                    axs[1].imshow(level2a_dm.data, cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                    axs[1].annotate('level 2A', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')
                    axs[2].imshow(level2b_dm.data, cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                    axs[2].annotate('level 2B', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                    # save the pipeline plot
                    out_fig_name = sim_name + '.pdf'
                    fig.savefig(os.path.join(out_fig_dir,out_fig_name), dpi=200)
                    del fig

                    # log pass
                    testing_logger.info('%s passed' % sim_name)

                except Exception as e:
                    testing_logger.warning('%s failed' % sim_name)
                    testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))

            elif level1b_dm.meta.exposure.type in ['MIR_MRS','MIR_LRS-FIXEDSLIT',
                                                    'MIR_LRS-SLITLESS']:

                # correct for the 'SLITLESSPRISM', 'SUBPRISM' conflict
                if level1b_dm.meta.exposure.type == 'MIR_LRS-SLITLESS':
                    level1b_dm.meta.subarray.name = 'SUBPRISM'

                try:
                    level2a_dm = SloperPipeline.call(level1b_dm, output_file='rate.fits')
                    level2b_dm = Spec2Pipeline.call(level2a_dm, output_file='cal.fits')

                    # level2b_dm doesn't contain anything. report on GitHub
                    # for now read from fits file
                    level2b_dm = datamodels.open('cal_cal.fits')

                    # set up output plots
                    fig,axs = plt.subplots(1, 3)
                    fig.set_figwidth(15.0)
                    fig.set_figheight(5.0)
                    axs = axs.ravel()
                    # TODO improve colour scales for plots

                    # plot level 1b, 2a, 2b
                    axs[0].imshow(level1b_dm.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                    axs[0].annotate('level 1B', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')
                    axs[1].imshow(level2a_dm.data, cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                    axs[1].annotate('level 2A', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')
                    axs[2].imshow(level2b_dm.data, cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                    axs[2].annotate('level 2B', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                    # save the pipeline plot
                    out_fig_name = sim_name + '.pdf'
                    fig.savefig(os.path.join(out_fig_dir,out_fig_name), dpi=200)
                    del fig

                    # log pass
                    testing_logger.info('%s passed' % sim_name)

                except Exception as e:
                    testing_logger.warning('%s failed' % sim_name)
                    testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))

        os.chdir(cwd)


if __name__ == "__main__":

    #TODO add proper command line parser
    pipeline_mirisim(input_dir=sys.argv[1])
