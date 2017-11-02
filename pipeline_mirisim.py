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

Build 7.1 of the pipeline is required.

"""
from __future__ import absolute_import, division, print_function

import os, shutil, logging, glob, sys
from subprocess import call

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.table import Table

import numpy as np

from jwst import datamodels
from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Image2Pipeline
from jwst.pipeline import Image3Pipeline
from jwst.pipeline import Spec2Pipeline
from jwst.pipeline import Spec3Pipeline
from jwst.pipeline import Tso3Pipeline


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

    # cycle through the simulations
    for simulation in simulations:

        os.chdir(os.path.join(simulation,'det_images'))
        level1b_files = glob.glob('*.fits')

        # isolate the simulation name for logging
        sim_name = simulation.split(os.sep)[1]

        # check which MIRI mode
        with datamodels.open(level1b_files[0]) as level1b_dm:
            miri_mode = level1b_dm.meta.exposure.type

        # IMAGER --------------------------------------------
        if miri_mode == 'MIR_IMAGE':
            # run level 1 and 2 imager pipelines
            for f in level1b_files:
                with datamodels.open(f) as level1b_dm:

                        try:
                            level2a_dm = Detector1Pipeline.call(level1b_dm, save_results=True)
                            Image2Pipeline.call(level2a_dm, save_results=True)

                            # log pass
                            testing_logger.info('%s levels 1 and 2 passed' % sim_name)
                            levels12_check = True

                        except Exception as e:
                            testing_logger.warning('%s failed' % sim_name)
                            testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))
                            levels12_check = False

            # run level 3 pipeline
            if len(level1b_files) > 1:
                try:
                    level2B_files = glob.glob(os.path.join('*_cal.fits'))
                    call(["asn_from_list", "-o", "IMA_asn.json"] + level2B_files + ["--product-name", "dither"])
                    dm_3_container = datamodels.ModelContainer("IMA_asn.json")
                    Image3Pipeline.call(dm_3_container, save_results=True,
                                        steps={'tweakreg': {'skip': True}, 'tweakreg_catalog': {'skip': True}})

                    # log pass
                    testing_logger.info('%s level 3 passed' % sim_name)
                    level3_check = True

                except Exception as e:
                    testing_logger.warning('%s failed' % sim_name)
                    testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))
                    level3_check = False


            if len(level1b_files) == 1 and levels12_check == True:
                level2A_file = glob.glob(os.path.join('*_rate.fits'))[0]
                level2B_file = glob.glob(os.path.join('*_cal.fits'))[0]

                # set up output plots
                fig, axs = plt.subplots(1, 3)
                fig.set_figwidth(15.0)
                fig.set_figheight(5.0)
                axs = axs.ravel()

                # plot level 1b, 2a, 2b
                with datamodels.open(level1b_files[0]) as level1b_dm:
                    with datamodels.open(level2A_file) as level2a_dm:
                        with datamodels.open(level2B_file) as level2b_dm:
                            axs[0].imshow(level1b_dm.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                            axs[0].annotate('level 1B', xy=(0.7, 0.95), xycoords='axes fraction', fontsize=10, fontweight='bold',
                                            color='w')
                            axs[1].imshow(level2a_dm.data, cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                            axs[1].annotate('level 2A', xy=(0.7, 0.95), xycoords='axes fraction', fontsize=10, fontweight='bold',
                                            color='w')
                            axs[2].imshow(level2b_dm.data, cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                            axs[2].annotate('level 2B', xy=(0.7, 0.95), xycoords='axes fraction', fontsize=10, fontweight='bold',
                                            color='w')

                # save the pipeline plot
                out_fig_name = sim_name + '.pdf'
                fig.savefig(os.path.join(out_fig_dir, out_fig_name), dpi=200)
                del fig

            elif len(level1b_files) > 1 and level3_check == True:
                driz_dm = datamodels.open('dither_i2d.fits')

                # set up output plots
                fig, axs = plt.subplots(1, 1)
                fig.set_figwidth(8.0)
                fig.set_figheight(8.0)

                # plot drizzled image
                axs.imshow(driz_dm.data, cmap='jet', interpolation='nearest', origin='lower',
                           norm=LogNorm(vmin=1, vmax=1000))
                axs.annotate('Drizzled image', xy=(0.0, 1.02), xycoords='axes fraction', fontsize=12, fontweight='bold',
                             color='k')
                axs.set_facecolor('black')

                # save the pipeline plot
                out_fig_name = sim_name + '.pdf'
                fig.savefig(os.path.join(out_fig_dir, out_fig_name), dpi=200)
                del fig

        # MRS --------------------------------------------
        elif miri_mode == 'MIR_MRS':

            # run level 1 and 2  pipelines
            for f in level1b_files:
                with datamodels.open(f) as level1b_dm:

                    try:
                        level2a_dm = Detector1Pipeline.call(level1b_dm, save_results=True)
                        Spec2Pipeline.call(level2a_dm, save_results=True, steps={'straylight':{'skip':True},
                                                                                  'extract_1d':{'save_results':True}})

                        # log pass
                        testing_logger.info('%s levels 1 and 2 passed' % sim_name)
                        levels12_check = True

                    except Exception as e:
                        testing_logger.warning('%s failed' % sim_name)
                        testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))
                        levels12_check = False

            # run level 3 pipeline
            if len(level1b_files) > 1:
                try:
                    level2B_files = glob.glob(os.path.join('*_cal.fits'))
                    call(["asn_from_list", "-o", "MRS_asn.json"] + level2B_files + ["--product-name",
                                                                                    "dither"])
                    dm_3_container = datamodels.ModelContainer("MRS_asn.json")
                    Spec3Pipeline.call(dm_3_container, save_results=True,
                                       steps={'outlier_detection': {'skip': True},
                                              'cube_build': {'save_results': True},
                                              'extract_1d': {'save_results': True}})

                    # log pass
                    testing_logger.info('%s level 3 passed' % sim_name)
                    level3_check = True

                except Exception as e:
                    testing_logger.warning('%s failed' % sim_name)
                    testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))
                    level3_check = False

            if len(level1b_files) == 1 and levels12_check == True:
                level2A_file = glob.glob('*_rate.fits')[0]
                level2B_file = glob.glob('*_cal.fits')[0]
                spec_file = glob.glob('*x1d.fits')[0]

                # set up output plots
                fig, axs = plt.subplots(2, 2)
                fig.set_figwidth(15.0)
                fig.set_figheight(15.0)
                axs = axs.ravel()

                with datamodels.open(level1b_files[0]) as level1b_dm:
                    with datamodels.open(level2A_file) as level2a_dm:
                        with datamodels.open(level2B_file) as level2b_dm:
                            with datamodels.open(spec_file) as spec_dm:

                                axs[0].imshow(level1b_dm.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                                axs[0].annotate('level 1B', xy=(0.7, 0.95), xycoords='axes fraction', fontsize=10, fontweight='bold',
                                                color='w')
                                axs[1].imshow(level2a_dm.data, cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                                axs[1].annotate('level 2A', xy=(0.7, 0.95), xycoords='axes fraction', fontsize=10, fontweight='bold',
                                                color='w')
                                axs[2].imshow(level2b_dm.data, cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                                axs[2].annotate('level 2B', xy=(0.7, 0.95), xycoords='axes fraction', fontsize=10, fontweight='bold',
                                                color='w')

                                # plot the spectrum
                                axs[3].plot(spec_dm.spec[0].spec_table['WAVELENGTH'], spec_dm.spec[0].spec_table['FLUX'], c='b',
                                            marker='.', markersize=3, linestyle='-', linewidth=2)
                                axs[3].set_ylabel(r'Flux ($\mu$Jy)')
                                axs[3].set_xlabel(r'Wavelength ($\mu$m)')
                                axs[3].set_xlim(4.0, 28.0)
                                # axs[0].set_ylim(0,6000)

                # save the pipeline plot
                out_fig_name = sim_name + '.pdf'
                fig.savefig(os.path.join(out_fig_dir, out_fig_name), dpi=200)
                del fig

            elif len(level1b_files) > 1 and level3_check == True:

                # cube
                cube_file = glob.glob("dither*s3d.fits")[0]
                cube_dm = datamodels.open(cube_file)

                # spec
                spec_file = glob.glob( "dither*1d.fits")[0]
                dm = datamodels.open(spec_file)

                fig, axs = plt.subplots(1,2, figsize=(12, 8))

                axs[0].imshow(np.sum(cube_dm.data, axis=0), cmap='jet', interpolation='nearest',
                              origin='lower', norm=LogNorm(vmin=100, vmax=5e5))
                axs[0].annotate('Collapsed cube', xy=(0.0, 1.02), xycoords='axes fraction', fontsize=12,
                                fontweight='bold', color='k')
                axs[0].set_facecolor('black')


                # plot the spectrum
                axs[1].plot(dm.spec[0].spec_table['WAVELENGTH'], dm.spec[0].spec_table['FLUX'], c='b', marker='.',
                            markersize=3, linestyle='-', linewidth=2)
                axs[1].set_ylabel(r'Flux ($\mu$Jy)')
                axs[1].set_xlabel(r'Wavelength ($\mu$m)')
                axs[1].set_xlim(4.0, 28.0)
                # axs[0].set_ylim(0,6000)
                axs[1].annotate('Spectrum)', xy=(0.0, 1.02), xycoords='axes fraction', fontsize=12, fontweight='bold',
                                color='k')

                # save the pipeline plot
                out_fig_name = sim_name + '.pdf'
                fig.savefig(os.path.join(out_fig_dir, out_fig_name), dpi=200)
                del fig


        # LRS-FIXEDSLIT --------------------------------------------
        elif miri_mode == 'MIR_LRS-FIXEDSLIT':

            # run level 1 and 2 pipelines
            for f in level1b_files:
                with datamodels.open(f) as level1b_dm:

                    try:
                        level2a_dm = Detector1Pipeline.call(level1b_dm, save_results=True)
                        Spec2Pipeline.call(level2a_dm, save_results=True, steps={'extract_1d': {'save_results': True}})

                        # log pass
                        testing_logger.info('%s levels 1 and 2 passed' % sim_name)
                        levels12_check = True

                    except Exception as e:
                        testing_logger.warning('%s failed' % sim_name)
                        testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))
                        levels12_check = False

            if len(level1b_files) == 1 and levels12_check == True:
                level2A_file = glob.glob('*_rate.fits')[0]
                level2B_file = glob.glob('*_cal.fits')[0]
                spec_file = glob.glob('*x1d.fits')[0]

                # set up output plots
                fig, axs = plt.subplots(2, 2)
                fig.set_figwidth(15.0)
                fig.set_figheight(15.0)
                axs = axs.ravel()

                with datamodels.open(level1b_files[0]) as level1b_dm:
                    with datamodels.open(level2A_file) as level2a_dm:
                        with datamodels.open(level2B_file) as level2b_dm:
                            with datamodels.open(spec_file) as spec_dm:

                                axs[0].imshow(level1b_dm.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(),
                                              origin='lower')
                                axs[0].annotate('level 1B', xy=(0.7, 0.95), xycoords='axes fraction', fontsize=10,
                                                fontweight='bold', color='w')
                                axs[1].imshow(level2a_dm.data, cmap='jet', interpolation='nearest', norm=LogNorm(),
                                              origin='lower')
                                axs[1].annotate('level 2A', xy=(0.7, 0.95), xycoords='axes fraction', fontsize=10,
                                                fontweight='bold',
                                                color='w')
                                axs[2].imshow(level2b_dm.data, cmap='jet', interpolation='nearest', norm=LogNorm(),
                                              origin='lower')
                                axs[2].annotate('level 2B', xy=(0.7, 0.95), xycoords='axes fraction', fontsize=10,
                                                fontweight='bold',color='w')

                                # plot the spectrum
                                axs[3].plot(spec_dm.spec[0].spec_table['WAVELENGTH'][1:-1], spec_dm.spec[0].spec_table['FLUX'][1:-1], c='b', marker='.',
                                         markersize=3, linestyle='-', linewidth=2)
                                axs[3].set_ylabel(r'Flux ($\mu$Jy)')
                                axs[3].set_xlabel(r'Wavelength ($\mu$m)')
                                axs[3].set_xlim(3.0, 15.0)
                                # axs[0].set_ylim(0,6000)
                                axs[3].annotate('Spectrum)', xy=(0.0, 1.02), xycoords='axes fraction', fontsize=12, fontweight='bold',
                                             color='k')
                                axs[3].set_facecolor('white')

                # save the pipeline plot
                out_fig_name = sim_name + '.pdf'
                fig.savefig(os.path.join(out_fig_dir, out_fig_name), dpi=200)
                del fig


        # LRS-SLITLESS --------------------------------------------
        elif miri_mode == 'MIR_LRS-SLITLESS':

                level1b_dm = datamodels.open(level1b_files[0])

                # correct for the 'SLITLESSPRISM', 'SUBPRISM' conflict
                #if level1b_dm.meta.exposure.type == 'MIR_LRS-SLITLESS':
                #    level1b_dm.meta.subarray.name = 'SUBPRISM'

                try:
                    Detector1Pipeline.call(level1b_dm)
                    level2a_ints = glob.glob('*rateints.fits')[0]
                    Spec2Pipeline.call(level2a_ints, save_results=True, steps={'extract_1d': {'save_results': True}})
                    level2b_ints = glob.glob('*calints.fits')[0]

                    #Tso3Pipeline.call(level2b_ints)

                    # log pass
                    testing_logger.info('%s passed' % sim_name)
                    levels12_check = True

                except Exception as e:
                    testing_logger.warning('%s failed' % sim_name)
                    testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))
                    levels12_check = False

                try:
                    level2B_files = glob.glob('*_calints.fits')
                    call(["asn_from_list", "-o", "LRS-SLITLESS_asn.json"] + level2B_files + ["--product-name",
                                                                                    "exposures"])
                    Tso3Pipeline.call('LRS-SLITLESS_asn.json')  # , steps={'white_light':{'skip':True}})

                    # log pass
                    testing_logger.info('%s level 3 passed' % sim_name)
                    level3_check = True

                except Exception as e:
                    testing_logger.warning('%s failed' % sim_name)
                    testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))
                    level3_check = False

                if len(level1b_files) == 1 and levels12_check == True:

                    spec_file = glob.glob('*x1dints.fits')[0]

                    # set up output plots
                    fig, axs = plt.subplots(3, 3, sharey=True, sharex=True)
                    fig.set_figwidth(15.0)
                    fig.set_figheight(15.0)
                    axs = axs.ravel()

                    with datamodels.open(spec_file) as spec_dm:

                        for n in range(9):

                            # plot the spectrum
                            axs[n].plot(spec_dm.spec[n].spec_table['WAVELENGTH'][1:-1], spec_dm.spec[n].spec_table['FLUX'][1:-1],
                                        c='b', marker='.', markersize=0, linestyle='-', linewidth=2)

                            if n in [0, 3, 6]: axs[n].set_ylabel(r'Flux ($\mu$Jy)')
                            if n in [6, 7, 8]: axs[n].set_xlabel(r'Wavelength ($\mu$m)')

                    # save the pipeline plot
                    out_fig_name = sim_name + '.pdf'
                    plt.tight_layout(pad=0.0)
                    fig.savefig(os.path.join(out_fig_dir, out_fig_name), dpi=200)
                    del fig

                    if level3_check == True:

                        my_lightcurve = 'tso_whtlt.ecsv'
                        lightcurve_data = Table.read(my_lightcurve, format='ascii.ecsv')

                        fig, axs = plt.subplots(1, 1, figsize=(10, 8))

                        # plot input and output ramps of the first integration
                        axs.plot(lightcurve_data[0][:], lightcurve_data[1][:], c='b', marker='o', markersize=3,
                                 linestyle='-', linewidth=2, label='white light curve')
                        axs.set_title('White light curve', fontsize=15)
                        axs.set_ylabel('white light flux', fontsize=15)
                        axs.set_xlabel('MJD', fontsize=15)

                        plt.tight_layout(h_pad=0)

                        # save the pipeline plot
                        out_fig_name = sim_name + '_whitelight.pdf'
                        plt.tight_layout(pad=0.0)
                        fig.savefig(os.path.join(out_fig_dir, out_fig_name), dpi=200)

                        del fig

        os.chdir(cwd)


if __name__ == "__main__":

    #TODO add proper command line parser
    pipeline_mirisim(input_dir=sys.argv[1])
