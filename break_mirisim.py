#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# @author: Patrick Kavanagh (DIAS)
#
"""

The purpose of this script is to run MIRISim simulations for covering the
entire input parameter space to try to break MIRISim.

Can also specify a subset of parameters to test (e.g., specific instrumemt,
vary filters only, etc.)

Flags failed simulations and produces plots of det_images, illum_models, and
skycubes (for MRS), for all successful simulations.

"""
from __future__ import absolute_import, division, print_function

import os, shutil, logging, glob

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits
import numpy as np

from mirisim.config_parser import *
from mirisim import MiriSimulation
from mirisim.skysim import *
from miri.datamodels import miri_illumination_model
from jwst import datamodels

def make_ima_sim_config(mode='FULL', dither=False, ndither=2, im_filter='F1130W',
                        readmode= 'FAST', exposures=1, integrations=5, groups=10):

    cfg_path = 'IMA_' + mode

    sim_config = SimConfig.makeSim(name="IMA Simulation",
                                   rel_obsdate=0.0,
                                   scene="scene.ini",
                                   POP='IMA',
                                   ConfigPath=cfg_path,
                                   Dither=dither,
                                   StartInd=1,
                                   NDither=ndither,
                                   DitherPat="ima_recommended_dither.dat",
                                   filter=im_filter,
                                   readDetect= mode,
                                   ima_mode= readmode,
                                   ima_exposures=exposures,
                                   ima_integrations=integrations,
                                   ima_frames=groups,
                                   disperser= 'SHORT',
                                   detector= 'BOTH',
                                   mrs_mode= 'SLOW',
                                   mrs_exposures=1,
                                   mrs_integrations=1,
                                   mrs_frames=20)

    return sim_config


def make_mrs_sim_config(mode='1SHORT', dither=False, ndither=2, grating='SHORT',
                        readmode= 'FAST',detector='SW', exposures=1, integrations=5,
                        groups=10):

    cfg_path = 'MRS_' + mode

    sim_config = SimConfig.makeSim(name="MRS Simulation",
                                   rel_obsdate=0.0,
                                   scene="scene.ini",
                                   POP='MRS',
                                   ConfigPath=cfg_path,
                                   Dither=dither,
                                   StartInd=1,
                                   NDither=ndither,
                                   DitherPat="mrs_recommended_dither.dat",
                                   filter='F1130W',
                                   readDetect= 'FULL',
                                   ima_mode= 'FAST',
                                   ima_exposures=1,
                                   ima_integrations=1,
                                   ima_frames=20,
                                   disperser= grating,
                                   detector= detector,
                                   mrs_mode= readmode,
                                   mrs_exposures=exposures,
                                   mrs_integrations=integrations,
                                   mrs_frames=groups)

    return sim_config



def make_scene_config(sky='simple', instrument='IMA', src_type='point'):
    """
    Make a scene config object. Only makes specific types of scenes.

    sky = 'grid':
    Creates a 5x5 (imager) or 3x3 (MRS) grid of point sources, centred on 0,0
    and an optional galaxy at the centre. The sources are spaced at 10" for
    the imager and 1" for the MRS

    sky = 'simple':
    Creates a scene with a single point source at 0,0

    Parameters:
    sky             --  type of sky to create. Options are 'simple' and 'grid'

    instrument      --  select the instrument being simulated. Sets what the
                        fluxes of the sources will be, how they are spaced
                        and the size of the grid of sources for the 'grid' sky

    src_type        --  sets the type of sources to simulate. Options are 'point'
                        and 'galaxy'

    Returns:
    cfg_obj         --  scene config object section
    """
    if instrument == 'IMA':
        if sky == 'simple':
            if src_type == 'point':
                #build scene
                background = Background(level='low', gradient=5., pa=15.0, centreFOV=(0., 0.))
                SED1 = BBSed(Temp=300., wref=10., flux=1.e4)
                Point1 = Point(Cen=(0.,0.), vel=0.0)
                Point1.set_SED(SED1)
                targets = [Point1]

                # create config object
                scene_config = SceneConfig.makeScene(loglevel=0, background=background, targets=targets)

            elif src_type == 'galaxy':
                #build scene
                background = Background(level='low', gradient=5., pa=15.0, centreFOV=(0., 0.))
                SED1 = BBSed(Temp=300., wref=10., flux=1.e4)
                Galaxy1 = Galaxy(Cen=(0.,0.), vel=0.0)
                Galaxy1.set_SED(SED1)
                targets = [Galaxy1]

                # create config object
                scene_config = SceneConfig.makeScene(loglevel=0, background=background, targets=targets)

        elif sky == 'grid':
                #build scene
                background = Background(level='low', gradient=5., pa=15.0, centreFOV=(0., 0.))
                SED_bright = BBSed(Temp=300., wref=10., flux=5.e4)
                SED_faint = BBSed(Temp=300., wref=10., flux=1.e4)

                Point1 = Point(Cen=(-10.,-10.), vel=0.0)
                Point1.set_SED(SED_faint)
                Point2 = Point(Cen=(0.,-10.), vel=0.0)
                Point2.set_SED(SED_faint)
                Point3 = Point(Cen=(10.,-10.), vel=0.0)
                Point3.set_SED(SED_faint)
                Point4 = Point(Cen=(-10.,0.), vel=0.0)
                Point4.set_SED(SED_faint)
                Point5 = Point(Cen=(0.,0.), vel=0.0)
                Point5.set_SED(SED_bright)
                Point6 = Point(Cen=(10.,0.), vel=0.0)
                Point6.set_SED(SED_faint)
                Point7 = Point(Cen=(-10.,10.), vel=0.0)
                Point7.set_SED(SED_faint)
                Point8 = Point(Cen=(0.,10.), vel=0.0)
                Point8.set_SED(SED_faint)
                Point9 = Point(Cen=(10.,10.), vel=0.0)
                Point9.set_SED(SED_faint)
                Point10 = Point(Cen=(-20.,-20.), vel=0.0)
                Point10.set_SED(SED_bright)
                Point11 = Point(Cen=(-10.,-20.), vel=0.0)
                Point11.set_SED(SED_bright)
                Point12 = Point(Cen=(0.,-20.), vel=0.0)
                Point12.set_SED(SED_bright)
                Point13 = Point(Cen=(10.,-20.), vel=0.0)
                Point13.set_SED(SED_bright)
                Point14 = Point(Cen=(20.,-20.), vel=0.0)
                Point14.set_SED(SED_bright)
                Point15 = Point(Cen=(-20.,20.), vel=0.0)
                Point15.set_SED(SED_bright)
                Point16 = Point(Cen=(-10.,20.), vel=0.0)
                Point16.set_SED(SED_bright)
                Point17 = Point(Cen=(0.,20.,), vel=0.0)
                Point17.set_SED(SED_bright)
                Point18 = Point(Cen=(10.,20.), vel=0.0)
                Point18.set_SED(SED_bright)
                Point19 = Point(Cen=(20.,20.), vel=0.0)
                Point19.set_SED(SED_bright)
                Point20 = Point(Cen=(-20.,-10.), vel=0.0)
                Point20.set_SED(SED_bright)
                Point21 = Point(Cen=(-20.,0.), vel=0.0)
                Point21.set_SED(SED_bright)
                Point22 = Point(Cen=(-20.,10.), vel=0.0)
                Point22.set_SED(SED_bright)
                Point23 = Point(Cen=(20.,-10.), vel=0.0)
                Point23.set_SED(SED_bright)
                Point24 = Point(Cen=(20.,0.), vel=0.0)
                Point24.set_SED(SED_bright)
                Point25 = Point(Cen=(20.,10.), vel=0.0)
                Point25.set_SED(SED_bright)
                targets = [Point1,Point2,Point3,Point4,Point5,Point6,Point7,Point8,
                            Point9,Point10,Point11,Point12,Point13,Point14,Point15,
                            Point16,Point17,Point18,Point19,Point20,Point21,Point22,
                            Point23,Point24,Point25]

                # create config object
                scene_config = SceneConfig.makeScene(loglevel=0, background=background, targets=targets)

    if instrument == 'MRS':
        if sky == 'simple':
            if src_type == 'point':
                #build scene
                background = Background(level='low', gradient=5., pa=15.0, centreFOV=(0., 0.))
                SED1 = BBSed(Temp=300., wref=10., flux=1.e5)
                Point1 = Point(Cen=(0.,0.), vel=0.0)
                Point1.set_SED(SED1)
                targets = [Point1]

                # create config object
                scene_config = SceneConfig.makeScene(loglevel=0, background=background, targets=targets)

            elif src_type == 'galaxy':
                #build scene
                background = Background(level='low', gradient=5., pa=15.0, centreFOV=(0., 0.))
                SED1 = BBSed(Temp=300., wref=10., flux=1.e5)
                Galaxy1 = Galaxy(Cen=(0.,0.), vel=0.0)
                Galaxy1.set_SED(SED1)
                targets = [Galaxy1]

                # create config object
                scene_config = SceneConfig.makeScene(loglevel=0, background=background, targets=targets)

        elif sky == 'grid':
                #build scene
                background = Background(level='low', gradient=5., pa=15.0, centreFOV=(0., 0.))
                SED_bright = BBSed(Temp=300., wref=10., flux=5.e5)
                SED_faint = BBSed(Temp=300., wref=10., flux=1.e5)

                Point1 = Point(Cen=(-1.,-1.), vel=0.0)
                Point1.set_SED(SED_faint)
                Point2 = Point(Cen=(0.,-1.), vel=0.0)
                Point2.set_SED(SED_faint)
                Point3 = Point(Cen=(1.,-1.), vel=0.0)
                Point3.set_SED(SED_faint)
                Point4 = Point(Cen=(-1.,0.), vel=0.0)
                Point5.set_SED(SED_faint)
                Point5 = Point(Cen=(0.,0.), vel=0.0)
                Point5.set_SED(SED_bright)
                Point6 = Point(Cen=(1.,0.), vel=0.0)
                Point6.set_SED(SED_faint)
                Point7 = Point(Cen=(-1.,1.), vel=0.0)
                Point7.set_SED(SED_faint)
                Point8 = Point(Cen=(0.,1.), vel=0.0)
                Point8.set_SED(SED_faint)
                Point9 = Point(Cen=(1.,1.), vel=0.0)
                Point9.set_SED(SED_faint)

                targets = [Point1,Point2,Point3,Point4,Point5,Point6,Point7,Point8,Point9]

                # create config object
                scene_config = SceneConfig.makeScene(loglevel=0, background=background, targets=targets)

    return scene_config


def get_output_product(product, channel=None, dither=False):
    """
    Function to determine the path of a specified MIRISim output product

    Parameters:
        product     --  name of the product required. Options are 'illum',
                        'det_image' and 'skycube'

        channel     --  if fetching MRS skycube, detemines which channel to get.
                        Opions are 1 and 2 for low and high channel, respectively

        dither      --  if True, returns the files associated with the second
                        dither position in the output product directory

    Returns:
        product_path    --  path to the output product

    ----------------------------------------------------------------------
    """
    # determine name of simulation folder
    sim_dir = glob.glob('2*')[0]

    # illumination model
    if product == 'illum' and dither == False:
        product_dir = os.path.join(sim_dir,'illum_models')
        product_path = glob.glob(os.path.join(product_dir,'ill*'))
        product_path = sorted(product_path)[0]
    elif product == 'illum' and dither == True:
        product_dir = os.path.join(sim_dir,'illum_models')
        product_path = glob.glob(os.path.join(product_dir,'ill*'))
        product_path = sorted(product_path)[1]

    # det_image
    if product == 'det_image' and dither == False:
        product_dir = os.path.join(sim_dir,'det_images')
        product_path = glob.glob(os.path.join(product_dir,'det*'))
        product_path = sorted(product_path)[0]
    elif product == 'det_image' and dither == True:
        product_dir = os.path.join(sim_dir,'det_images')
        product_path = glob.glob(os.path.join(product_dir,'det*'))
        product_path = sorted(product_path)[1]

    # skycube
    if product == 'skycube' and dither == False:
        product_dir = os.path.join(sim_dir,'skycubes')
        product_path = glob.glob(os.path.join(product_dir,'sky*'))

        if channel == 1: product_path = sorted(product_path)[0]
        elif channel == 2: product_path = sorted(product_path)[1]

    elif product == 'skycube' and dither == True:
        product_dir = os.path.join(sim_dir,'skycubes')
        product_path = glob.glob(os.path.join(product_dir,'sky*'))

        if channel == 1: product_path = sorted(product_path)[2]
        elif channel == 2: product_path = sorted(product_path)[3]

    return product_path


def break_mirisim(imager=False, ima_filters=False, ima_subarrays=False, ima_readmodes=False,
                 mrs=False, mrs_paths=False, mrs_gratings=False, mrs_detectors=False,
                 mrs_readmodes=False, lrs=False, lrs_slits=False, lrs_readmodes=False,
                 dither=False, scene='point'):
    """
    Run functional tests on MIRISim based on user supplied options.

    Output of simulations are saved under 'mirisim_functional_testing' in the
    working directory and named according to the simulation parameters.

    A testing log file is saved in test_MIRISim.log, which lists which simulations
    have passed or failed.

    Parameters:
        ----------------------------------------------------------------------
        imager          --  run imager simulations

        ima_filter      --  test all imager filters. If False, defaults to F1130W

        ima_subarrays   --  test all imager subarrays. If False, defaults to FULL

        ima_readmodes   --  test imager readmodes. If False, defaults to FAST

        ----------------------------------------------------------------------

        mrs             --  run MRS simulations

        mrs_paths       --  test all MRS paths. If False, defaults to 1SHORT

        mrs_gratings    --  test all MRS gratings. If False, defaults to SHORT

        mrs_detectors   --  test all MRS detectors. If False, defaults to SW

        mrs_readmodes   --  test MRS readmodes. If False, defaults to SLOW

        ----------------------------------------------------------------------

        dither              --  if True, performs two point dither for all
                                simulations

        include_galaxies    --  if True, includes a galaxy at 0,0 in scene,ini
                                files

        scene               --  specifies the type of scene. Options are
                                'point','galaxy', and 'grid'

    Returns:
        A folder named 'mirisim_functional_testing' containing simulation
        output and the testing log file test_MIRISim.log
    """
    # set cwd
    cwd = os.getcwd()

    # set the output directory
    out_dir = os.path.join(cwd,'mirisim_functional_tests')
    try: shutil.rmtree(out_dir)
    except: pass
    os.mkdir(out_dir)

    # set the output figure directory
    out_fig_dir = os.path.join(cwd,'mirisim_functional_tests','simulation_plots')
    try: shutil.rmtree(out_fig_dir)
    except: pass
    os.mkdir(out_fig_dir)

    # move to out_dir
    os.chdir(out_dir)

    # set up logging
    log_file = os.path.join(out_dir,'test_MIRISim.log')
    testing_logger = logging.getLogger(__name__)
    testing_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    testing_logger.addHandler(handler)

    # generate the default simultor config object
    simulator_cfg = SimulatorConfig.from_default()

    # imager simulations
    if imager == True:
        testing_logger.info('Starting imager simulations')

        if ima_subarrays == True:
            modes = ['FULL','BRIGHTSKY', 'SUB256', 'SUB128', 'SUB64']
        else: modes = ['FULL']  # set as default

        if ima_filters == True:
            im_filters = ['F560W', 'F770W', 'F1000W', 'F1130W', 'F1280W', 'F1500W',
                          'F1800W', 'F2100W', 'F2550W', 'F1065C', 'F1140C',
                          'F1550C', 'F2300C', 'F2550WR']
        else: im_filters = ['F1130W']  # set as default

        if ima_readmodes == True:
            read_modes = ['FAST', 'SLOW']
        else: read_modes = ['FAST']  # set as default

        # set observation parameters (hardcoded for now)
        ndither = 2
        exposures=1
        integrations=1

        if scene == 'point':
            scene_cfg = make_scene_config(sky='simple', instrument='IMA', src_type='point')
        elif scene == 'grid':
            scene_cfg = make_scene_config(sky='grid', instrument='IMA', src_type='point')


        for mode in modes:
            for im_filter in im_filters:
                for read_mode in read_modes:

                    # set the number of groups depending on the readout mode
                    if read_mode == 'FAST': groups=50
                    elif read_mode == 'SLOW': groups=10

                    sim_dir = 'IMA_' + mode + '_' + im_filter + '_' + read_mode + '_dithering-' + str(dither)
                    sim_fig = 'IMA_' + mode + '_' + im_filter + '_' + read_mode + '_dithering-' + str(dither) + '.pdf'
                    os.mkdir(sim_dir)
                    os.chdir(sim_dir)

                    sim_cfg = make_ima_sim_config(mode=mode, dither=dither, ndither=ndither,
                                                    im_filter=im_filter,readmode=read_mode,
                                                    exposures=exposures, integrations=integrations,
                                                    groups=groups)

                    print('Simulating %s' % sim_dir)
                    try:
                        mysim = MiriSimulation(sim_cfg, scene_cfg, simulator_cfg)
                        mysim.run()

                        # log pass
                        testing_logger.info('%s passed' % sim_dir)

                        if dither == False:
                            fig,axs = plt.subplots(1, 2)
                            fig.set_figwidth(12.0)
                            fig.set_figheight(6.0)
                        elif dither == True:
                            fig,axs = plt.subplots(2,2)
                            fig.set_figwidth(12.0)
                            fig.set_figheight(12.0)
                        #plt.tight_layout(pad=0.5)
                        axs = axs.ravel()
                        axs_index = -1

                        # plot output, illumination model and last frame of first integration (only one per exposure)
                        illum_file = get_output_product('illum')
                        illum_datamodel = miri_illumination_model.MiriIlluminationModel(illum_file)

                        axs_index += 1
                        axs[axs_index].imshow(illum_datamodel.intensity[0], cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                        axs[axs_index].annotate(sim_dir, xy=(0.0,1.02), xycoords='axes fraction', fontsize=14, fontweight='bold', color='k')
                        axs[axs_index].annotate('illum_model', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                        det_file = get_output_product('det_image')
                        det_datamodel = datamodels.open(det_file)

                        axs_index += 1
                        axs[axs_index].imshow(det_datamodel.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                        axs[axs_index].annotate('det_image', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                        if dither == True:
                            illum_file = get_output_product('illum', dither=True)
                            illum_datamodel = miri_illumination_model.MiriIlluminationModel(illum_file)

                            axs_index += 1
                            axs[axs_index].imshow(illum_datamodel.intensity[0], cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                            axs[axs_index].annotate('dither position 2', xy=(0.0,1.02), xycoords='axes fraction', fontsize=12, fontweight='bold', color='k')
                            axs[axs_index].annotate('illum_model', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                            det_file = get_output_product('det_image', dither=True)
                            det_datamodel = datamodels.open(det_file)

                            axs_index += 1
                            axs[axs_index].imshow(det_datamodel.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                            axs[axs_index].annotate('det_image', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                        fig.savefig(os.path.join(out_fig_dir,sim_fig), dpi=200)
                        del fig

                    except Exception as e:
                        testing_logger.warning('%s failed' % sim_dir)
                        testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))

                    os.chdir(out_dir)




    # MRS simulations
    if mrs == True:
        testing_logger.info('Starting MRS simulations')

        if mrs_paths == True:
            modes = ['1SHORT','2SHORT', '3SHORT', '4SHORT',
                     '1MEDIUM','2MEDIUM', '3MEDIUM', '4MEDIUM',
                     '1LONG','2LONG', '3LONG', '4LONG']
        else: modes = ['1SHORT']  # set as default

        if mrs_gratings == True:
            dispersers = ['SHORT', 'MEDIUM', 'LONG']
        else: dispersers = ['SHORT']  # set as default

        if mrs_detectors == True:
            detectors = ['SW', 'LW', 'BOTH']
        else: detectors = ['SW']  # set as default

        if mrs_readmodes == True:
            read_modes = ['FAST', 'SLOW']
        else: read_modes = ['FAST']  # set as default

        # set observation parameters (hardcoded for now)
        ndither = 2
        exposures=1
        integrations=1

        if scene == 'point':
            scene_cfg = make_scene_config(sky='simple', instrument='MRS', src_type='point')
        elif scene == 'grid':
            scene_cfg = make_scene_config(sky='grid', instrument='MRS', src_type='point')

        for mode in modes:
            for disperser in dispersers:
                for detector in detectors:
                    for read_mode in read_modes:

                        # set the number of groups depending on the readout mode
                        if read_mode == 'FAST': groups=50
                        elif read_mode == 'SLOW': groups=10

                        sim_dir = 'MRS_' + mode + '_' + disperser + '_' + detector + '_' + read_mode + '_dithering-' + str(dither)
                        sim_fig = 'MRS_' + mode + '_' + disperser + '_' + detector + '_' + read_mode + '_dithering-' + str(dither) + '.pdf'
                        os.mkdir(sim_dir)
                        os.chdir(sim_dir)

                        sim_cfg = make_mrs_sim_config(mode=mode, dither=dither, ndither=ndither,
                                                        grating=disperser,readmode=read_mode, detector=detector,
                                                        exposures=exposures, integrations=integrations,
                                                        groups=groups)

                        'Simulating %s' % sim_dir
                        try:
                            mysim = MiriSimulation(sim_cfg, scene_cfg, simulator_cfg)
                            mysim.run()

                            # log pass
                            testing_logger.info('%s passed' % sim_dir)

                            if dither == False:
                                fig,axs = plt.subplots(1, 4)
                                fig.set_figwidth(16.0)
                                fig.set_figheight(4.0)
                            elif dither == True:
                                fig,axs = plt.subplots(2,4)
                                fig.set_figwidth(16.0)
                                fig.set_figheight(8.0)
                            #plt.tight_layout(pad=0.5)
                            axs = axs.ravel()
                            axs_index = -1

                            # plot output, skycube, illumination model and last frame of first integration (only one per exposure)
                            # skycube, first channel
                            cube_file = get_output_product('skycube', channel=1)
                            hdulist = fits.open(cube_file)
                            sky_data = hdulist[0].data
                            hdulist.close()

                            axs_index += 1
                            axs[axs_index].imshow(np.sum(sky_data, axis=0), cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                            axs[axs_index].annotate(sim_dir, xy=(0.0,1.02), xycoords='axes fraction', fontsize=14, fontweight='bold', color='k')
                            axs[axs_index].annotate('skycube short', xy=(0.5,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                            # skycube, second channel
                            cube_file = get_output_product('skycube', channel=2)
                            hdulist = fits.open(cube_file)
                            sky_data = hdulist[0].data
                            hdulist.close()

                            axs_index += 1
                            axs[axs_index].imshow(np.sum(sky_data, axis=0), cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                            axs[axs_index].annotate('skycube long', xy=(0.5,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                            # illumination model
                            illum_file = get_output_product('illum')
                            illum_datamodel = miri_illumination_model.MiriIlluminationModel(illum_file)

                            axs_index += 1
                            axs[axs_index].imshow(illum_datamodel.intensity, cmap='jet', interpolation='nearest', origin='lower')
                            axs[axs_index].annotate('illum_model', xy=(0.5,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                            # det_image
                            det_file = get_output_product('det_image')
                            det_datamodel = datamodels.open(det_file)

                            axs_index += 1
                            axs[axs_index].imshow(det_datamodel.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                            axs[axs_index].annotate('det_image', xy=(0.5,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                            if dither == True:
                                # skycube low
                                cube_file = get_output_product('skycube', channel=1, dither=True)
                                hdulist = fits.open(cube_file)
                                sky_data = hdulist[0].data
                                hdulist.close()

                                axs_index += 1
                                axs[axs_index].imshow(np.sum(sky_data, axis=0), cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                                axs[axs_index].annotate('dither position 2', xy=(0.0,1.02), xycoords='axes fraction', fontsize=12, fontweight='bold', color='k')
                                axs[axs_index].annotate('skycube short', xy=(0.5,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                                # skycube high
                                cube_file = get_output_product('skycube', channel=2, dither=True)
                                hdulist = fits.open(cube_file)
                                sky_data = hdulist[0].data
                                hdulist.close()

                                axs_index += 1
                                axs[axs_index].imshow(np.sum(sky_data, axis=0), cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                                axs[axs_index].annotate('skycube long', xy=(0.5,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                                # illumination model
                                illum_file = get_output_product('illum', dither=True)
                                illum_datamodel = miri_illumination_model.MiriIlluminationModel(illum_file)

                                axs_index += 1
                                axs[axs_index].imshow(illum_datamodel.intensity, cmap='jet', interpolation='nearest', origin='lower')
                                axs[axs_index].annotate('illum_model', xy=(0.5,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                                # det_image
                                det_file = get_output_product('det_image', dither=True)
                                det_datamodel = datamodels.open(det_file)

                                axs_index += 1
                                axs[axs_index].imshow(det_datamodel.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                                axs[axs_index].annotate('det_image', xy=(0.5,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                            fig.savefig(os.path.join(out_fig_dir,sim_fig), dpi=200)
                            del fig

                        except Exception as e:
                            testing_logger.warning('%s failed' % sim_dir)
                            testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))

                        os.chdir(out_dir)



    # LRS simulations
    if lrs == True:
        testing_logger.info('Starting LRS simulations')

        if lrs_slits == True:
            modes = ['LRS_SLIT','LRS_SLITLESS']
        else: modes = ['LRS_SLIT']  # set as default

        lrs_filters = ['P750L']  # only 1 available

        if lrs_readmodes == True:
            read_modes = ['FAST', 'SLOW']
        else: read_modes = ['SLOW']  # set as default

        # set observation parameters (hardcoded for now)
        ndither = 2
        exposures=1
        integrations=1
        groups=20

        # set up pyplot for the imager simulations
        # need to know the number of rows. The following is a bit messy, can improve
        # in the future, e.g., using add_subplot
        if lrs_slits == True:
            len_modes = len(modes)
        else: len_modes = 1
        len_filters = 1
        if lrs_readmodes == True:
            len_readmodes = len(read_modes)
        else: len_readmodes = 1

        # determine the number of plot rows
        if dither == True: num_rows = len_modes * len_filters * len_readmodes * ndither
        else: num_rows = len_modes * len_filters * len_readmodes

        # specify the shape of the subplots
        plot_rows = num_rows
        plot_cols = 5

        fig,axs = plt.subplots(plot_rows, plot_cols)
        fig.set_figwidth(10.0)
        fig.set_figheight(2.0*num_rows)
        #plt.tight_layout(pad=0.5)
        axs = axs.ravel()
        axs_index = -1

        for mode in modes:
            for lrs_filter in lrs_filters:
                for read_mode in read_modes:
                    sim_dir = 'LRS' + mode + '_' + lrs_filter + '_' + read_mode + '_dithering-' + str(dither)
                    os.mkdir(sim_dir)
                    os.chdir(sim_dir)
                    generate_test_scene(output_scene='test_scene.ini', instrument='LRS',
                                        include_galaxy=include_galaxy, simple_scene=simple_scene)
                    generate_lrs_simulation_ini(scene='test_scene.ini', mode=mode, dither=dither,
                                                ndither=ndither, readmode=read_mode, exposures=exposures,
                                                integrations=integrations, groups=groups)

                    print('Simulating %s') % sim_dir
                    try:
                        mysim = MiriSimulation.from_configfiles('lrs_simulation.ini', scene_file='test_scene.ini')
                        mysim.run()

                        # log pass
                        testing_logger.info('%s passed' % sim_dir)

                        # plot output, illumination model and last frame of first integration (only one per exposure)
                        illum_file = get_output_product('illum')
                        illum_datamodel = miri_illumination_model.MiriIlluminationModel(illum_file)

                        axs_index += 1
                        axs[axs_index].imshow(illum_datamodel.intensity[0], cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                        axs[axs_index].annotate(sim_dir, xy=(0.0,1.02), xycoords='axes fraction', fontsize=14, fontweight='bold', color='k')
                        axs[axs_index].annotate('illum_model', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                        det_file = get_output_product('det_image')
                        det_datamodel = datamodels.open(det_file)

                        axs_index += 1
                        vmax = image_stats(det_datamodel.data[0][-1], 'mean')  + 1000.
                        axs[axs_index].imshow(det_datamodel.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(vmin=10000., vmax=vmax), origin='lower')
                        axs[axs_index].annotate('det_image', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                        if dither == True:
                            illum_file = get_output_product('illum', dither=True)
                            illum_datamodel = miri_illumination_model.MiriIlluminationModel(illum_file)

                            axs_index += 1
                            axs[axs_index].imshow(illum_datamodel.intensity[0], cmap='jet', interpolation='nearest', norm=LogNorm(), origin='lower')
                            axs[axs_index].annotate('dither position 2', xy=(0.0,1.02), xycoords='axes fraction', fontsize=12, fontweight='bold', color='k')
                            axs[axs_index].annotate('illum_model', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                            det_file = get_output_product('det_image', dither=True)
                            det_datamodel = datamodels.open(det_file)

                            axs_index += 1
                            vmax = image_stats(det_datamodel.data[0][-1], 'mean')  + 1000.
                            axs[axs_index].imshow(det_datamodel.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(vmin=10000., vmax=vmax), origin='lower')
                            axs[axs_index].annotate('det_image', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')


                    except Exception as e:
                        testing_logger.warning('%s failed' % sim_dir)
                        testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))

                    os.chdir(out_dir)

        fig.savefig('test_MIRISim_LRS_output.pdf', dpi=200)
        del fig


    os.chdir(cwd)


if __name__ == "__main__":

    break_mirisim(imager=False, ima_filters=False, ima_subarrays=False, ima_readmodes=False,
                 mrs=True, mrs_paths=False, mrs_gratings=False, mrs_detectors=False,
                 mrs_readmodes=True, lrs=False, lrs_slits=False, lrs_readmodes=False,
                 dither=True, scene='point')
