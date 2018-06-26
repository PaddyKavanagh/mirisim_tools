#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# @author: Patrick Kavanagh (DIAS)
#
"""

Try all possible scene components, only one source per sim, imager only

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

def make_ima_sim_config(mode='FULL', dither=False, ndither=2, filter='F1130W',
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
                                   filter=filter,
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


def make_simulator_config(noise=True):
    mrs_simulator_config = SimulatorConfig.from_default()
    mrs_simulator_config['SCASim']['include_refpix'] = 'T'
    mrs_simulator_config['SCASim']['include_badpix'] = 'T'
    mrs_simulator_config['SCASim']['include_dark'] = 'T'
    mrs_simulator_config['SCASim']['include_flat'] = 'T'
    mrs_simulator_config['SCASim']['include_gain'] = 'T'
    mrs_simulator_config['SCASim']['include_nonlinearity'] = 'T'
    mrs_simulator_config['SCASim']['include_drifts'] = 'F'
    mrs_simulator_config['SCASim']['include_latency'] = 'F'

    if not noise:
        mrs_simulator_config['SCASim']['cosmic_ray_mode'] = 'NONE'
        mrs_simulator_config['SCASim']['include_poisson'] = 'F'
        mrs_simulator_config['SCASim']['include_readnoise'] = 'F'
    else:
        mrs_simulator_config['SCASim']['cosmic_ray_mode'] = 'SOLAR_MIN'
        mrs_simulator_config['SCASim']['include_poisson'] = 'T'
        mrs_simulator_config['SCASim']['include_readnoise'] = 'T'

    return mrs_simulator_config


def make_scene_config(src_type='point', src_spec='bb', vel=None, losvd=False):
    """
    make scene object

    src_type:   point or galaxy
    src_spec:   bb, pl,
    """
    if src_type == 'point':
        #build scene
        background = Background(level='low', gradient=5., pa=15.0, centreFOV=(0., 0.))

        # spec
        if src_spec == 'bb':
            SED1 = BBSed(Temp=300., wref=10., flux=1.e4)
        elif src_spec == 'pl':
            SED1 = PLSed(alpha=1., wref=10., flux=1.e4)

        Point1 = Point(Cen=(0.,0.), vel=0.0)
        Point2 = Point(Cen=(10., 10.), vel=0.0)
        Point1.set_SED(SED1)
        Point2.set_SED(SED1)
        targets = [Point1, Point2]

        # create config object
        scene_config = SceneConfig.makeScene(loglevel=0, background=background, targets=targets)

    elif src_type == 'galaxy':
        #build scene
        background = Background(level='low', gradient=5., pa=15.0, centreFOV=(0., 0.))

        # spec
        SED1 = BBSed(Temp=300., wref=10., flux=1.e6)

        Galaxy1 = Galaxy(Cen=(0.,0.), n=2., re=20., pa=59., q=0.8)
        Galaxy2 = Galaxy(Cen=(20., 20.), n=1., re=50., pa=120., q=0.2)
        Galaxy1.set_SED(SED1)
        Galaxy2.set_SED(SED1)

        # vel
        if vel is not None:
            if vel == 'flatdisk':
                velomap1 = FlatDisk(Cen=(0.,0.), vrot=200., pa=59., q=0.8, c=0.)
                velomap2 = FlatDisk(Cen=(20., 20.), vrot=500., pa=120., q=0.2, c=0.)
            elif vel == 'keplerdisk':
                velomap1 = KeplerDisk(Cen=(0., 0.), v0=200., r0=10., pa=59., q=0.8, c=0.)
                velomap2 = KeplerDisk(Cen=(20., 20.), v0=500., r0=25., pa=120., q=0.2, c=0.)

            Galaxy1.set_velomap(velomap1)
            Galaxy2.set_velomap(velomap2)

        if losvd:
            losvd1 = Losvd(sigma=200., h3=3., h4=1.)
            losvd2 = Losvd(sigma=300., h3=5., h4=3.)

            Galaxy1.set_LOSVD(losvd1)
            Galaxy2.set_LOSVD(losvd2)

        targets = [Galaxy1, Galaxy2]

        # create config object
        scene_config = SceneConfig.makeScene(loglevel=0, background=background, targets=targets)

    elif src_type == 'exp_disk':
        #build scene
        background = Background(level='low', gradient=5., pa=15.0, centreFOV=(0., 0.))

        # spec
        if src_spec == 'bb':
            SED1 = BBSed(Temp=300., wref=10., flux=1.e6)
        elif src_spec == 'pl':
            SED1 = PLSed(alpha=1., wref=10., flux=1.e6)

        Galaxy1 = ExpDisk(Cen=(0.,0.), h=20., q=0.8, pa=30.)
        Galaxy2 = ExpDisk(Cen=(20., 20.), h=50., q=0.2, pa=80.)
        Galaxy1.set_SED(SED1)
        Galaxy2.set_SED(SED1)
        targets = [Galaxy1, Galaxy2]

        # create config object
        scene_config = SceneConfig.makeScene(loglevel=0, background=background, targets=targets)

    elif src_type == 'sersic_disk':
        #build scene
        background = Background(level='high', gradient=5., pa=15.0, centreFOV=(0., 0.))

        # spec
        if src_spec == 'bb':
            SED1 = BBSed(Temp=300., wref=10., flux=1.e6)
        elif src_spec == 'pl':
            SED1 = PLSed(alpha=1., wref=10., flux=1.e6)

        Galaxy1 = SersicDisk(Cen=(0.,0.), q=0.8, pa=30., n=1.0, re=50)
        Galaxy2 = SersicDisk(Cen=(20., 20.), q=0.2, pa=80, n=2.0, re=100.)
        Galaxy1.set_SED(SED1)
        Galaxy2.set_SED(SED1)
        targets = [Galaxy1, Galaxy2]

        # create config object
        scene_config = SceneConfig.makeScene(loglevel=0, background=background, targets=targets)

    elif src_type == 'pysynphot':
        #build scene
        background = Background(level='low', gradient=5., pa=15.0, centreFOV=(0., 0.))

        # spec
        SED1 = PYSPSed(family='bc95', sedname='bc95_b_10E6', wref=10., flux=1.e4)
        #SED2 = PYSPSed(family='ck04models', sedname='ckm10/ckm10_3500', wref=10., flux=1.e4)
        SED2 = PYSPSed(family='bc95', sedname='bc95_b_10E6', wref=10., flux=1.e4)
        SED3 = PYSPSed(family='agn', sedname='liner_template', wref=10., flux=1.e4)
        SED4 = PYSPSed(family='kc96', sedname='s0_template', wref=10., flux=1.e6)
        SED5 = PYSPSed(family='kc96', sedname='elliptical_template', wref=10., flux=1.e5)

        Point1 = Point(Cen=(0.,0.), vel=0.0)
        Point2 = Point(Cen=(10., 10.), vel=0.0)
        Point3 = Point(Cen=(-20., -20.), vel=0.0)
        Galaxy1 = Galaxy(Cen=(0., 0.), n=2., re=20., pa=59., q=0.3)
        Galaxy2 = Galaxy(Cen=(-30., 30.), n=1., re=30., pa=80., q=0.9)
        Point1.set_SED(SED1)
        Point2.set_SED(SED2)
        Point3.set_SED(SED3)
        Galaxy1.set_SED(SED4)
        Galaxy2.set_SED(SED5)
        targets = [Point1, Point2, Point3, Galaxy1, Galaxy2]

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


def break_mirisim_scene(imager=True, ima_filters=False, ima_subarrays=False, ima_readmodes=True,
                 dither=False, noise=True):
    """
    """
    # set cwd
    cwd = os.getcwd()

    # set the output directory
    out_dir = os.path.join(cwd,'mirisim_scene_functional_tests')
    try: shutil.rmtree(out_dir)
    except: pass
    os.mkdir(out_dir)

    # set the output figure directory
    out_fig_dir = os.path.join(cwd,'mirisim_scene_functional_tests','simulation_plots')
    try: shutil.rmtree(out_fig_dir)
    except: pass
    os.mkdir(out_fig_dir)

    # move to out_dir
    os.chdir(out_dir)

    # set up logging
    log_file = os.path.join(out_dir,'test_MIRISim_scene.log')
    testing_logger = logging.getLogger(__name__)
    testing_logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    testing_logger.addHandler(handler)

    # generate the default simultor config object
    simulator_cfg = make_simulator_config(noise=noise)

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

        # sources with no vel distribution
        src_types = ['point', 'galaxy', 'exp_disk', 'sersic_disk', 'pysynphot']
        src_specs = ['bb', 'pl']

        for mode in modes:
            for im_filter in im_filters:
                for read_mode in read_modes:
                    for src_type in src_types:
                        for src_spec in src_specs:

                            scene_cfg = make_scene_config(src_type=src_type, src_spec=src_spec)

                            # set the number of groups depending on the readout mode
                            if read_mode == 'FAST': groups=50
                            elif read_mode == 'SLOW': groups=10

                            sim_dir = 'IMA_' + mode + '_' + im_filter + '_' + read_mode + '_dithering-' + str(dither) + \
                                      '_' + src_type + '_' + src_spec
                            sim_fig = sim_dir + '.pdf'
                            os.mkdir(sim_dir)
                            os.chdir(sim_dir)

                            sim_cfg = make_ima_sim_config(mode=mode, dither=dither, ndither=ndither,
                                                            filter=im_filter,readmode=read_mode,
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
                                vmin = np.median(det_datamodel.data[0][-1]) * 0.9
                                vmax = np.median(det_datamodel.data[0][-1]) * 4
                                axs[axs_index].imshow(det_datamodel.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower')
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
                                    axs[axs_index].imshow(det_datamodel.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower')
                                    axs[axs_index].annotate('det_image', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                                fig.savefig(os.path.join(out_fig_dir,sim_fig), dpi=200)
                                del fig

                            except Exception as e:
                                testing_logger.warning('%s failed' % sim_dir)
                                testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))

                            os.chdir(out_dir)

        # pysnphot
        for mode in modes:
            for im_filter in im_filters:
                for read_mode in read_modes:

                            scene_cfg = make_scene_config(src_type='pysynphot')

                            # set the number of groups depending on the readout mode
                            if read_mode == 'FAST':
                                groups = 50
                            elif read_mode == 'SLOW':
                                groups = 10

                            sim_dir = 'IMA_' + mode + '_' + im_filter + '_' + read_mode + '_dithering-' + str(dither) + \
                                     '_pysynphot'
                            sim_fig = sim_dir + '.pdf'
                            os.mkdir(sim_dir)
                            os.chdir(sim_dir)

                            sim_cfg = make_ima_sim_config(mode=mode, dither=dither, ndither=ndither,
                                                          filter=im_filter, readmode=read_mode,
                                                          exposures=exposures,
                                                          integrations=integrations,
                                                          groups=groups)

                            print('Simulating %s' % sim_dir)
                            try:
                                mysim = MiriSimulation(sim_cfg, scene_cfg, simulator_cfg)
                                mysim.run()

                                # log pass
                                testing_logger.info('%s passed' % sim_dir)

                                if dither == False:
                                    fig, axs = plt.subplots(1, 2)
                                    fig.set_figwidth(12.0)
                                    fig.set_figheight(6.0)
                                elif dither == True:
                                    fig, axs = plt.subplots(2, 2)
                                    fig.set_figwidth(12.0)
                                    fig.set_figheight(12.0)
                                # plt.tight_layout(pad=0.5)
                                axs = axs.ravel()
                                axs_index = -1

                                # plot output, illumination model and last frame of first integration (only one per exposure)
                                illum_file = get_output_product('illum')
                                illum_datamodel = miri_illumination_model.MiriIlluminationModel(
                                    illum_file)

                                axs_index += 1
                                axs[axs_index].imshow(illum_datamodel.intensity[0], cmap='jet',
                                                      interpolation='nearest', norm=LogNorm(),
                                                      origin='lower')
                                axs[axs_index].annotate(sim_dir, xy=(0.0, 1.02),
                                                        xycoords='axes fraction', fontsize=14,
                                                        fontweight='bold', color='k')
                                axs[axs_index].annotate('illum_model', xy=(0.7, 0.95),
                                                        xycoords='axes fraction', fontsize=10,
                                                        fontweight='bold', color='w')

                                det_file = get_output_product('det_image')
                                det_datamodel = datamodels.open(det_file)

                                axs_index += 1
                                vmin = np.median(det_datamodel.data[0][-1]) * 0.9
                                vmax = np.median(det_datamodel.data[0][-1]) * 4
                                axs[axs_index].imshow(det_datamodel.data[0][-1], cmap='jet',
                                                      interpolation='nearest',
                                                      norm=LogNorm(vmin=vmin, vmax=vmax),
                                                      origin='lower')
                                axs[axs_index].annotate('det_image', xy=(0.7, 0.95),
                                                        xycoords='axes fraction', fontsize=10,
                                                        fontweight='bold', color='w')

                                if dither == True:
                                    illum_file = get_output_product('illum', dither=True)
                                    illum_datamodel = miri_illumination_model.MiriIlluminationModel(
                                        illum_file)

                                    axs_index += 1
                                    axs[axs_index].imshow(illum_datamodel.intensity[0], cmap='jet',
                                                          interpolation='nearest', norm=LogNorm(),
                                                          origin='lower')
                                    axs[axs_index].annotate('dither position 2', xy=(0.0, 1.02),
                                                            xycoords='axes fraction', fontsize=12,
                                                            fontweight='bold', color='k')
                                    axs[axs_index].annotate('illum_model', xy=(0.7, 0.95),
                                                            xycoords='axes fraction', fontsize=10,
                                                            fontweight='bold', color='w')

                                    det_file = get_output_product('det_image', dither=True)
                                    det_datamodel = datamodels.open(det_file)

                                    axs_index += 1
                                    axs[axs_index].imshow(det_datamodel.data[0][-1], cmap='jet',
                                                          interpolation='nearest',
                                                          norm=LogNorm(vmin=vmin, vmax=vmax),
                                                          origin='lower')
                                    axs[axs_index].annotate('det_image', xy=(0.7, 0.95),
                                                            xycoords='axes fraction', fontsize=10,
                                                            fontweight='bold', color='w')

                                fig.savefig(os.path.join(out_fig_dir, sim_fig), dpi=200)
                                del fig

                            except Exception as e:
                                testing_logger.warning('%s failed' % sim_dir)
                                testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))

                            os.chdir(out_dir)


        # galaxies with velomaps and LOSVD profiles
        src_velomap = ['flatdisk', 'keplerdisk']
        src_losvd = [True, False]

        for mode in modes:
            for im_filter in im_filters:
                for read_mode in read_modes:
                        for src_vd in src_losvd:
                            for src_vm in src_velomap:

                                scene_cfg = make_scene_config(src_type='galaxy', src_spec='bb', losvd=src_vd, vel=src_vm)

                                # set the number of groups depending on the readout mode
                                if read_mode == 'FAST': groups=50
                                elif read_mode == 'SLOW': groups=10

                                sim_dir = 'IMA_' + mode + '_' + im_filter + '_' + read_mode + '_dithering-' + str(dither)\
                                          + '_losvd-' + str(src_vd) + '_velomap-' + str(src_vm)
                                sim_fig = sim_dir + '.pdf'
                                os.mkdir(sim_dir)
                                os.chdir(sim_dir)

                                sim_cfg = make_ima_sim_config(mode=mode, dither=dither, ndither=ndither,
                                                                filter=im_filter,readmode=read_mode,
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
                                    vmin = np.median(det_datamodel.data[0][-1]) * 0.9
                                    vmax = np.median(det_datamodel.data[0][-1]) * 4
                                    axs[axs_index].imshow(det_datamodel.data[0][-1], cmap='jet',
                                                          interpolation='nearest', norm=LogNorm(vmin=vmin, vmax=vmax),
                                                          origin='lower')
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
                                        axs[axs_index].imshow(det_datamodel.data[0][-1], cmap='jet', interpolation='nearest', norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower')
                                        axs[axs_index].annotate('det_image', xy=(0.7,0.95), xycoords='axes fraction', fontsize=10, fontweight='bold', color='w')

                                    fig.savefig(os.path.join(out_fig_dir,sim_fig), dpi=200)
                                    del fig

                                except Exception as e:
                                    testing_logger.warning('%s failed' % sim_dir)
                                    testing_logger.warning('  %s: %s' % (e.__class__.__name__, str(e)))

                                os.chdir(out_dir)
    os.chdir(cwd)


if __name__ == "__main__":

    #TODO add command line parser
    break_mirisim_scene(imager=True, ima_filters=False, ima_subarrays=False, ima_readmodes=True,
                 dither=True, noise=False)
