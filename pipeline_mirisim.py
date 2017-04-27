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


def pipeline_mirisim(input_dir):
    """

    """
    # check if the simulation folder exists
    try:
        with open(input_dir) as indir:
            pass
    except IOError:
        print("Simulation folder not found")


    # go through individual simulations and run through pipeline
    simulations = glob.glob(os.path.join(input_dir,'IMA*'))
    simulations.extend(glob.glob(os.path.join(input_dir,'MRS*')))
    simulations.extend(glob.glob(os.path.join(input_dir,'LRS*')))
    for simulation in simulations: print(simulation)

if __name__ == "__main__":

    #TODO add proper command line parser
    pipeline_mirisim(input_dir=sys.argv[1])
