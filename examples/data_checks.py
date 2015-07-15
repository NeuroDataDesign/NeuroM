#!/usr/bin/env python

# Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
# All rights reserved.
#
# This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the names of
#        its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''Examples of basic data checks'''
from neurom.io.readers import load_data
from neurom.io import check as io_chk
import argparse
import os


EXAMPLE_TEXT = '''
Examples
--------
data_checks.py --help               # print this help
data_checks.py some/path/neuron.h5  # Process an HDF5 file
data_checks.py some/path/neuron.swc # Process an SWC file
data_checks.py some/path/           # Process all HDF5 and SWC files found in directory
'''


def parse_args():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(description='NeuroM Data Checker',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=EXAMPLE_TEXT)
    parser.add_argument('datapath',
                        help='Path to morphology data file or directory')

    return parser.parse_args()


def get_morph_files(directory):
    '''Get a list of all morphology files in directory dir

    Returns files with extensions '.swc' or '.h5' (case insensitive)
    '''
    lsdir = [os.path.join(directory, m) for m in os.listdir(directory)]
    return [m for m in lsdir
            if os.path.isfile(m) and
            os.path.splitext(m)[1].lower() in ('.swc', '.h5')]


if __name__ == '__main__':

    args = parse_args()
    data_path = args.datapath
    if os.path.isfile(data_path):
        files = [data_path]
    elif os.path.isdir(data_path):
        print 'Checking files in directory', data_path
        files = get_morph_files(data_path)

    for f in files:
        rd = load_data(f)
        print '\nCheck file %s...' % f
        print 'Has soma? %s' % io_chk.has_soma(rd)
        fr = io_chk.has_all_finite_radius_neurites(rd)
        print 'All neurites have finite radius? %s' % fr[0]
        if not fr[0]:
            print 'Points with zero radius detected:', fr[1]
        ci = io_chk.has_sequential_ids(rd)
        print 'Consecutive indices? %s' % ci[0]
        if not ci[0]:
            print'Non consecutive IDs detected:', ci[1]
        else:
            fs = io_chk.has_all_finite_length_segments(rd)
            print 'Finite length segments? %s' % fs[0]
            if not fs[0]:
                print '\tSegments with zero length detected:', fs[1]
