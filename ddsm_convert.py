#!/usr/bin/env python
"""
ddsm_convert.py generates compressed, open-format mamograms and label files at
their original resolution from the DDSM (Digital Database for Screening
Mammography) dataset. It expects, for each case, all of the files that can be
pulled from the official database here: ftp://figment.csee.usf.edu/pub/DDSM/
"""
import argparse
from glob import glob
import os, sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import png
from scipy import ndimage as ndi
from shutil import copyfile, rmtree
import subprocess as proc
import tempfile

# logs to ~/ddsm.log by default
HOME= str(Path.home())
logging.basicConfig(level=logging.WARN, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger(os.path.basename(__file__))

class Abnormality:
    """
    Represents overlay data from the ddms dataset.

    The keywords that describe the lesion type are taken from the ACR Bi-RADS
    lexicon. The assessment code is a value from 1 to 5, and also comes from the
    ACR Bi-RADS standard. The subtlety rating is not from the Bi-RADS standard.
    It is a value from 1 to 5, where 1 means "x1," 2 means "x2," 3 means "x3,"
    4 means "x4," and 5 means "x5."

    In some cases there is more than one outline for the same abnormality. In
    these situations the "TOTAL_OUTLINES" number is more than one. The first
    boundary will contain all of the other boundaries, and all boundaries after
    the first one will begin with the word "CORE".

    Each boundary is specified as a chain code. This chain code is found on the
    line after the keyword "BOUNDARY" or "CORE" as discussed above. The first
    two values of each chain code are the starting column and row of the chain
    code, followed by the chain code. "#" terminates the chain code. The numbers
    correspond to the directions as follows:

      X-->
     Y  7 0 1
     |  6 X 2
     v  5 4 3

    All boundaries found will be returned as their own binary numpy arrays.
    """
    def __init__(self, abnormality):
        """
        self.img = binary numpy array of segmentation
        """
        self.img = None
        self.lesion_type = ''
        self.assessment = None
        self.subtlety = None
        self.pathology = None
        self.n_outlines = None
        self.boundaries = [] # list of chain codes
        self.segmentations = [] # list of numpy arrays
        self.proportions = [] # list of segmentation coverage (as a %)

        # the end of each .OVERLAY file is a set of boundaries
        collect_boundaries = False

        for l in abnormality:
            if collect_boundaries:
                if l.strip() not in ['BOUNDARY', 'CORE']:
                    # replace # (chaincode termination signal) with a valid int
                    # and strip() collapses whitespace so we can split
                    candidate = np.array(l.replace('#', '-1').strip().split()).astype(np.int)
                    if len(candidate):
                        self.boundaries.append(candidate)

            else:
                fields = l.strip().split(' ')
                if l.startswith('LESION_TYPE'):
                    self.lesion_type += ' '.join(fields[1:]) + ' '
                elif l.startswith('ASSESSMENT'):
                    self.assessment = int(fields[1])
                elif l.startswith('SUBTLETY'):
                    self.subtlety = int(fields[1])
                elif l.startswith('PATHOLOGY'):
                    self.pathology = fields[1]
                elif l.startswith('TOTAL_OUTLINES'):
                    self.n_outlines = int(fields[1])
                    collect_boundaries = True


    def gen_segs(self, rows, cols, name, scan, output):
        """
        Generates a segmentation (binary numpy array) given the found
        boundaries, and renders it as a 2-bit png.
        """
        for i, chain_code in enumerate(self.boundaries):

            filename = os.path.join(output, '{0}.{1}_SEG_{2:03d}.png'.format(name, scan, i+1))
            logger.debug('generating segmentation for {}'.format(filename))
            # don't repeat work
            if os.path.isfile(filename):
                continue

            segmentation = np.zeros((rows, cols))
            # NB: chain code starts w cols, rows (flipped from numpy convention)
            idx = np.array([chain_code[1], chain_code[0]]) # rows, cols
            segmentation[idx[0], idx[1]] = 1

            for c in range(2, len(chain_code)):
                if   chain_code[c] == 0: idx += [-1,  0]
                elif chain_code[c] == 1: idx += [-1,  1]
                elif chain_code[c] == 2: idx += [ 0,  1]
                elif chain_code[c] == 3: idx += [ 1,  1]
                elif chain_code[c] == 4: idx += [ 1,  0]
                elif chain_code[c] == 5: idx += [ 1, -1]
                elif chain_code[c] == 6: idx += [ 0, -1]
                elif chain_code[c] == 7: idx += [-1, -1]

                segmentation[idx[0], idx[1]] = 1

            segmentation = ndi.binary_fill_holes(segmentation)
            proportion = (np.sum(segmentation) / (rows*cols)) *100

            self.segmentations.append(segmentation)
            self.proportions.append(proportion)
            png.from_array(segmentation, mode='L').save(filename)


class Metadata:
    """
    The .ics file included for each scan contains crucial information for
    reconstructing the data, and useful metadata about the patient.

    Returns an object with all parameters extracted.
    """
    def __init__(self, ics):
        self.name = None
        self.site = None
        self.age = None
        self.density = None
        self.scandate = None
        self.digitizer = None
        self.scans = {}

        # first space-delimited field of each line denotes data type
        # values can be of variable length
        f = open(ics, 'r')

        sequences = []
        collect_sequences = False

        for l in [line.strip() for line in f.readlines()]:

            # skips empty entries
            if not l:
                continue

            fields = l.split(' ') # used later

            if collect_sequences:
                sequences.append(l)

            # subject ID and corresponding site
            elif l.startswith('filename'):

                self.name = fields[1].replace('-', '_')

                if self.name.startswith('A'):
                    self.site = 'MGH'
                elif self.name.startswith('B') or self.name.startswith('C'):
                    self.site = 'WFU'
                elif self.name.startswith('D'):
                    self.site = 'ISMD'

            elif l.startswith('PATIENT_AGE'):
                self.age = int(fields[1])
            elif l.startswith('DENSITY'):
                self.density = int(fields[1])
            elif l.startswith('DATE_OF_STUDY'):
                self.scandate = ' '.join(fields[1:])

            # uses .ics digitizer name and site to find final name
            # this is used by ddsmraw2pnm to normalize grey values to optical
            # density values across scanners
            elif l.startswith('DIGITIZER'):
                digitizer_name = ' '.join(fields[1:])
                if 'DBA' in digitizer_name:
                    self.digitizer = 'dba'
                elif 'LUMISYS' in digitizer_name:
                    self.digitizer = 'lumisys'
                elif 'HOWTEK' in digitizer_name and self.site == 'MGH':
                    self.digitizer = 'howtek-mgh'
                elif 'HOWTEK' in digitizer_name and self.site == 'ISMD':
                    self.digitizer = 'howtek-ismd'

            # assumes that the list of sequences is last in all .ics files
            elif l.startswith('SEQUENCE'):
                collect_sequences = True

        # sequences have format NAME,ROWS,COLS,BPP,RES,IS_OVERLAY
        # BPP = bits per pixel
        self.scans = {}
        for seq in sequences:
            logger.debug('parsing sequence: {}'.format(seq))
            seq = seq.split(' ')
            seq_name = seq[0]

            self.scans[seq_name] = {}
            self.scans[seq_name]['rows'] = int(seq[2])
            self.scans[seq_name]['cols'] = int(seq[4])
            self.scans[seq_name]['bpp'] = int(seq[6])
            self.scans[seq_name]['res'] = float(seq[8])
            if seq[-1] == "NON_OVERLAY":
                self.scans[seq_name]['has_overlay'] = False
            elif seq[-1] == "OVERLAY":
                self.scans[seq_name]['has_overlay'] = True


def only_one(l):
    """complains if list l contains more than one element"""
    if len(l) != 1:
        return(False)
    else:
        return(True)


def run(cmd):
    """
    Runs the command in default shell, returning STDOUT and a return code.
    """
    logger.debug("running: {}".format(cmd))
    p = proc.Popen(cmd, shell=True, stdout=proc.PIPE, stderr=proc.PIPE)
    out, err = p.communicate()

    if p.returncode > 0:
        logger.error('{} returned {}: {}'.format(cmd, p.returncode, err))

    return(p.returncode, out)


def generate_segmentations(subject, output, files, metadata, dataframe, tempdir='/tmp'):
    """
    reads overlay files, and writes out each abnormality into it's own file.
    """
    for scan in list(metadata.scans.keys()):
        #if not metadata.scans[scan]['has_overlay']:
        #    continue

        overlay = list(filter(lambda x: scan + '.OVERLAY' in x, files))
        if not only_one(overlay):
            logger.debug('found {} overlays for scan {}, skipping'.format(
                len(overlay), scan))
            continue

        fstem = os.path.splitext(overlay[0])[0]

        # grab each abnormality as a blob of text
        f = open(os.path.join(subject, overlay[0]), 'r').read()
        abnormalities = f.split('ABNORMALITY')[1:] # idx 0 is abnormality count
        for abnormality_text in abnormalities:

            # split lines here so Abnormality behaves similarly to Metadata
            abnormality = Abnormality(abnormality_text.splitlines())

            # render segmentations to disk
            rows = metadata.scans[scan]['rows']
            cols = metadata.scans[scan]['cols']
            abnormality.gen_segs(rows, cols, metadata.name, scan, output)

            # save metadata to dataframe
            df.loc[os.path.basename(subject)] = [abnormality.assessment,
                                                 abnormality.pathology,
                                                 abnormality.subtlety,
                                                 abnormality.lesion_type,
                                                 metadata.age,
                                                 metadata.site,
                                                 metadata.density]


def convert_scans(subject, output, files, metadata, tempdir='/tmp'):
    """
    file conversion pipeline:
      - ljpeg --> RAW: jpeg -d -s {ljpeg_file}" (appends .1 to filename)
      - RAW --> pnm: ddsmraw2pnm
      - pnm --> png: convert -depth 16 {pnm_file} {png_file}
    """
    working_dir_exists = False

    for scan in list(metadata.scans.keys()):

        # obtain the input data (propriatary .LJPEG format)
        ljpeg = list(filter(lambda x: scan + '.LJPEG' in x, files))
        assert only_one(ljpeg)
        fstem = os.path.splitext(ljpeg[0])[0]

        # skip work if we already have the output file
        png = os.path.join(output, fstem + '.png')
        if os.path.isfile(png):
            continue

        if not working_dir_exists:
            working_dir = tempfile.mkdtemp(dir=tempdir)
            logger.debug('generating working directory {}'.format(working_dir))
            working_dir_exists = True

        # have to copy input file because conversion tool does not respond to
        # output path specification
        ljpeg_raw = os.path.join(subject, ljpeg[0])
        ljpeg_tmp = os.path.join(working_dir, ljpeg[0])
        raw = ljpeg_tmp + '.1'
        pnm = raw + '-ddsmraw2pnm.pnm'

        copyfile(ljpeg_raw, ljpeg_tmp)

        run('jpeg -d -s {}'.format(ljpeg_tmp)) # outputs a file with .1 appended
        run('ddsmraw2pnm {filename} {n_rows} {n_cols} {digitizer}'.format(
            filename=raw,
            n_rows=metadata.scans[scan]['rows'],
            n_cols=metadata.scans[scan]['cols'],
            digitizer=metadata.digitizer))
        run('convert -depth 16 {} {}'.format(pnm, png))

    if working_dir_exists:
        rmtree(working_dir)


def main(subject, output, df, tempdir='/tmp'):

    files = os.listdir(subject)
    ics_file = list(filter(lambda x: '.ics' in x, files))
    assert only_one(ics_file)

    # extracts all metadata from the .ics file
    metadata = Metadata(os.path.join(subject, ics_file[0]))

    # converts .LJPEG format to a compressed 16-bit .png
    convert_scans(subject, output, files, metadata, tempdir=tempdir)

    # generates segmentations from .OVERLAY files (saved as 2-bit .png)
    generate_segmentations(subject, output, files, metadata, df, tempdir=tempdir)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('subject', help='DDSM subject folder (for an individual), or text tile for batch mode')
    argparser.add_argument('output', help='folder to place all outputs')
    argparser.add_argument('-t', '--tempdir', help='specify a custom working directory')
    argparser.add_argument('-v', '--verbose', action='count', help='turns on debug messages')
    argparser.add_argument('-l', '--log', help='specify log file')
    args = argparser.parse_args()

    # determine if input is subject folder or batch file
    if os.path.isfile(args.subject):
        batch_mode = True
    else:
        batch_mode = False

    # set logging
    if args.log:
        logger_handle = logging.FileHandler(args.log)
    else:
        logger_handle = logging.FileHandler(os.path.join(HOME, 'ddsm.log'))

    logger_handle.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(logger_handle)

    # set debugging
    logger.info('starting')
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # handle temporary directory
    if args.tempdir:
        tempdir = args.tempdir
    else:
        tempdir = '/tmp'

    # metadata file
    metadata_file = os.path.join(args.output, 'metadata.csv')
    if os.path.isfile(metadata_file):
        df = pd.read_csv(metadata_file, index_col=0)
    else:
        df = pd.DataFrame(columns=['assessment', 'pathology', 'subtlety', 'lesion_type', 'age', 'site', 'density'])
        df.index.name = 'id'

    # batch mode or single mode
    if batch_mode:
        f = open(args.subject, 'r')
        subjects = f.readlines()

        for subject in subjects:
            main(subject.strip(), args.output, df, tempdir)
    else:
        main(args.subject, args.output, df, tempdir)

    df.to_csv(metadata_file)

