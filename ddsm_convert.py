#!/usr/bin/env python
"""
ddsm_convert.py generates compressed, open-format mamograms and label files at
their original resolution from the DDSM (Digital Database for Screening
Mammography) dataset. It expects, for each case, all of the files that can be
pulled from the official database here: ftp://figment.csee.usf.edu/pub/DDSM/

ZARR STRUCTURE:

root = zarr.open_group('output.zarr', mode='a')
subj = root.create_group('A_1614_1') # subject

subj.attrs['age']
subj.attrs['site']
subj.attrs['density']
subj.attrs['pathology'] # default healthy

subj[scan].attrs['rows']
subj[scan].attrs['cols']
subj[scan].attrs['assessment'] # 1-5
subj[scan].attrs['subtlety'] # 1-5
subj[scan].attrs['lesion_type'] # unstructured text
subj[scan].attrs['coverage'] # list, one entry per boundary

where scan is one of ['LEFT_MLO', 'RIGHT_MLO', 'LEFT_CC', 'RIGHT_CC']

images: [:,:,0] = scan, [:,:,1] = border (broad), [:,:,2:] = nodules (fine)
NB: not all scans with a border have nodules defined

subj.create_dataset('LEFT_MLO')
subj.create_dataset('RIGHT_MLO')
subj.create_dataset('LEFT_CC')
subj.create_dataset('RIGHT_CC')
"""
import argparse, os, sys
from glob import glob
import logging
import numpy as np
from pathlib import Path
import png
from scipy import ndimage as ndi
from shutil import copyfile, rmtree
import subprocess as proc
import tempfile
import zarr

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
                    try:
                        self.assessment = int(fields[1])
                    except IndexError:
                        self.assessment = -1
                elif l.startswith('SUBTLETY'):
                    try:
                        self.subtlety = int(fields[1])
                    except IndexError:
                        self.subtlety = -1
                elif l.startswith('PATHOLOGY'):
                    try:
                        self.pathology = fields[1]
                    except IndexError:
                        self.pathology = 'N/A'
                elif l.startswith('TOTAL_OUTLINES'):
                    self.n_outlines = int(fields[1])
                    collect_boundaries = True


    def gen_segs(self, name, scan, zarr_subj):
        """
        Generates a segmentation (binary numpy array) given the found
        boundaries, and renders it as a 2-bit png.
        """
        image = zarr_subj[scan][:]
        rows = image.shape[0]
        cols = image.shape[1]

        # dont repeat work (i.e., dataset already includes the boundaries)
        if image.shape[-1] == len(self.boundaries)+1:
            return

        # we append segmentations to this list
        segmentations = []
        proportions = []

        for i, chain_code in enumerate(self.boundaries):

            logger.debug('generating segmentation {} for {}/{}'.format(
                i+1, zarr_subj.path, scan))

            segmentation = np.zeros((rows, cols))

            # NB: idx = [row, col], chain_code = [col, row]
            for c in range(1, len(chain_code)):
                if c == 1: idx = np.array([chain_code[1], chain_code[0]]) # init
                if   chain_code[c] == 0: idx += [-1,  0]
                elif chain_code[c] == 1: idx += [-1,  1]
                elif chain_code[c] == 2: idx += [ 0,  1]
                elif chain_code[c] == 3: idx += [ 1,  1]
                elif chain_code[c] == 4: idx += [ 1,  0]
                elif chain_code[c] == 5: idx += [ 1, -1]
                elif chain_code[c] == 6: idx += [ 0, -1]
                elif chain_code[c] == 7: idx += [-1, -1]

                # Occasionally, the chain code goes over the edge of the image.
                # in this case we replace that index with the index of the
                # border. Should keep an eye on this, but in tested cases the
                # outputs look accurate.
                try:
                    segmentation[idx[0], idx[1]] = 1
                except:
                    logger.debug('edge case in segmentation found, inspect {}'.format(filename))
                    if idx[0] == rows and idx[1] == cols:
                        segmentation[rows-1, cols-1] = 1
                    elif idx[0] >= rows:
                        segmentation[rows-1, idx[1]] = 1
                    elif idx[1] >= cols:
                        segmentation[idx[0], cols-1] = 1

            segmentation = ndi.binary_fill_holes(segmentation)
            proportion = (np.sum(segmentation) / (rows*cols)) *100

            segmentations.append(segmentation)
            proportions.append(proportion)

        # append segmentations to the image
        segmentations = np.stack(segmentations, axis=2)
        zarr_subj[scan].append(segmentations, axis=2)
        zarr_subj[scan].attrs['coverage'] = proportions


class Metadata:
    """
    The .ics file included for each scan contains crucial information for
    reconstructing the data, and useful metadata about the patient.

    Returns an object with the following parameters extracted: name, site, age
    density, scandate, digitizer, and scans.

    'scans' is a dictionary with the following fields:
    scans
        - rows (# of rows)
        - cols (# of cols)
        - bpp  (bits per pixel)
        - res  (resolution)
        - has_overlay (bool)
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

            # skips empty lines
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
                try:
                    self.age = int(fields[1])
                except IndexError:
                    self.age = -1
            elif l.startswith('DENSITY'):
                try:
                    self.density = int(fields[1])
                except IndexError:
                    self.density = -1
            elif l.startswith('DATE_OF_STUDY'):
                try:
                    self.scandate = ' '.join(fields[1:])
                except IndexError:
                    self.scandate = 'N/A'

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
        # no error handling yet - we assume all sequences are formatted the same
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
    """false if list l contains more than one element"""
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


def generate_segmentations(raw_dir, zarr_subj, files, metadata, tempdir='/tmp'):
    """
    reads overlay files, and writes out each abnormality into a numpy array
    concatenated with it's associated image into the zarr directory structure.
    Also overwrites any initialized disease-related attributes.
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
        f = open(os.path.join(raw_dir, overlay[0]), 'r').read()
        abnormalities = f.split('ABNORMALITY')[1:] # idx 0 is abnormality count
        for abnormality_text in abnormalities:

            # split lines here so Abnormality behaves similarly to Metadata
            abnormality = Abnormality(abnormality_text.splitlines())

            # render segmentations to disk
            abnormality.gen_segs(metadata.name, scan, zarr_subj)

            # save anbnormality labels
            zarr_subj.attrs['pathology'] = abnormality.pathology
            zarr_subj[scan].attrs['assessment'] = abnormality.assessment
            zarr_subj[scan].attrs['subtlety'] = abnormality.subtlety
            zarr_subj[scan].attrs['lesion_type'] = abnormality.lesion_type


def convert_scans(raw_dir, zarr_subj, files, metadata, tempdir='/tmp'):
    """
    file conversion pipeline:
      - ljpeg --> RAW: jpeg -d -s {ljpeg_file}" (appends .1 to filename)
      - RAW --> pnm: ddsmraw2pnm
      - pnm --> png: convert -depth 16 {pnm_file} {png_file}
      - png --> numpy array stored in zarr dataset

    NB: in line with https://arxiv.org/pdf/1703.07047.pdf
        flip all RIGHT images along horizontal axis so both right and left scans
        have the same view.
    """
    working_dir_exists = False

    for scan in list(metadata.scans.keys()):

        # obtain the input data (propriatary .LJPEG format)
        ljpeg = list(filter(lambda x: x.endswith(scan + '.LJPEG'), files))
        assert only_one(ljpeg)
        fstem = os.path.splitext(ljpeg[0])[0]

        # skip work if we already have the output file
        if scan in list(zarr_subj.array_keys()):
            continue

        # initalize metadata for subject
        zarr_subj.attrs['age'] = metadata.age
        zarr_subj.attrs['site'] = metadata.site
        zarr_subj.attrs['density'] = metadata.density
        zarr_subj.attrs['pathology'] = "HEALTHY" # overwritten if Abnormality

        if not working_dir_exists:
            working_dir = tempfile.mkdtemp(dir=tempdir)
            logger.debug('generating working directory {}'.format(working_dir))
            working_dir_exists = True

        # have to copy input file because conversion tool does not respond to
        # output path specification
        ljpeg_raw = os.path.join(raw_dir, ljpeg[0])
        ljpeg_tmp = os.path.join(working_dir, ljpeg[0])
        raw = ljpeg_tmp + '.1'
        pnm = raw + '-ddsmraw2pnm.pnm'
        png_file = os.path.join(working_dir, fstem + '.png')

        copyfile(ljpeg_raw, ljpeg_tmp)

        run('jpeg -d -s {}'.format(ljpeg_tmp)) # outputs a file with .1 appended
        run('ddsmraw2pnm {filename} {n_rows} {n_cols} {digitizer}'.format(
            filename=raw,
            n_rows=metadata.scans[scan]['rows'],
            n_cols=metadata.scans[scan]['cols'],
            digitizer=metadata.digitizer))
        run('convert -depth 16 {} {}'.format(pnm, png_file))
        image = np.vstack(png.Reader(png_file).read()[2])

        # all images are flipped to match the orientations of the LEFT view
        if 'RIGHT' in scan:
            image = np.fliplr(image)

        # 3rd axis is to allow us to append segmentations
        rows, cols = image.shape
        zarr_subj.create_dataset(scan, shape=(rows, cols, 1), mode='w', dtype=image.dtype)
        zarr_subj[scan][:, :, 0] = image
        zarr_subj[scan].attrs['rows'] = rows
        zarr_subj[scan].attrs['cols'] = cols

    if working_dir_exists:
        rmtree(working_dir)


def main(raw_dir, output, tempdir='/tmp'):
    """
    Runs metadata extraction, image conversion to 16-bit .png, abnormality
    extraction, and finally segmentation rendering, in order.

    Ignores all files ending with .1 and ~, as these are unclean files that
    technically should not be in the DDSM database.
    """
    logger.debug('starting work on folder {}'.format(raw_dir))
    files = os.listdir(raw_dir)
    files = list(filter(lambda x: '~' not in x, files)) # edge case
    ics_file = list(filter(lambda x: '.ics' in x, files))

    try:
        assert only_one(ics_file)
    except:
        logger.error('more than one .ics file found in {}, please inspect'.format(raw_dir))
        return

    # extracts all metadata from the .ics file
    metadata = Metadata(os.path.join(raw_dir, ics_file[0]))

    # add subject to dataset if it does not exist, otherwise point to it
    subj = output.require_group(metadata.name)

    # converts .LJPEG format to a compressed 16-bit .png
    convert_scans(raw_dir, subj, files, metadata, tempdir=tempdir)

    # generates segmentations from .OVERLAY files (saved as 2-bit .png)
    generate_segmentations(raw_dir, subj, files, metadata, tempdir=tempdir)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('input', help='DDSM subject folder (for an individual), or text tile for batch mode')
    argparser.add_argument('output', help='output zarr dataset')
    argparser.add_argument('-t', '--tempdir', help='specify a custom working directory')
    argparser.add_argument('-v', '--verbose', action='count', help='turns on debug messages')
    argparser.add_argument('-l', '--log', help='specify log file')
    args = argparser.parse_args()

    # determine if input is subject folder or batch file
    if os.path.isfile(args.input):
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

    # initialize output, create if does not exist
    output = zarr.open_group(args.output, mode='a')

    # batch mode or single mode
    if batch_mode:
        f = open(args.input, 'r')
        input_dirs = f.readlines()

        for input_dir in input_dirs:
            main(input_dir.strip(), output, tempdir)
    else:
        main(args.input, output, tempdir)


