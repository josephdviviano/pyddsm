pyddsm
------

Pipeline to extract the data available from the [DDSM project](http://marathon.csee.usf.edu/Mammography/Database.html). Tested on Debian 9 (64 bit) with Python 3.6. Requires `imagemagick`, `ncftp`, `jpeg`, `ddsmraw2pnm`, and python packages in `requirements.txt`. Some of these utilities are available in `utils/`.

Documentation is thin, but comments in `ddsm_convert.py` should fill in the gaps.

**download.sh**

Requires the ncftp package to allow for anon ftp logins, compile from source [here](https://www.ncftp.com/download/).

**software**

+ The official software is [here](http://marathon.csee.usf.edu/Mammography/software/heathusf_v1.1.0.html) and is a compile nightmare from the year 2000.
+ Someone ported the decompression software `jpeg` [here](http://www.cs.unibo.it/~roffilli/sw.html), which builds.
+ This approach was heavily inspired by [previous work](https://github.com/multinormal/ddsm.git).

**metadata.csv**

Contains details about each subfolder, including which scanner each set was collected on and the number of scans in each set.

bwc = benign_without_callback

**demographics.csv**

Contains the distribution of ethicities collected per site.

MGH = Massachusetts General Hospital, largest proportion
WFUSM = Wake Forest University School of Medicine, 2nd largest proportion

**grey_to_optical.csv**

The grey values in each image can be mapped to optical density (OD) values to normalize each pixel's grey level (GL) values to have a standard scale. The equation used for each normalization are stored in this `.csv`. The DESCRIPTION column denotes the naming convention that can be used to map an individual scan to it's scanner, and therefore to it's appropriate conversion formula.

The included program `ddsmraw2pnm` takes care of this automatically, these notes are just for posterity.

