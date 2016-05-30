import sys
if sys.version_info < (3,):
    try:
        from builtins import (bytes, dict, int, list, object, range, str,
                              ascii, chr, hex, input, next, oct, open,
                              pow, round, super, filter, map, zip)
        from future.builtins.disabled import (apply, cmp, coerce, execfile,
                                              file, long, raw_input,
                                              reduce, reload,
                                              unicode, xrange, StandardError)
    except:
        print("need future module (try 'pip install future')")
#
from math import *
#
import time
import profile
#
import numpy as np
import scipy
#
import matplotlib as mpl
import pylab as plt
#
import matplotlib.backends
from matplotlib.backends.backend_pgf import FigureCanvasPgf
#
mpl.rcParams['xtick.labelsize']=8
mpl.rcParams['ytick.labelsize']=8
mpl.rcParams['xtick.major.pad']=5 # xticks too close to border!
#
mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
#
pgf_with_custom_preamble = {
    "font.family": "serif", # use serif/main font for text elements
 #   "text.usetex": True,    # use inline math for ticks
#    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    "pgf.preamble": ["\\PassOptionsToPackage{cmyk}{xcolor}"]
}
#
mpl.rcParams.update(pgf_with_custom_preamble)
#
import imp
