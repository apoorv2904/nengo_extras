from .version import version as __version__
from .rc import rc

# --- nengo_extras namespace (API)
from .convnet import Conv2d, Pool2d
from .neurons import FastLIF, SoftLIFRate
from . import data, dists, graphviz, gui, networks, neurons, probe, vision
