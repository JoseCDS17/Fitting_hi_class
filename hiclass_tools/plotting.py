"""Minimal plotting helpers (style and colour scales). The figures themselves live in the notebooks."""
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def use_style(fontsize=12, usetex=True):
    matplotlib.rcParams['text.usetex'] = usetex
    matplotlib.rcParams['font.family'] = 'CM Roman'
    matplotlib.rcParams['font.size'] = fontsize


def truncated_cmap(name, lo=0.2, hi=0.7):
    return mcolors.LinearSegmentedColormap.from_list('trunc', plt.get_cmap(name)(np.linspace(lo, hi, 256)))


class ColorScale:
    """Maps a parameter value to a colour: scale(value). `scale.mappable` feeds plt.colorbar."""

    def __init__(self, values, cmap='Reds', log=False):
        vmin, vmax = np.min(values), np.max(values)
        self.norm = mcolors.LogNorm(vmin, vmax) if log else mcolors.Normalize(vmin, vmax)
        self.cmap = truncated_cmap(cmap) if isinstance(cmap, str) else cmap

    def __call__(self, value):
        return self.cmap(self.norm(value))

    @property
    def mappable(self):
        sm = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        sm.set_array([])
        return sm
