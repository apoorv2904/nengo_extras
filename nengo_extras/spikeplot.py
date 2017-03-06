import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_spikes(t, spikes, contrast_scale=1.0, ax=None, **kwargs):
    t = np.asarray(t)
    spikes = np.asarray(spikes)
    if ax is None:
        ax = plt.gca()

    kwargs.setdefault('aspect', 'auto')
    kwargs.setdefault('cmap', matplotlib.cm.gray_r)
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('extent', (t[0], t[-1], 0., spikes.shape[1]))

    spikeraster = ax.imshow(spikes.T, **kwargs)
    spikeraster.set_clim(0., np.max(spikes) * contrast_scale)
    return spikeraster
