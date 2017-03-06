import nengo
import numpy as np
import pytest

from nengo_extras.spikeplot import plot_spikes


@pytest.mark.noassertions
def test_plot_spikes(plt, seed, RefSimulator):
    with nengo.Network(seed) as model:
        ens = nengo.Ensemble(10, 1)
        inp = nengo.Node(np.sin)
        nengo.Connection(inp, ens)
        p = nengo.Probe(ens.neurons, 'spikes')

    with nengo.Simulator(model) as sim:
        sim.run(1.)

    plot_spikes(sim.trange(), sim.data[p], ax=plt.gca())
