from itertools import combinations

import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as np
from mapc_sim.constants import DEFAULT_TX_POWER

from mapc_surrogate.attributes import *


RATE_MEAN = 657.2
RATE_STD = 462.2

CCA_THRESHOLD = -82.0
RSSI_MEAN = -51.6
RSSI_STD = 14.3
RSSI_THR = (CCA_THRESHOLD - RSSI_MEAN) / RSSI_STD


def conf_to_nx(scenario, configuration):
    G = nx.Graph()

    distance = np.sqrt(np.sum((scenario.pos[:, None, :] - scenario.pos[None, ...]) ** 2, axis=-1))
    path_loss = scenario.path_loss_fn(distance, scenario.walls)

    for ap, stas in scenario.associations.items():
        for sta in stas:
            if not isinstance(sta, int):
                sta = sta.item()

            rssi = (DEFAULT_TX_POWER - path_loss[ap, sta] - RSSI_MEAN) / RSSI_STD
            G.add_edge(ap, sta, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.FALSE, tx_power=TxPowerValue.NA)

    for link in configuration.links:
        G.edges[(link.ap, link.sta)]['mcs'] = McsValue(link.mcs)
        G.edges[(link.ap, link.sta)]['selected'] = IsSelected.TRUE
        G.edges[(link.ap, link.sta)]['tx_power'] = TxPowerValue(link.tx_power)

    for ap_1, ap_2 in combinations(scenario.associations.keys(), r=2):
        rssi = (DEFAULT_TX_POWER - path_loss[ap_1, ap_2] - RSSI_MEAN) / RSSI_STD
        if rssi > RSSI_THR:
            G.add_edge(ap_1, ap_2, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, tx_power=TxPowerValue.NA)

    return G


def nearest_power_of_two(n):
    if not isinstance(n, int):
        n = n.item()
    return 2 ** (n - 1).bit_length()


def make_batch(graph_list, pad=True):
    batch = jraph.batch_np(graph_list)

    if pad:
        pad_n_node = nearest_power_of_two(batch.n_node.sum() + 1)
        pad_n_edge = nearest_power_of_two(batch.senders.shape[0])
        pad_n_graph = batch.n_node.shape[0] + 1
        batch = jraph.pad_with_graphs(batch, pad_n_node, pad_n_edge, pad_n_graph)

    return batch


def unbatch(batched_graph):
    graphs = jraph.unpad_with_graphs(batched_graph)
    return jraph.unbatch(graphs)


def nx_to_jraph(G):
    edges, senders, receivers = zip(*[[jax.tree.map(np.atleast_1d, Connection(**data)), u, v] for u, v, data in G.edges(data=True)])
    senders, receivers = senders + receivers, receivers + senders
    edges += edges

    return jraph.GraphsTuple(
        nodes=jnp.empty((len(G.nodes), 0), dtype=jnp.float32),
        edges=jax.tree.map(lambda *args: jnp.asarray(np.stack(args, dtype=np.float32)), *edges),
        receivers=jnp.asarray(receivers),
        senders=jnp.asarray(senders),
        globals=None,
        n_node=jnp.asarray([len(G.nodes)]),
        n_edge=jnp.asarray([len(edges)])
    )


def clear_graph(x):
    return x._replace(nodes=jnp.empty((x.nodes.shape[0], 0), dtype=x.nodes.dtype), globals=None)
