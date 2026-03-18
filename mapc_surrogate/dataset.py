import os
from dataclasses import dataclass

import cloudpickle
import jax
import lz4.frame
import numpy as np
import jax.numpy as jnp
from mapc_sim.constants import DEFAULT_TX_POWER, DATA_RATES
from mapc_research.envs.scenario import Scenario
from mapc_research.envs.scenario_impl import *
from tqdm import tqdm

from mapc_surrogate.attributes import *
from mapc_surrogate.graphs import RATE_MEAN, RATE_STD, conf_to_nx, nx_to_jraph, make_batch


SCENARIOS = [
    (
        toy_scenario_1,
        {'d': (10, 31)}
    ),
    (
        toy_scenario_2,
        {'d_ap': (10, 31), 'd_sta': (1, 11)}
    ),
    (
        small_office_scenario,
        {'d_ap': (10, 21), 'd_sta': (1, 11)}
    ),
    (
        random_scenario,
        {'d_ap': (20, 101), 'n_ap': (2, 11), 'd_sta': (1, 9), 'n_sta_per_ap': (1, 6), 'randomize': (0, 1)}
    ),
    (
        residential_scenario,
        {'x_apartments': (2, 6), 'y_apartments': (2, 3), 'n_sta_per_ap': (1, 5), 'size': (5, 21)}
    ),
    (
        hidden_station_scenario,
        {'d': (21, 51)}
    ),
    (
        flow_in_the_middle_scenario,
        {'d': (1, 31)}
    ),
    (
        dense_point_scenario,
        {'n_ap': (2, 11), 'n_associations': (1, 6)}
    ),
    (
        spatial_reuse_scenario,
        {'d_ap': (10, 21), 'd_sta': (1, 11)}
    ),
    (
        test_scenario,
        {'scale': (10, 31)}
    ),
    (
        indoor_small_bsss_scenario,
        {'grid_layers': (3, 4), 'n_sta_per_ap': (3, 11), 'frequency_reuse': (3, 4), 'bss_radius': (5, 21)}
    ),
]


@dataclass
class TxPair:
    ap: int
    sta: int
    mcs: int
    tx_power: int
    success: float


@dataclass
class Configuration:
    links: list[TxPair]


@dataclass
class DatasetItem:
    scenario: Scenario
    configurations: list


N_TX_POWER_LEVELS = 4
TX_POWER_DELTA = 3.0
TX_POWER_LEVELS = np.array([DEFAULT_TX_POWER - i * TX_POWER_DELTA for i in range(N_TX_POWER_LEVELS)])
MCS_VALUES = DATA_RATES[80]


def rate_to_mcs(rate):
    return (np.abs(MCS_VALUES - rate)).argmin().item()


def tx_power_to_lvl(tx_power):
    return (np.abs(TX_POWER_LEVELS - tx_power)).argmin().item()


def save_dataset(dataset, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with lz4.frame.open(path, 'wb') as f:
        f.write(cloudpickle.dumps(dataset))


def load_dataset(path):
    with lz4.frame.open(path, 'rb') as f:
        return cloudpickle.loads(f.read())


def random_tx(key, scenario):
    n_aps = len(scenario.associations)
    n_nodes = n_aps + sum(len(stas) for stas in scenario.associations.values())
    ap_list = list(scenario.associations.keys())

    tx = np.zeros((n_nodes, n_nodes), dtype=int)
    tx_power = np.zeros(n_nodes, dtype=int)
    mcs = np.zeros(n_nodes, dtype=int)

    # Draw k ~ Uniform(1, n_aps), then pick k APs at random
    key, k_key, perm_key = jax.random.split(key, 3)
    k = jax.random.randint(k_key, (), 1, n_aps + 1).item()
    active_aps = jax.random.permutation(perm_key, n_aps)[:k]

    for ap_idx in active_aps:
        ap = ap_list[ap_idx.item()]
        stas = scenario.associations[ap]
        key, sta_key, power_key, mcs_key = jax.random.split(key, 4)

        sta_idx = jax.random.choice(sta_key, len(stas)).item()
        sta = stas[sta_idx]
        tx[ap, sta] = 1
        tx_power[ap] = jax.random.choice(power_key, len(TxPowerValue) - 1).item()
        mcs[ap] = jax.random.choice(mcs_key, len(McsValue) - 1).item()

    return tx, tx_power, mcs


def draw_realizations(key, scenario_fn, param_ranges):
    *param_keys, seed_key = jax.random.split(key, len(param_ranges) + 1)
    params = {p: jax.random.randint(k, (), *v) for (p, v), k in zip(param_ranges.items(), param_keys)}
    seed = jax.random.randint(seed_key, (), 0, 2**30)
    (scenario, _), *_ = scenario_fn(seed=seed, channel_width=80, **params).split_scenario()
    yield DatasetItem(scenario, configurations=[])


def draw_scenarios(n_realizations, key, scenarios):
    n_params = list(map(len, [p for _, p in scenarios]))
    n_params = np.asarray(n_params)
    probs = n_params / n_params.sum()

    key, subkey = jax.random.split(key)
    scenario_idx = jax.random.choice(subkey, len(scenarios), p=probs, shape=(n_realizations,)).tolist()
    selected_scenarios = [scenarios[i] for i in scenario_idx]

    for scenario, param_ranges in tqdm(selected_scenarios, desc='Scenarios'):
        key, subkey = jax.random.split(key)
        yield from draw_realizations(subkey, scenario, param_ranges)


def draw_configuration(n_configurations, key, dataset_item):
    for _ in range(n_configurations):
        key, random_key, scenario_key = jax.random.split(key, 3)
        tx, tx_power, mcs = random_tx(random_key, dataset_item.scenario)
        _, _, internals = dataset_item.scenario(scenario_key, tx, tx_power, mcs, return_internals=True)
        succ_prob = (internals.frames_transmitted / np.maximum(internals.ampdu_size, 1))
        ap, sta = np.where(tx)
        mcs_list = [mcs[a].item() for a in ap]
        tx_power_list = [tx_power[a].item() for a in ap]
        succ_prob_list = [succ_prob[a].item() for a in ap]
        yield Configuration([TxPair(a, s, m, t, p) for a, s, m, t, p in zip(ap, sta, mcs_list, tx_power_list, succ_prob_list)])


def draw_history(n_configurations, key, dataset):
    for dataset_item in tqdm(dataset, desc='Configurations'):
        key, subkey = jax.random.split(key)
        dataset_item.configurations = list(draw_configuration(n_configurations, subkey, dataset_item))

    return dataset


def calculate_data_rate(configuration):
    """Calculate data rate directly from a Configuration's TxPair links."""
    return sum(MCS_VALUES[link.mcs] * link.success for link in configuration.links)


def generate_dataset(seed, n_realizations, n_configurations, save_path, batch_size=32):
    key = jax.random.PRNGKey(seed)
    scenarios_key, configurations_key = jax.random.split(key)

    dataset = list(draw_scenarios(n_realizations, scenarios_key, SCENARIOS))
    dataset = draw_history(n_configurations, configurations_key, dataset)

    # Convert to (graph, rate) pairs
    graph_rate_pairs = []
    for item in tqdm(dataset, desc='Converting to graphs'):
        for conf in item.configurations:
            rate = calculate_data_rate(conf)
            rate = (rate - RATE_MEAN) / RATE_STD
            G = conf_to_nx(item.scenario, conf)
            graph = nx_to_jraph(G)
            graph_rate_pairs.append((graph, rate))

    # Batch the pairs
    batched_dataset = []
    for i in range(0, len(graph_rate_pairs), batch_size):
        batch_pairs = graph_rate_pairs[i:i + batch_size]
        graphs, rates = zip(*batch_pairs)
        batch = make_batch(list(graphs))
        rates_array = jnp.asarray(list(rates) + [0.0])  # Padding rate
        batched_dataset.append((batch, rates_array))

    save_dataset(batched_dataset, save_path)
    print(f'Saved {len(graph_rate_pairs)} samples in {len(batched_dataset)} batches to {save_path}')


if __name__ == '__main__':
    generate_dataset(
        seed=42,
        n_realizations=1000,
        n_configurations=30,
        save_path='datasets/random_dataset.pkl.lz4'
    )
    generate_dataset(
        seed=43,
        n_realizations=200,
        n_configurations=30,
        save_path='datasets/random_val_dataset.pkl.lz4'
    )
