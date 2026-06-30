import inspect
import os
from dataclasses import dataclass

import cloudpickle
import jax
import lz4.frame
import numpy as np
import jax.numpy as jnp
from mapc_sim.constants import DEFAULT_TX_POWER, DATA_RATES, DEFAULT_NAKAGAMI_M, DEFAULT_NAKAGAMI_SIGMA
from mapc_research.envs.scenario import Scenario
from mapc_research.envs.scenario_impl import *
from tqdm import tqdm

from mapc_surrogate.attributes import *
from mapc_surrogate.graphs import conf_to_nx, nx_to_jraph, make_batch


SCENARIOS = [
    (toy_scenario_1,              {'d': (10, 31)},                                                                                    0.5),
    (toy_scenario_2,              {'d_ap': (10, 31), 'd_sta': (1, 11)},                                                               1.0),
    (small_office_scenario,       {'d_ap': (10, 21), 'd_sta': (1, 11)},                                                               1.0),
    (random_scenario,             {'d_ap': (20, 101), 'n_ap': (2, 7), 'd_sta': (1, 9), 'n_sta_per_ap': (1, 6), 'randomize': (0, 1)},  2.0),
    (symm_residential_scenario,   {'x_apartments': (2, 6), 'y_apartments': (2, 3), 'n_sta_per_ap': (1, 5), 'size': (5, 21)},          2.0),
    (hidden_station_scenario,     {'d': (21, 51)},                                                                                    0.5),
    (flow_in_the_middle_scenario, {'d': (1, 31)},                                                                                     1.0),
    (dense_point_scenario,        {'n_ap': (2, 11), 'n_associations': (1, 6)},                                                        2.5),
    (spatial_reuse_scenario,      {'d_ap': (10, 21), 'd_sta': (1, 11)},                                                               0.5),
    (test_scenario,               {'scale': (10, 31)},                                                                                0.5),
    (indoor_small_bsss_scenario,  {'grid_layers': (3, 4), 'n_sta_per_ap': (3, 11), 'frequency_reuse': (3, 4), 'bss_radius': (5, 21)}, 2.0),
]


@dataclass
class TxPair:
    ap: int
    sta: int
    mcs: int
    tx_power: int


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


DEFAULT_SCENARIO_PARAMS = {
    'channel_width': 80,
    'n_steps': 2000,
    'sigma': DEFAULT_NAKAGAMI_SIGMA,
    'nakagami_m': DEFAULT_NAKAGAMI_M,
}


def draw_realizations(key, scenario_fn, param_ranges):
    *param_keys, seed_key = jax.random.split(key, len(param_ranges) + 1)
    params = {p: jax.random.randint(k, (), *v) for (p, v), k in zip(param_ranges.items(), param_keys)}
    seed = jax.random.randint(seed_key, (), 0, 2**30)
    sig = inspect.signature(scenario_fn).parameters
    call_kwargs = {**DEFAULT_SCENARIO_PARAMS, **params}
    if 'seed' in sig:
        call_kwargs['seed'] = seed
    (scenario, _), *_ = scenario_fn(**call_kwargs).split_scenario()
    yield DatasetItem(scenario, configurations=[])


def draw_scenarios(n_realizations, key, scenarios):
    weights = np.asarray([w for _, _, w in scenarios], dtype=float)
    probs = weights / weights.sum()

    key, subkey = jax.random.split(key)
    scenario_idx = jax.random.choice(subkey, len(scenarios), p=probs, shape=(n_realizations,)).tolist()
    selected_scenarios = [scenarios[i] for i in scenario_idx]

    for scenario, param_ranges, _ in tqdm(selected_scenarios, desc='Scenarios'):
        key, subkey = jax.random.split(key)
        yield from draw_realizations(subkey, scenario, param_ranges)


N_SIM_REPEATS = 3
N_RANDOM_CONFIGS = 40
N_IDEAL_MCS_CONFIGS = 10


def _avg_rate(key, scenario, tx, tx_power, mcs):
    rates = []
    for _ in range(N_SIM_REPEATS):
        key, scenario_key = jax.random.split(key)
        data_rate, _, _ = scenario(scenario_key, tx, tx_power, mcs, return_internals=True)
        rates.append(data_rate.item())
    return key, float(np.mean(rates))


def draw_configuration(key, dataset_item):
    scenario = dataset_item.scenario

    for _ in range(N_RANDOM_CONFIGS):
        key, random_key = jax.random.split(key)
        tx, tx_power, mcs = random_tx(random_key, scenario)
        ap, sta = np.where(tx)
        conf = Configuration([TxPair(a, s, mcs[a].item(), tx_power[a].item()) for a, s in zip(ap, sta)])
        key, rate = _avg_rate(key, scenario, tx, tx_power, mcs)
        yield conf, rate

    for _ in range(N_IDEAL_MCS_CONFIGS):
        key, random_key, ideal_key = jax.random.split(key, 3)
        tx, tx_power, _ = random_tx(random_key, scenario)
        _, _, internals = scenario(ideal_key, tx, tx_power, return_internals=True)
        mcs = internals.mcs
        ap, sta = np.where(tx)
        conf = Configuration([TxPair(a, s, mcs[a].item(), tx_power[a].item()) for a, s in zip(ap, sta)])
        key, rate = _avg_rate(key, scenario, tx, tx_power, mcs)
        yield conf, rate


def draw_history(key, dataset):
    for dataset_item in tqdm(dataset, desc='Configurations'):
        key, subkey = jax.random.split(key)
        dataset_item.configurations = list(draw_configuration(subkey, dataset_item))
    return dataset


def generate_dataset(seed, n_realizations, save_path, batch_size=16,
                     rate_mean=None, rate_std=None):
    key = jax.random.PRNGKey(seed)
    scenarios_key, configurations_key = jax.random.split(key)

    dataset = list(draw_scenarios(n_realizations, scenarios_key, SCENARIOS))
    dataset = draw_history(configurations_key, dataset)

    # Collect raw (graph, rate) pairs
    graph_rate_pairs = []
    for item in tqdm(dataset, desc='Converting to graphs'):
        for conf, rate in item.configurations:
            G = conf_to_nx(item.scenario, conf)
            graph = nx_to_jraph(G)
            graph_rate_pairs.append((graph, rate))

    # Compute normalization stats from data if not provided (train set)
    raw_rates = np.array([r for _, r in graph_rate_pairs])
    if rate_mean is None or rate_std is None:
        rate_mean = float(np.mean(raw_rates))
        rate_std = float(np.std(raw_rates))
    print(f'Rate stats: mean={rate_mean:.1f}, std={rate_std:.1f}')

    # Shuffle before batching so batches contain mixed scenarios
    rng = np.random.default_rng(seed)
    rng.shuffle(graph_rate_pairs)

    # Normalize and batch
    batched_dataset = []
    for i in range(0, len(graph_rate_pairs), batch_size):
        batch_pairs = graph_rate_pairs[i:i + batch_size]
        graphs, rates = zip(*batch_pairs)
        batch = make_batch(list(graphs))
        rates_norm = [(r - rate_mean) / rate_std for r in rates]
        rates_array = jnp.asarray(rates_norm + [0.0])
        batched_dataset.append((batch, rates_array))

    save_dataset(batched_dataset, save_path)
    print(f'Saved {len(graph_rate_pairs)} samples in {len(batched_dataset)} batches to {save_path}')
    return rate_mean, rate_std


if __name__ == '__main__':
    rate_mean, rate_std = generate_dataset(
        seed=42,
        n_realizations=1000,
        save_path='datasets/random_dataset.pkl.lz4'
    )
    generate_dataset(
        seed=43,
        n_realizations=200,
        save_path='datasets/random_val_dataset.pkl.lz4',
        rate_mean=rate_mean,
        rate_std=rate_std
    )
