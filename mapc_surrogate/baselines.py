import json
import os
from argparse import ArgumentParser
from itertools import chain
from math import erf, sqrt

import jax
import jax.numpy as jnp
import numpy as np
import pulp as plp
import simpy
from joblib import Parallel, delayed
from mapc_dcf.channel import Channel
from mapc_dcf.constants import TAU, DEFAULT_TX_POWER
from mapc_dcf.logger import Logger
from mapc_dcf.nodes import AccessPoint
from mapc_mab import MapcAgentFactory
from mapc_optimal import OptimizationType, Solver, positions_to_path_loss
from mapc_research.envs.scenario_impl import *
from mapc_sim.constants import DEFAULT_NAKAGAMI_M, DEFAULT_NAKAGAMI_SIGMA, \
    NOISE_FLOOR, REFERENCE_DISTANCE, DATA_RATES, MEAN_SNRS, STD_SNR
from mapc_sim.utils import nakagami_fading_db
from reinforced_lib.agents.mab import UCB
from tqdm import tqdm, trange

from mapc_surrogate.sim import SCENARIO_SETS, eval_candidate, tx_to_action

NOISE_LIN = 10 ** (NOISE_FLOOR / 10)


def to_python_dict(d):
    result = {}
    for k, v in d.items():
        if hasattr(k, 'item'):
            k = k.item()
        result[k] = np.asarray(v).tolist()
    return result


def to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj


def run_h_mab_once(scenario, n_steps, key):
    agent = MapcAgentFactory(
        associations=to_python_dict(scenario.associations),
        agent_type=UCB,
        agent_params_lvl1={'c': 1.5, 'gamma': 0.5},
        agent_params_lvl2={'c': 0.5, 'gamma': 0.5},
        agent_params_lvl3={'c': 0.2, 'gamma': 0.8},
        hierarchical=True,
        seed=int(key[0])
    ).create_mapc_agent()

    reward = 0.0
    steps = []

    for _ in trange(n_steps, desc='H-MAB steps', leave=False):
        key, scenario_key = jax.random.split(key)
        tx, tx_power = agent.sample(reward)
        data_rate, reward, internals = scenario(scenario_key, tx, tx_power, return_internals=True)
        steps.append({
            'data_rate': data_rate.item(),
            'action': tx_to_action(scenario.associations, internals, tx, tx_power, internals.mcs)
        })

    return steps


def run_h_mab(scenario, n_steps, seed=42, n_reps=5):
    key = jax.random.PRNGKey(seed)
    reps = []

    for _ in range(n_reps):
        key, rep_key = jax.random.split(key)
        steps = run_h_mab_once(scenario, n_steps, rep_key)
        reps.append(steps)

    all_results = []

    for step_idx in range(n_steps):
        runs = [reps[rep][step_idx] for rep in range(n_reps)]
        all_results.append({'configs': [{'runs': runs}]})

    return all_results


def flatten_scenarios(scenarios):
    scenarios_flattened = []
    for scenario in scenarios:
        str_repr = scenario.__str__()
        list_of_scenarios = scenario.split_scenario()
        for i, s in enumerate(list_of_scenarios):
            suffix = f"_{chr(ord('a') + i)}" if len(list_of_scenarios) > 1 else ""
            scenarios_flattened.append((s[0], s[1], f"{str_repr}{suffix}"))
    return scenarios_flattened


def run_dcf_single(key, run, scenario, sim_time, logger):
    key, key_channel = jax.random.split(key)
    des_env = simpy.Environment()
    channel = Channel(key_channel, False, scenario.channel_width, scenario.pos, scenario.walls, nakagami_m=DEFAULT_NAKAGAMI_M, sigma=DEFAULT_NAKAGAMI_SIGMA)
    aps: dict[int, AccessPoint] = {}

    for ap in scenario.associations:
        key, key_ap = jax.random.split(key)
        clients = jnp.array(scenario.associations[ap])
        aps[ap] = AccessPoint(key_ap, ap, scenario.pos, DEFAULT_TX_POWER, clients, channel, des_env, logger)
        aps[ap].start_operation(run)

    des_env.run(until=(logger.warmup_length + sim_time))
    logger.dump_acumulators(run)
    del des_env


def run_dcf(scenarios, seed, n_runs, warmup, output_dir):
    scenarios = flatten_scenarios(scenarios)
    key = jax.random.PRNGKey(seed)
    sim_time = scenarios[0][0].n_steps * TAU

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for scenario, _, scenario_name in tqdm(scenarios, desc='DCF Scenarios'):
        results_path = os.path.join(output_dir, scenario_name)
        logger = Logger(sim_time, warmup, results_path)
        Parallel(n_jobs=min(n_runs, 16))(
            delayed(run_dcf_single)(k, r, scenario, sim_time, logger)
            for k, r in zip(jax.random.split(key, n_runs), range(1, n_runs + 1))
        )
        logger.shutdown({
            "name": scenario_name,
            "simulation_length": sim_time,
            "warmup_length": warmup,
            "n_runs": n_runs
        })

        with open(results_path + '.json') as f:
            results = json.load(f)
        all_results.append(results['DataRate']['Data'])

    return all_results


def estimate_mcs(sinr_db, channel_width):
    rates = DATA_RATES[channel_width]
    mus = MEAN_SNRS[channel_width]
    exp = [float(rates[i]) * 0.5 * (1 + erf((sinr_db - float(mus[i])) / (STD_SNR * sqrt(2))))
           for i in range(len(rates))]
    return int(max(range(len(exp)), key=lambda i: exp[i]))


def estimated_sinr(a, s, group, rssi):
    interf = NOISE_LIN + sum(10 ** (rssi[j, s] / 10) for j in group if j != a)
    return rssi[a, s] - 10 * np.log10(interf)


def rssi_matrix(scenario):
    pos = np.asarray(scenario.pos)
    walls = np.asarray(scenario.walls)
    tx_power = np.asarray(scenario.tx_power)
    d = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    d = np.clip(d, REFERENCE_DISTANCE, None)
    pl = np.asarray(scenario.path_loss_fn(d, walls))
    return tx_power[:, None] - pl


def group_min_sinr(group, assoc, rssi):
    worst = np.inf
    for a in group:
        for s in assoc[a]:
            interf = NOISE_LIN + sum(10 ** (rssi[j, s] / 10) for j in group if j != a)
            sinr = rssi[a, s] - 10 * np.log10(interf)
            worst = min(worst, sinr)
    return worst


def form_groups(scenario, rssi, gamma, K):
    assoc = {int(k): [int(s) for s in v] for k, v in scenario.associations.items()}
    aps = list(assoc.keys())
    groups = []

    for ref in aps:
        group = [ref]
        cand = sorted((j for j in aps if j != ref), key=lambda j: max(rssi[j, s] for s in assoc[ref]))
        for j in cand:
            if len(group) >= K:
                break
            if group_min_sinr(group + [j], assoc, rssi) >= gamma:
                group.append(j)
        groups.append(tuple(sorted(group)))

    uniq = []
    for g in groups:
        if g not in uniq:
            uniq.append(g)
    return uniq, assoc


def rank_groups(groups, assoc, rssi):
    return sorted(groups, key=lambda g: (len(g), group_min_sinr(g, assoc, rssi)), reverse=True)


def run_sr_groups(scenario, n_steps, seed, gamma, K, top_k, ideal_mcs=True):
    n_nodes = int(np.asarray(scenario.pos).shape[0])
    key = jax.random.PRNGKey(seed)
    steps = []

    rssi = rssi_matrix(scenario)
    key, nakagami_key = jax.random.split(key)
    rssi = rssi + nakagami_fading_db(nakagami_key, 1.5, rssi.shape)

    for _ in range(n_steps):
        configs = []

        groups, assoc = form_groups(scenario, rssi, gamma, K)
        ranked = rank_groups(groups, assoc, rssi)
        selected = [ranked[i % len(ranked)] for i in range(top_k)]

        for group in selected:
            tx = np.zeros((n_nodes, n_nodes), dtype=int)
            tx_power = np.zeros(n_nodes, dtype=int)
            mcs = np.zeros(n_nodes, dtype=int)

            for a in group:
                key, sta_key = jax.random.split(key)
                stas = assoc[a]
                s = stas[jax.random.choice(sta_key, len(stas)).item()]
                tx[a, s] = 1
                mcs[a] = estimate_mcs(estimated_sinr(a, s, group, rssi), scenario.channel_width)

            key, eval_key = jax.random.split(key)
            data_rate, internals, eval_tx = eval_candidate(scenario, eval_key, (tx, tx_power, mcs), ideal_mcs)
            configs.append({
                'score': 0.0,
                'data_rate': data_rate,
                'action': tx_to_action(scenario.associations, internals, *eval_tx),
            })
            rssi = internals.signal_power  # update for next step (fading is time-correlated)

        steps.append({'configs': configs})

    return steps


def run_optimal(scenario, channel_width):
    associations = scenario.get_associations()
    access_points = list(associations.keys())
    stations = list(chain.from_iterable(associations.values()))
    path_loss = positions_to_path_loss(scenario.pos, scenario.walls)

    results = {}
    
    for opt_type, name in [(OptimizationType.SUM, 't_optimal'), (OptimizationType.MAX_MIN, 'f_optimal')]:
        solver = Solver(
            stations, access_points, opt_type=opt_type, channel_width=channel_width,
            solver=plp.CPLEX_CMD(msg=False, threads=None),
        )
        _, total_rate = solver(path_loss, associations)
        results[name] = total_rate

    return results


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--output', type=str, default='baseline_results.json')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--n_reps', type=int, default=64)
    args.add_argument('--n_steps', type=int, default=64)
    args.add_argument('--gamma', type=float, default=20.0)
    args.add_argument('--K', type=int, default=3)
    args.add_argument('--top_k', type=int, default=1)
    args.add_argument('--channel_width', type=int, default=80)
    args.add_argument('--agent', type=str, default='h_mab', choices=['h_mab', 'dcf', 'sr_groups', 'optimal'])
    args.add_argument('--scenario_set', type=str, default='sweep', choices=list(SCENARIO_SETS.keys()))
    args = args.parse_args()

    scenarios = SCENARIO_SETS[args.scenario_set]

    if args.agent == 'h_mab':
        all_results = []
        for scenario in tqdm(scenarios, desc='H-MAB Scenarios'):
            all_results.append(run_h_mab(scenario, scenario.n_steps, args.seed, args.n_reps))
        with open(args.output, 'w') as file:
            json.dump(to_serializable(all_results), file)

    elif args.agent == 'dcf':
        output_dir = os.path.dirname(args.output) or '.'
        dcf_dir = os.path.join(output_dir, 'dcf_results')
        all_results = run_dcf(scenarios, args.seed, args.n_reps, 0.1, dcf_dir)
        with open(args.output, 'w') as file:
            json.dump(all_results, file)

    elif args.agent == 'sr_groups':
        all_results = []
        for scenario in tqdm(scenarios, desc='SR-groups Scenarios'):
            split_results = []
            for split_scenario, _ in scenario.split_scenario():
                steps = run_sr_groups(split_scenario, args.n_steps, args.seed, args.gamma, args.K, args.top_k)
                split_results.append([steps])
            all_results.append({'splits': split_results})
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as file:
            json.dump(all_results, file)

    elif args.agent == 'optimal':
        all_results = []
        for scenario in tqdm(scenarios, desc='Optimal Scenarios'):
            results = run_optimal(scenario, args.channel_width)
            all_results.append(results)
        with open(args.output, 'w') as file:
            json.dump(all_results, file)
