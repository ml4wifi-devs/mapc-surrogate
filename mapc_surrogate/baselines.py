import json
import os
from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import numpy as np
import simpy
from joblib import Parallel, delayed
from mapc_dcf.channel import Channel
from mapc_dcf.constants import TAU, DEFAULT_TX_POWER
from mapc_dcf.logger import Logger
from mapc_dcf.nodes import AccessPoint
from mapc_mab import MapcAgentFactory
from mapc_research.envs.scenario_impl import *
from reinforced_lib.agents.mab import UCB
from tqdm import tqdm, trange

from mapc_surrogate.sim import SCENARIO_SETS, tx_to_action


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
    channel = Channel(key_channel, False, scenario.channel_width, scenario.pos, scenario.walls)
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
        Parallel(n_jobs=n_runs)(
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


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--output', type=str, default='baseline_results.json')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--n_reps', type=int, default=5)
    args.add_argument('--agent', type=str, default='h_mab', choices=['h_mab', 'dcf'])
    args.add_argument('--scenario_set', type=str, default='sweep', choices=list(SCENARIO_SETS.keys()))
    args = args.parse_args()

    scenarios = SCENARIO_SETS[args.scenario_set]

    if args.agent == 'h_mab':
        all_results = []
        for scenario in tqdm(scenarios, desc='H-MAB Scenarios'):
            all_results.append(run_h_mab(
                scenario, scenario.n_steps,
                seed=args.seed, n_reps=args.n_reps
            ))
        with open(args.output, 'w') as file:
            json.dump(to_serializable(all_results), file)

    elif args.agent == 'dcf':
        output_dir = os.path.dirname(args.output) or '.'
        dcf_dir = os.path.join(output_dir, 'dcf_results')
        all_results = run_dcf(scenarios, args.seed, args.n_reps, 0.1, dcf_dir)
        with open(args.output, 'w') as file:
            json.dump(all_results, file)
