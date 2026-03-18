import json
from argparse import ArgumentParser

import jax
import numpy as np
from mapc_mab import MapcAgentFactory
from mapc_research.envs.scenario_impl import *
from reinforced_lib.agents.mab import UCB
from tqdm import tqdm, trange

from mapc_surrogate.sim import TEST_SCENARIOS, tx_to_action


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


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--output', type=str, default='h_mab_results.json')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--n_steps', type=int, default=32)
    args.add_argument('--n_reps', type=int, default=5)
    args = args.parse_args()

    all_results = []

    for scenario in tqdm(TEST_SCENARIOS, desc='Scenarios'):
        all_results.append(run_h_mab(
            scenario, args.n_steps,
            seed=args.seed, n_reps=args.n_reps
        ))

    with open(args.output, 'w') as file:
        json.dump(to_serializable(all_results), file)
