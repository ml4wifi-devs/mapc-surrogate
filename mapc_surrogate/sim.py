import os
import json
from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from mapc_research.envs.scenario_impl import *
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from mapc_surrogate.dataset import Configuration, TxPair, random_tx
from mapc_surrogate.graphs import conf_to_nx, nx_to_jraph, make_batch
from mapc_surrogate.model import SurrogateModel


TEST_SCENARIOS = [
    random_scenario(seed=100, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000, channel_width=80),
    random_scenario(seed=101, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000, channel_width=80),
    random_scenario(seed=102, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=103, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000, channel_width=80),
    random_scenario(seed=104, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=105, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000, channel_width=80),
    random_scenario(seed=106, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000, channel_width=80),
    random_scenario(seed=107, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000, channel_width=80),
    random_scenario(seed=108, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=109, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000, channel_width=80),
    random_scenario(seed=110, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=111, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000, channel_width=80),
    random_scenario(seed=112, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000, channel_width=80),
    random_scenario(seed=113, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000, channel_width=80),
    random_scenario(seed=114, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=115, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000, channel_width=80),
    random_scenario(seed=116, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=117, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000, channel_width=80),
    random_scenario(seed=118, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000, channel_width=80),
    random_scenario(seed=119, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000, channel_width=80),
    random_scenario(seed=120, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000, channel_width=80),
    random_scenario(seed=121, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000, channel_width=80),
    random_scenario(seed=122, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=2000, channel_width=80),
    random_scenario(seed=123, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000, channel_width=80),
]


def tx_to_conf(tx, tx_power, mcs, internals=None):
    ap, sta = np.where(tx)
    mcs_list = [mcs[a].item() for a in ap]
    tx_power_list = [tx_power[a].item() for a in ap]

    if internals is not None:
        succ_prob = (internals.frames_transmitted / np.maximum(internals.ampdu_size, 1))
        succ_prob_list = [succ_prob[a].item() for a in ap]
    else:
        succ_prob_list = [0.0 for _ in ap]

    return Configuration([TxPair(a, s, m, t, p) for a, s, m, t, p in zip(ap, sta, mcs_list, tx_power_list, succ_prob_list)])


def tx_to_action(associations, internals, tx, tx_power, mcs):
    return {
        ap: (sta, internals.average_data_rate[ap].item() / 1e6, tx_power[ap].item(), mcs[ap].item())
        for ap in associations.keys() for sta in associations[ap] if tx[ap, sta]
    }


def select_best_configuration(logits, means, scales, risk_averse=True, risk_factor=2.5):
    probs = jax.nn.softmax(logits, axis=-1)
    expected_value = jnp.sum(probs * means, axis=-1)

    if not risk_averse:
        scores = expected_value
    else:
        second_moment = (probs * (scales ** 2 + means ** 2)).sum(axis=-1)
        variance = second_moment - expected_value ** 2
        std_dev = jnp.sqrt(jnp.maximum(variance, 0.0))
        scores = expected_value - risk_factor * std_dev

    scores = scores[:-1]
    best_idx = jnp.argmax(scores)
    return best_idx, scores


def select_top_k(scores, top_k):
    sorted_indices = np.argsort(-scores)[:top_k]
    return [(int(idx), float(scores[idx])) for idx in sorted_indices]


def select_cover_stations(scores, candidate_txs, associations):
    """Select configs by score until all stations are covered, skipping configs that add no new stations."""
    all_stas = {int(sta) for stas in associations.values() for sta in stas}
    sorted_indices = np.argsort(-scores)
    covered_stas = set()
    selected = []

    for idx in sorted_indices:
        idx = int(idx)
        tx_matrix = candidate_txs[idx][0]
        _, cfg_stas = np.where(tx_matrix)
        new_stas = set(cfg_stas.tolist()) - covered_stas

        if new_stas:
            selected.append((idx, float(scores[idx])))
            covered_stas |= new_stas

        if covered_stas >= all_stas:
            break

    return selected


def run_scenario(
        scenario, n_steps, n_samples_eval, seed, ideal_mcs,
        surrogate_fn, batch_size, use_simulator, random_baseline, top_k, n_eval_repeats,
        selection='top_k'
):
    key = jax.random.PRNGKey(seed)
    all_results = []

    for step in trange(n_steps, desc='Steps', leave=False):
        # Draw N random configurations and build graphs
        candidate_txs = []
        candidate_graphs = []

        for _ in range(n_samples_eval):
            key, tx_key, sim_key = jax.random.split(key, 3)
            tx, tx_power, mcs = random_tx(tx_key, scenario)
            candidate_txs.append((tx, tx_power, mcs))

            if not random_baseline:
                conf = tx_to_conf(tx, tx_power, mcs)
                G = conf_to_nx(scenario, conf)
                graph = nx_to_jraph(G)
                candidate_graphs.append(graph)

        # Score candidates
        if random_baseline:
            key, perm_key = jax.random.split(key)
            random_indices = jax.random.permutation(perm_key, n_samples_eval)[:top_k]
            selected = [(int(idx), 0.0) for idx in random_indices]
        else:
            all_scores = []

            if use_simulator:
                for tx in candidate_txs:
                    key, sim_key = jax.random.split(key)
                    data_rate, _, _ = scenario(sim_key, *tx, return_internals=True)
                    all_scores.append(data_rate.item())
            else:
                for i in range(0, len(candidate_graphs), batch_size):
                    batch_graphs = candidate_graphs[i:i + batch_size]
                    batch = make_batch(batch_graphs)
                    logits, means, scales = surrogate_fn(batch)
                    _, scores = select_best_configuration(logits, means, scales)
                    all_scores.extend(scores.tolist())

            scores_array = np.asarray(all_scores[:n_samples_eval], dtype=np.float64)
            if selection == 'cover':
                selected = select_cover_stations(scores_array, candidate_txs, scenario.associations)
            else:
                selected = select_top_k(scores_array, top_k)
        results = []

        for idx, score in selected:
            eval_tx = candidate_txs[idx]
            repeats = []

            for _ in range(n_eval_repeats):
                key, scenario_key = jax.random.split(key)

                if ideal_mcs:
                    tx_mat, tx_power, _ = eval_tx
                    data_rate, _, internals = scenario(scenario_key, tx_mat, tx_power, return_internals=True)
                    eval_tx = (tx_mat, tx_power, internals.mcs)
                else:
                    data_rate, _, internals = scenario(scenario_key, *eval_tx, return_internals=True)

                repeats.append({
                    'data_rate': data_rate.item(),
                    'action': tx_to_action(scenario.associations, internals, *eval_tx)
                })

            results.append({
                'score': score,
                'runs': repeats
            })

        all_results.append({'configs': results})

    return all_results


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--surrogate', type=str, default='runs/surrogate_base')
    args.add_argument('--n_samples_eval', type=int, default=128)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--ideal_mcs', default=False, action='store_true')
    args.add_argument('--use_simulator', default=False, action='store_true')
    args.add_argument('--random_baseline', default=False, action='store_true')
    args.add_argument('--output', type=str, default='sim_results.json')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--top_k', type=int, default=1)
    args.add_argument('--n_eval_repeats', type=int, default=5)
    args.add_argument('--n_steps', type=int, default=32)
    args.add_argument('--selection', type=str, default='top_k', choices=['top_k', 'cover'])
    args = args.parse_args()

    surrogate_fn = None

    if not args.use_simulator and not args.random_baseline:
        if args.surrogate is None:
            raise ValueError('--surrogate is required when not using --use_simulator')

        with ocp.CheckpointManager(os.path.abspath(args.surrogate), item_names=['params']) as ckpt_manager:
            fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
            restore = ocp.args.Composite(params=ocp.args.StandardRestore(fallback_sharding=fallback_sharding))
            s_params = ckpt_manager.restore(None, args=restore).params

        cfg = OmegaConf.load(os.path.join(args.surrogate, 'config.yaml'))
        model = SurrogateModel(**cfg.model)
        surrogate_fn = jax.jit(partial(model.apply, s_params, training=False))

    all_results = []

    for scenario in tqdm(TEST_SCENARIOS, desc='Scenarios'):
        split_results = []

        for split_scenario, _ in scenario.split_scenario():
            split_results.append([run_scenario(
                split_scenario, args.n_steps, args.n_samples_eval,
                args.seed, args.ideal_mcs, surrogate_fn, args.batch_size,
                args.use_simulator, args.random_baseline, args.top_k, args.n_eval_repeats,
                selection=args.selection
            )])

        all_results.append({'splits': split_results})

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
