import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'

import logging
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from scipy.stats import spearmanr
from omegaconf import OmegaConf

from mapc_surrogate.dataset import load_dataset
from mapc_surrogate.model import SurrogateModel


def cyclic_shuffled_batches(dataset, seed):
    if len(dataset) == 0:
        raise ValueError('Prebatched dataset is empty')
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    while True:
        rng.shuffle(indices)
        for idx in indices:
            yield dataset[idx]


@hydra.main(config_path='configs', config_name='surrogate', version_base=None)
def main(cfg):

    @jax.jit
    def mdn_loss(logits, means, scales, rates):
        log_probs_pi = jax.nn.log_softmax(logits, axis=-1)
        log_probs_gauss = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(scales) - 0.5 * ((rates[..., None] - means) / scales) ** 2
        log_probs_total = log_probs_pi + log_probs_gauss
        return -jax.nn.logsumexp(log_probs_total, axis=-1)

    @partial(jax.jit, static_argnames=('training',))
    def mdn_loss_fn(params, key, batch, rates, training=True):
        logits, means, scales = model.apply(params, batch, training=training, rngs=key)
        probs = jax.nn.softmax(logits, axis=-1)
        expected_value = jnp.sum(probs * means, axis=-1)

        mask = jraph.get_graph_padding_mask(batch)
        loss = mdn_loss(logits, means, scales, rates)
        loss = (mask * loss).sum() / mask.sum()

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        entropy = -jnp.sum(probs * log_probs, axis=-1)
        eff_components = jnp.exp((mask * entropy).sum() / mask.sum())

        second_moment = jnp.sum(probs * (scales ** 2 + means ** 2), axis=-1)
        variance = second_moment - expected_value ** 2
        std_dev = jnp.sqrt(jnp.maximum(variance, 0.0))

        return loss, (expected_value, std_dev, mask, eff_components)

    @jax.jit
    def step_fn(params, opt_state, key, batch, rates):
        (loss, _), grads = jax.value_and_grad(mdn_loss_fn, has_aux=True)(params, key, batch, rates)
        grad_norm = optax.global_norm(grads)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grad_norm

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key = jax.random.split(key, 3)

    data_list = load_dataset(cfg.train_dataset_path)
    val_data_list = load_dataset(cfg.val_dataset_path)
    data = cyclic_shuffled_batches(data_list, cfg.train.seed)
    val_data = cyclic_shuffled_batches(val_data_list, cfg.train.seed + 1)
    sample = data_list[0][0]

    model = SurrogateModel(**cfg.model)
    params = model.init({'params': init_key}, sample)
    print(model.tabulate(jax.random.PRNGKey(0), sample))

    lr = optax.cosine_onecycle_schedule(cfg.train.n_steps, cfg.optimizer.learning_rate, pct_start=0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimizer.grad_clip),
        optax.adamw(lr, weight_decay=cfg.optimizer.weight_decay, b1=cfg.optimizer.b1, b2=cfg.optimizer.b2)
    )
    opt_state = optimizer.init(params)

    ckpt_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_manager = ocp.CheckpointManager(
        directory=ckpt_path,
        options=ocp.CheckpointManagerOptions(max_to_keep=None if cfg.train.ckpt.intermediate else 1, create=True),
        item_names=['params', 'opt_state', 'train_key', 'val_key']
    )
    OmegaConf.save(config=cfg, f=os.path.join(ckpt_path, 'config.yaml'))

    if cfg.train.log.type == 'wandb':
        wandb.init(project='mapc-surrogate', name=cfg.name)

    for step in range(cfg.train.n_steps):
        batch, rates = next(data)
        log_dict = {'step': step}

        train_key, subkey = jax.random.split(train_key)
        params, opt_state, loss, grad_norm = step_fn(params, opt_state, subkey, batch, rates)

        if step % cfg.train.val.freq == 0 or step == cfg.train.n_steps - 1:
            val_loss = 0.0
            all_preds, all_targets, all_stds = [], [], []
            total_eff_components = 0.0

            for _ in range(cfg.train.val.n_steps):
                val_key, subkey = jax.random.split(val_key)
                val_batch, val_rates = next(val_data)
                v_loss, (v_pred, v_std, v_mask, v_eff) = mdn_loss_fn(params, subkey, val_batch, val_rates, training=False)
                val_loss += v_loss
                total_eff_components += v_eff
                all_preds.append(np.asarray(v_pred[v_mask]))
                all_targets.append(np.asarray(val_rates[v_mask]))
                all_stds.append(np.asarray(v_std[v_mask]))

            preds = np.concatenate(all_preds)
            targets = np.concatenate(all_targets)
            stds = np.concatenate(all_stds)
            errors = np.abs(targets - preds)

            mse = float(np.mean((preds - targets) ** 2))
            mae = float(np.mean(errors))
            ss_res = np.sum((targets - preds) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            spearman_r = float(spearmanr(preds, targets).statistic) if len(preds) > 1 else 0.0

            calib_1sigma = float(np.mean(errors < 1.0 * stds))
            calib_2sigma = float(np.mean(errors < 2.0 * stds))
            calib_3sigma = float(np.mean(errors < 3.0 * stds))

            log_dict['val/loss'] = val_loss / cfg.train.val.n_steps
            log_dict['val/mse'] = mse
            log_dict['val/mae'] = mae
            log_dict['val/r2'] = r2
            log_dict['val/spearman_r'] = spearman_r
            log_dict['val/eff_components'] = total_eff_components / cfg.train.val.n_steps
            log_dict['val/calib_1sigma'] = calib_1sigma
            log_dict['val/calib_2sigma'] = calib_2sigma
            log_dict['val/calib_3sigma'] = calib_3sigma

        if step % cfg.train.log.freq == 0 or step == cfg.train.n_steps - 1:
            log_dict['train/loss'] = loss
            log_dict['train/grad_norm'] = grad_norm
            logging.info(log_dict)
            if cfg.train.log.type == 'wandb':
                wandb.log(log_dict)

        if step % cfg.train.ckpt.freq == 0 or step == cfg.train.n_steps - 1:
            ckpt_manager.save(step, args=ocp.args.Composite(
                params=ocp.args.PyTreeSave(params),
                opt_state=ocp.args.PyTreeSave(opt_state),
                train_key=ocp.args.JaxRandomKeySave(train_key),
                val_key=ocp.args.JaxRandomKeySave(val_key)
            ))

    ckpt_manager.wait_until_finished()
    ckpt_manager.close()


if __name__ == '__main__':
    main()
