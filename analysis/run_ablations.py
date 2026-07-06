"""
Weekend retraining launcher for the ablation experiments (Q-A, Q-B) and their controls.

Runs each condition across several seeds, checkpointing every `checkpoint_every` steps and
recording the full learning trajectory. Designed to survive interruption: every network's
final model + history is written to its own directory under `outdir` as soon as it finishes,
so a crash loses at most the in-flight network.

Usage (from the repo dir, biornn env):
    python run_ablations.py --conditions drop_pv drop_dend shrink baseline --seeds 0 1 2 3 4 \
        --outdir ../retrain/ablations --max_steps 6000 --converge_perf 0.85

Then snapshot ../retrain/ablations to an artifact.
"""
import argparse, os, json, time
import numpy as np
import torch

import ablations as ab
from train_local import train_local, compressed_timing

CONDITIONS = {
    'baseline':  lambda: ab.hp_baseline(),
    'drop_pv':   lambda: ab.hp_drop_pv(),
    'drop_dend': lambda: ab.hp_drop_dendrites(5),
    'shrink':    lambda: ab.hp_shrink_only(5),
}


def run_one(cond, seed, outdir, max_steps, converge_perf, checkpoint_every):
    hp = CONDITIONS[cond]()
    hp['torch_seed'] = seed
    hp_task = compressed_timing()
    ndir = os.path.join(outdir, f'{cond}_seed{seed}')
    ckdir = os.path.join(ndir, 'ckpts')
    os.makedirs(ckdir, exist_ok=True)
    if os.path.exists(os.path.join(ndir, 'final.pt')):
        print(f'[skip] {cond} seed{seed} already done', flush=True)
        return
    t0 = time.time()
    torch.set_num_threads(12)
    model, hist = train_local(hp, hp_task, max_steps=max_steps, record_activity=False,
                              seed=seed, checkpoint_dir=ckdir, checkpoint_every=checkpoint_every,
                              converge_perf=converge_perf, log_every=200, verbose=True)
    np.savez(os.path.join(ndir, 'history.npz'),
             step=hist['step'], perf=hist['perf'], perf_rule=hist['perf_rule'],
             loss=hist['loss'], stage=hist['stage'])
    json.dump({'cond': cond, 'seed': seed, 'converged_step': hist.get('converged_step'),
               'final_perf': hist.get('final_perf'), 'final_perf_rule': hist.get('final_perf_rule'),
               'wall_time_s': hist.get('wall_time_s'), 'exploded': hist.get('exploded', False),
               'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)},
              open(os.path.join(ndir, 'meta.json'), 'w'))
    torch.save({'model_state_dict': model.state_dict(), 'hp': hp, 'hp_task': hp_task},
               os.path.join(ndir, 'final.pt'))
    print(f'[done] {cond} seed{seed}: converged_step={hist.get("converged_step")} '
          f'final_perf={hist.get("final_perf"):.3f} ({time.time()-t0:.0f}s)', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', nargs='+', default=['drop_pv', 'drop_dend', 'shrink', 'baseline'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--outdir', default='../retrain/ablations')
    ap.add_argument('--max_steps', type=int, default=6000)
    ap.add_argument('--converge_perf', type=float, default=0.85)
    ap.add_argument('--checkpoint_every', type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    grid = [(c, s) for c in args.conditions for s in args.seeds]
    print(f'launching {len(grid)} networks: {args.conditions} x seeds {args.seeds}', flush=True)
    for c, s in grid:
        try:
            run_one(c, s, args.outdir, args.max_steps, args.converge_perf, args.checkpoint_every)
        except Exception as e:
            print(f'[FAIL] {c} seed{s}: {repr(e)[:200]}', flush=True)


if __name__ == '__main__':
    main()
