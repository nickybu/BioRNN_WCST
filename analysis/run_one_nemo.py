"""
Single-network training entrypoint for Nemo GPU jobs (one condition+seed per SLURM job).
Runs inside biornn.sif via: apptainer exec --nv biornn.sif python run_one_nemo.py --cond X --seed N ...

Full timing (per the timing-cap finding: compressed timing shrinks the pre-card ITI gap
~5x and caps test performance at ~0.5). GPU device by default.
Writes to <outdir>/<cond>_seed<seed>/: ckpts/, history.npz, meta.json, final.pt
Resumable: skips if final.pt already exists.
"""
import argparse, os, json, time
# Pin thread counts BEFORE importing torch/numpy. On Nemo's 256-core nodes,
# torch/OpenMP/MKL otherwise size their thread pools to the physical core count
# while the cgroup grants only a few cores -> pathological oversubscription
# (observed: 21 min for 0 steps at 4 cores). Respect SLURM's allocation.
_ncores = os.environ.get('SLURM_CPUS_PER_TASK') or os.environ.get('SLURM_CPUS_ON_NODE') or '4'
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = _ncores
import numpy as np
import torch
torch.set_num_threads(int(_ncores))
torch.set_num_interop_threads(1)

import ablations as ab
from train_local import train_local, full_timing, compressed_timing

CONDITIONS = {
    'baseline':  lambda n=5: ab.hp_baseline(),
    'drop_pv':   lambda n=5: ab.hp_drop_pv(),
    'drop_dend': lambda n=5: ab.hp_drop_dendrites(n),
    'shrink':    lambda n=5: ab.hp_shrink_only(n),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', required=True, choices=list(CONDITIONS))
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--max_steps', type=int, default=8000)
    ap.add_argument('--converge_perf', type=float, default=0.85)
    ap.add_argument('--checkpoint_every', type=int, default=50)
    ap.add_argument('--timing', choices=['full','compressed'], default='full')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--n_esoma', type=int, default=5)
    args = ap.parse_args()

    device = args.device if (args.device=='cpu' or torch.cuda.is_available()) else 'cpu'
    print(f'device={device} cuda_avail={torch.cuda.is_available()}', flush=True)
    if device=='cuda':
        print('gpu:', torch.cuda.get_device_name(0), flush=True)

    hp = CONDITIONS[args.cond](args.n_esoma)
    hp['torch_seed'] = args.seed
    hp_task = full_timing() if args.timing=='full' else compressed_timing()

    ndir = os.path.join(args.outdir, f'{args.cond}_seed{args.seed}')
    ckdir = os.path.join(ndir, 'ckpts'); os.makedirs(ckdir, exist_ok=True)
    if os.path.exists(os.path.join(ndir, 'final.pt')):
        print(f'[skip] {args.cond} seed{args.seed} already done', flush=True); return

    t0 = time.time()
    model, hist = train_local(hp, hp_task, device=device, max_steps=args.max_steps,
                              record_activity=False, seed=args.seed,
                              checkpoint_dir=ckdir, checkpoint_every=args.checkpoint_every,
                              converge_perf=args.converge_perf, log_every=100, verbose=True)
    np.savez(os.path.join(ndir, 'history.npz'),
             step=hist['step'], perf=hist['perf'], perf_rule=hist['perf_rule'],
             loss=hist['loss'], stage=hist['stage'])
    json.dump({'cond': args.cond, 'seed': args.seed, 'timing': args.timing,
               'converged_step': hist.get('converged_step'),
               'final_perf': hist.get('final_perf'), 'final_perf_rule': hist.get('final_perf_rule'),
               'wall_time_s': hist.get('wall_time_s'), 'exploded': hist.get('exploded', False),
               'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad)},
              open(os.path.join(ndir, 'meta.json'), 'w'))
    # save model on CPU for portable reload
    torch.save({'model_state_dict': {k: v.cpu() for k,v in model.state_dict().items()},
                'hp': hp, 'hp_task': hp_task},
               os.path.join(ndir, 'final.pt'))
    fp = hist.get('final_perf'); fp = fp if fp is not None else float('nan')
    print(f'[done] {args.cond} seed{args.seed}: converged_step={hist.get("converged_step")} '
          f'final_perf={fp:.3f} ({time.time()-t0:.0f}s)', flush=True)

if __name__ == '__main__':
    main()
