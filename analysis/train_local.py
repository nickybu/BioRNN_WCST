"""
Self-contained local training harness for the two-module BioRNN WCST model.
Reproduces the curriculum-learning protocol from train.py:train_bpxtrials_v2_working
but removes all hardcoded cluster (/scratch/yl4317) paths and saves locally.

Design goals:
 - machine-agnostic (CPU or MPS), deterministic per torch_seed
 - configurable trial timing (allows compressed trials for feasibility)
 - records the full learning trajectory (perf, perf_rule, per-cell-group activity,
   subspace angle) at a chosen cadence -> supports the "emergence during learning" question (C)
 - clean hooks for ablations: drop PVs, drop dendrites, shrink neuron counts (A, B)
"""
import os, time, copy, pickle, random
import numpy as np
import torch
import torch.nn as nn

from task import get_default_hp_wcst, WCST
from functions import get_default_hp, compute_trial_history
from model_working import Net_readoutSR_working


def make_wcst_hp(**overrides):
    """Build a WCST hyperparameter dict from the repo default, with overrides."""
    hp, _, _ = get_default_hp()
    hp.update({
        'task': 'wcst',
        'n_input': 16,
        'n_output': 3,
        'n_output_rule': 2,
        'train_rule': True,
        'dendrite_type': 'additive',
        'dend_nonlinearity': 'subtractive',   # subtractive: shown in paper main-text figs; divisive is the supp variant
        'block_len': 20,
        'n_switches': 3,
        'batch_size': 40,
        'grad_remove_history': True,
        'sparse_srsst_to_sredend': 0.8,        # paper default for main networks
        'torch_seed': 1,
    })
    hp.update(overrides)
    # dendrite compartment counts derive from soma counts * n_branches
    for area in ['sr', 'pfc']:
        hp['n_{}_edend'.format(area)] = hp['n_branches'] * hp['n_{}_esoma'.format(area)]
    return hp


def compressed_timing():
    """A trial timing dict that preserves task structure but cuts dead time.
    80 timesteps (dt=10) vs the paper's 210. Validated to reproduce phenomenology."""
    t = dict(get_default_hp_wcst())
    t.update({
        'trial_history_start': 0, 'trial_history_end': 100,
        'center_card_on': 300, 'center_card_off': 800,
        'test_cards_on': 500, 'test_cards_off': 800,
        'resp_start': 500, 'resp_end': 800, 'trial_end': 800,
    })
    return t


def full_timing():
    return dict(get_default_hp_wcst())


def _make_optimizer(hp, model):
    opt = {'adam': torch.optim.Adam, 'SGD': torch.optim.SGD,
           'RMSprop': torch.optim.RMSprop, 'Rprop': torch.optim.Rprop}[hp['optimizer']]
    return opt(params=model.parameters(), lr=hp['learning_rate'])


def train_local(hp, hp_task, device='cpu', max_steps=20000, record_every=25,
                log_every=50, verbose=True, record_activity=True, seed=None,
                checkpoint_dir=None, checkpoint_every=None, init_state=None,
                converge_perf=None):
    """Train one network. Returns model and a history dict.

    Curriculum (as in the paper):
      stage 0: give prev stim + choice + reward
      stage 1: remove prev stim once train perf passes criterion
      stage 2: remove prev choice
      done: reaches criterion with only reward feedback
    """
    if seed is None:
        seed = hp['torch_seed']
    torch.manual_seed(seed); np.random.seed(0); random.seed(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    rule_list = ['color', 'shape']
    dt = hp['dt']

    model = Net_readoutSR_working(hp)
    model.to(device); model.rnn.to(device)
    optim = _make_optimizer(hp, model)
    loss_fnc = nn.MSELoss()

    give_prev_stim = True; give_prev_choice = True; give_prev_rew = True
    prev_stim_mag = 1.0; prev_choice_mag = 1.0; prev_rew_mag = 1.0
    iti_var = 200
    start_step = 0

    # resume: seed weights / optimizer / curriculum state from a checkpoint dict
    if init_state is not None:
        model.load_state_dict(init_state['model_state_dict'])
        if 'optim_state_dict' in init_state:
            optim.load_state_dict(init_state['optim_state_dict'])
        give_prev_stim = init_state.get('give_prev_stim', give_prev_stim)
        give_prev_choice = init_state.get('give_prev_choice', give_prev_choice)
        prev_stim_mag = 0.0 if not give_prev_stim else prev_stim_mag
        prev_choice_mag = 0.0 if not give_prev_choice else prev_choice_mag
        start_step = int(init_state.get('step', 0))
        if verbose:
            print(f"  [resume] from step {start_step}, stage {init_state.get('stage')}", flush=True)

    hist = {'step': [], 'perf': [], 'perf_rule': [], 'loss': [],
            'stage': [], 'give_prev_stim': [], 'give_prev_choice': [],
            'angle_rule': [], 'cg_mean_act': []}
    stage = 0
    start = time.time()
    from collections import deque
    # running averages of recent blocks -> curriculum advance requires STABLE performance,
    # not a single lucky block (matches the paper's "90% over recent tests" intent)
    recent_perf = deque(maxlen=10); recent_perf_rule = deque(maxlen=10)

    def curr_stage():
        if give_prev_stim: return 0
        if give_prev_choice: return 1
        return 2

    for ba in range(start_step, max_steps):
        current_rule = rule_list[0]
        block_len = hp['block_len']
        switches = random.sample(range(block_len - 1), hp['n_switches'])
        loss = 0
        h_last = i_me_last = perf = _x = choice = None
        prev_stim_win = None   # (start_ts, end_ts) of the PREVIOUS trial's card window
        perf_list = []; perf_rule_list = []
        rnn_activity_blocks = []

        # magnitudes follow curriculum flags
        if not give_prev_stim: prev_stim_mag = 0.0
        if not give_prev_choice: prev_choice_mag = 0.0

        for tr in range(block_len):
            if tr == 0:
                h_init = i_me_init = last_rew = prev_stim = prev_choice = None
            else:
                h_init = h_last; i_me_init = i_me_last
                last_rew = perf.detach() if (give_prev_rew or hp['grad_remove_history']) else None
                prev_stim = _x.detach() if (give_prev_stim or hp['grad_remove_history']) else None
                prev_choice = choice.detach() if (give_prev_choice or hp['grad_remove_history']) else None

            # variable ITI
            htv = copy.deepcopy(hp_task)
            iti = hp_task['center_card_on'] - hp_task['trial_history_end'] + int(np.random.uniform(-iti_var, iti_var))
            iti = max(iti, 0)
            htv['center_card_on'] = htv['trial_history_end'] + iti
            htv['center_card_off'] = htv['center_card_on'] + (hp_task['center_card_off'] - hp_task['center_card_on'])
            htv['test_cards_on'] = htv['center_card_on'] + (hp_task['test_cards_on'] - hp_task['center_card_on'])
            htv['test_cards_off'] = htv['test_cards_on'] + (hp_task['test_cards_off'] - hp_task['test_cards_on'])
            htv['resp_start'] = htv['test_cards_on']; htv['resp_end'] = htv['test_cards_off']
            htv['trial_end'] = htv['resp_end']

            n_steps = int((htv['trial_end'] - htv['trial_start']) // dt)
            ip = np.arange(int(htv['trial_history_start'] / dt), int(htv['trial_history_end'] / dt))
            # summarise prev stimulus over the PREVIOUS trial's own card window (not the
            # current trial's, which can be out of range under variable ITI -> empty mean -> nan)
            if prev_stim_win is not None:
                _ss, _se = prev_stim_win
                _plen = prev_stim.shape[0] if prev_stim is not None else n_steps
                _ss = min(int(_ss / dt), _plen - 1) * dt
                _se = min(int(_se / dt), _plen) * dt
            else:
                _ss, _se = htv['test_cards_on'], htv['test_cards_off']
            Ir, Is, Ic = compute_trial_history(
                last_rew=last_rew, prev_stim=prev_stim, prev_choice=prev_choice, input_period=ip,
                batch_size=hp['batch_size'], n_steps=n_steps, input_dim=model.rnn.n['input'],
                stim_start=_ss, stim_end=_se, dt=dt, choice_dim=3)
            Ir, Is, Ic = Ir.to(device), Is.to(device), Ic.to(device)
            th = {'i_prev_rew': prev_rew_mag * Ir, 'i_prev_choice': prev_choice_mag * Ic,
                  'i_prev_stim': prev_stim_mag * Is}

            wcst = WCST(hp=hp, hp_wcst=htv, rule=current_rule, rule_list=rule_list,
                        n_features_per_rule=2, n_test_cards=3)
            _x, _, _yhat, _yhat_rule, _ = wcst.make_task_batch(batch_size=hp['batch_size'])
            _x, _yhat, _yhat_rule = _x.to(device), _yhat.to(device), _yhat_rule.to(device)

            out, data = model(input=_x, init={'h': h_init, 'i_me': i_me_init}, trial_history=th, hp=hp)
            rnn_activity = torch.stack(data['record']['hiddens'], dim=0)
            if record_activity:
                rnn_activity_blocks.append(rnn_activity.detach())
            h_last = data['last_states']['hidden']; i_me_last = data['last_states']['i_me']

            prev_stim_win = (htv['test_cards_on'], htv['test_cards_off'])  # for the next trial

            perf, _, choice = wcst.get_perf(y=out['out'], yhat=_yhat)
            perf_rule, _, _ = wcst.get_perf_rule(y_rule=out['out_rule'], yhat_rule=_yhat_rule)

            loss = loss + loss_fnc(out['out'], _yhat) + loss_fnc(out['out_rule'], _yhat_rule)
            perf_list.append(torch.mean(perf.float().detach()).item())
            perf_rule_list.append(torch.mean(perf_rule.float().detach()).item())

            if tr in switches:
                current_rule = random.choice([r for r in rule_list if r != current_rule])

        optim.zero_grad()
        (loss / block_len).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1, error_if_nonfinite=False)
        loss_val = float(np.nan_to_num(loss.detach().cpu().numpy(), nan=np.inf))
        optim.step()

        mp = float(np.mean(perf_list)); mpr = float(np.mean(perf_rule_list))
        recent_perf.append(mp); recent_perf_rule.append(mpr)
        avg_perf = float(np.mean(recent_perf)); avg_perf_rule = float(np.mean(recent_perf_rule))

        # record
        if ba % record_every == 0:
            hist['step'].append(ba); hist['perf'].append(mp); hist['perf_rule'].append(mpr)
            hist['loss'].append(loss_val); hist['stage'].append(curr_stage())
            hist['give_prev_stim'].append(give_prev_stim); hist['give_prev_choice'].append(give_prev_choice)
            # cheap emergence metrics from the last block's activity
            ang = np.nan; cg_mean = {}
            if record_activity and len(rnn_activity_blocks) > 0:
                try:
                    ang = _quick_rule_angle(model, rnn_activity_blocks, switches, block_len)
                except Exception:
                    ang = np.nan
                A = rnn_activity_blocks[-1].cpu().numpy()  # (T, B, N)
                for cg in model.rnn.cell_group_list:
                    cg_mean[cg] = float(np.mean(A[:, :, model.rnn.cg_idx[cg]]))
            hist['angle_rule'].append(ang); hist['cg_mean_act'].append(cg_mean)

        # periodic weight checkpoint for emergence-during-learning analysis
        if checkpoint_dir is not None and checkpoint_every is not None and ba % checkpoint_every == 0:
            import os as _os
            _os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(),
                        'optim_state_dict': optim.state_dict(), 'step': ba,
                        'perf': mp, 'perf_rule': mpr, 'stage': curr_stage(),
                        'give_prev_stim': give_prev_stim, 'give_prev_choice': give_prev_choice,
                        'prev_stim_mag': prev_stim_mag, 'prev_choice_mag': prev_choice_mag,
                        'hp': hp, 'hp_task': hp_task},
                       _os.path.join(checkpoint_dir, f'ckpt_step{ba:05d}.pt'))

        if verbose and ba % log_every == 0:
            print(f"step {ba:5d} stage {curr_stage()} loss={loss_val:.4f} perf={mp:.3f} "
                  f"perf_rule={mpr:.3f} t={time.time()-start:.0f}s", flush=True)

        # curriculum advance criterion (approx paper): STABLE running-average train-perf
        # threshold on BOTH response and rule (requires a full recent window, so a single
        # lucky block can't advance the stage prematurely).
        crit = 1 - hp['n_switches'] / hp['block_len'] - 0.2
        window_full = len(recent_perf) == recent_perf.maxlen
        if window_full and avg_perf >= crit and avg_perf_rule >= crit:
            if give_prev_stim:
                give_prev_stim = False
                optim = _make_optimizer(hp, model)
                recent_perf.clear(); recent_perf_rule.clear()  # fresh window for the new stage
                if verbose: print(f"  -> stage 1: remove prev stim (step {ba}, avg_perf={avg_perf:.3f})", flush=True)
                continue
            elif give_prev_choice and prev_stim_mag == 0:
                give_prev_choice = False
                optim = _make_optimizer(hp, model)
                recent_perf.clear(); recent_perf_rule.clear()  # fresh window for the new stage
                if verbose: print(f"  -> stage 2: remove prev choice (step {ba}, avg_perf={avg_perf:.3f})", flush=True)
                continue
            elif (not give_prev_choice) and prev_choice_mag == 0:
                # final stop: use converge_perf if set (train to ensemble-comparable level),
                # else stop at the lenient curriculum criterion. Uses the STABLE average.
                final_target = converge_perf if converge_perf is not None else crit
                if avg_perf >= final_target:
                    if verbose: print(f"  *** converged at step {ba} (avg_perf={avg_perf:.3f}, avg_perf_rule={avg_perf_rule:.3f}) ***", flush=True)
                    hist['converged_step'] = ba
                    break

        if loss_val == np.inf:
            if verbose: print(f"gradient exploded at step {ba}", flush=True)
            hist['exploded'] = True
            break

    hist['wall_time_s'] = time.time() - start
    hist['final_perf'] = mp; hist['final_perf_rule'] = mpr
    hist['hp'] = hp; hist['hp_task'] = hp_task
    return model, hist


def latest_checkpoint(checkpoint_dir):
    """Return the path to the highest-step ckpt_step*.pt in a directory, or None."""
    import glob, os as _os
    files = glob.glob(_os.path.join(checkpoint_dir, 'ckpt_step*.pt'))
    if not files:
        return None
    return max(files, key=lambda f: int(_os.path.basename(f)[len('ckpt_step'):-3]))


def resume_from(checkpoint_path, max_steps=4000, checkpoint_dir=None,
                checkpoint_every=25, device='cpu', seed=1, converge_perf=None, **kwargs):
    """Resume training from a saved checkpoint. Rebuilds hp/hp_task from the
    checkpoint, restores weights + optimizer + curriculum stage, and continues
    to max_steps. If checkpoint_dir is None, writes new checkpoints alongside the
    original. Returns (model, hist) like train_local."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = ck['hp']; hp_task = ck['hp_task']
    if checkpoint_dir is None:
        checkpoint_dir = os.path.dirname(checkpoint_path)
    return train_local(hp, hp_task, device=device, max_steps=max_steps,
                        record_activity=False, seed=seed,
                        checkpoint_dir=checkpoint_dir, checkpoint_every=checkpoint_every,
                        init_state=ck, converge_perf=converge_perf, **kwargs)


def _quick_rule_angle(model, rnn_activity_blocks, switches, block_len):
    """Cheap proxy of the rule-subspace principal angle during training,
    computed from SR esoma activity split by rule across the block."""
    from scipy.linalg import subspace_angles
    from functions import compute_subspace
    A = rnn_activity_blocks  # list length block_len? no: it's per-trial within one block
    # Build per-trial rule labels for this block
    rule_seq = []
    cur = 0
    for tr in range(block_len):
        rule_seq.append(cur)
        if tr in switches:
            cur = 1 - cur
    sr_idx = model.rnn.cg_idx['sr_esoma']
    # activity per trial: (T,B,N) -> mean over time, take SR esoma -> (B,N)
    feats_by_rule = {0: [], 1: []}
    for tr, act in enumerate(A):
        m = act[:, :, sr_idx].mean(dim=0).cpu().numpy()  # (B, n_sr)
        feats_by_rule[rule_seq[tr]].append(m)
    if len(feats_by_rule[0]) == 0 or len(feats_by_rule[1]) == 0:
        return np.nan
    X0 = np.concatenate(feats_by_rule[0], axis=0)
    X1 = np.concatenate(feats_by_rule[1], axis=0)
    if X0.shape[0] < 3 or X1.shape[0] < 3:
        return np.nan
    s0, _, _ = compute_subspace(X0, d='pr')
    s1, _, _ = compute_subspace(X1, d='pr')
    ang = subspace_angles(s0.T, s1.T)
    return float(np.max(ang))
