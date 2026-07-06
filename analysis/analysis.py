"""
Reusable analysis toolkit for the two-module BioRNN WCST networks.

Wraps the repo's own simulation code (test_frozen_weights / generate_neural_data_test)
and adds the decoding / subspace / lesion / silencing analyses needed to test:
  - where the rule is maintained (PFC vs SR)
  - whether rule is held in recurrent dynamics vs feedforward connectivity
  - whether SST uniquely controls the rule-subspace angle

All functions take a loaded `model` (from load_net) plus its hp/hp_task.
"""
import os, io, time, copy, random, warnings
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import subspace_angles

warnings.filterwarnings("ignore")

from functions import (test_frozen_weights, generate_neural_data_test,
                       compute_subspace, participation_ratio)

DT = 10
CELL_GROUPS = ['sr_esoma','sr_edend','sr_pv','sr_sst','sr_vip',
               'pfc_esoma','pfc_edend','pfc_pv','pfc_sst','pfc_vip']


# ---------------------------------------------------------------- loading
def load_net(path):
    """Load a checkpoint, patching attributes missing on older saves."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = ck['model']
    for attr, val in [('output_noise', 0.0)]:
        if not hasattr(m, attr):
            setattr(m, attr, val)
    return m, ck['hp'], ck['hp_task'], ck


def cg_idx(model, cg):
    return np.array(model.rnn.cg_idx[cg])


def module_idx(model, module):
    """All excitatory soma indices for a module ('sr' or 'pfc')."""
    return cg_idx(model, f'{module}_esoma')


def windows(hp_task, dt=DT):
    return {k: int(hp_task[k] / dt) for k in
            ['trial_history_start','trial_history_end','center_card_on',
             'center_card_off','test_cards_on','test_cards_off','trial_end']
            if k in hp_task}


# ---------------------------------------------------------------- simulate
def simulate(model, hp, hp_task, n_trials=120, switch_every=15, n_switches=None,
             random_switch=False, opto=None, seed=None, batch_size=1):
    """Run the network and return activity + trial labels as numpy.

    Returns dict with:
      act:   (n_trials, T, N)  single-sample activity
      rules: (n_trials,) int   0/1 rule label per trial
      perf:  (n_trials,)       response correct
      perf_rule: (n_trials,)   rule-readout correct
      rule_names: list of str
    """
    if seed is not None:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if n_switches is None:
        n_switches = max(1, n_trials // switch_every - 1)
    out = generate_neural_data_test(
        model=model, n_trials_test=n_trials, hp_test=hp, hp_task_test=hp_task,
        batch_size=batch_size, random_switch=random_switch, n_switches=n_switches,
        switch_every_test=switch_every, opto=opto)
    act = out['rnn_activity'][:, :, 0, :].detach().cpu().numpy()  # (ntr, T, N)
    td = out['test_data']
    rule_names = list(td['rules'])
    rules = np.array([0 if r == 'color' else 1 for r in rule_names])
    perf = np.array([float(np.mean(p)) for p in td['perfs']])
    perf_rule = np.array([float(np.mean(p)) for p in td['perf_rules']])
    return dict(act=act, rules=rules, perf=perf, perf_rule=perf_rule,
                rule_names=rule_names)


# ---------------------------------------------------------------- decoding
def _trial_feature(act, idx, t0, t1):
    """Mean activity over [t0,t1) for units idx -> (n_trials, len(idx))."""
    return act[:, t0:t1, :][:, :, idx].mean(axis=1)


def rule_decode_cv(X, y, n_splits=5, C=1.0):
    """Cross-validated linear rule decoding accuracy. Returns mean acc."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    if len(np.unique(y)) < 2:
        return np.nan
    # need enough of each class
    counts = np.bincount(y)
    if counts.min() < n_splits:
        n_splits = max(2, counts.min())
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000)
        clf.fit(sc.transform(X[tr]), y[tr])
        accs.append(clf.score(sc.transform(X[te]), y[te]))
    return float(np.mean(accs))


def rule_decode_timecourse(act, rules, idx, win=5, step=5):
    """Sliding-window CV rule decoding accuracy across the trial for population idx.
    Returns (centers, accs)."""
    T = act.shape[1]
    centers, accs = [], []
    for t0 in range(0, T - win + 1, step):
        X = act[:, t0:t0+win, :][:, :, idx].mean(axis=1)
        accs.append(rule_decode_cv(X, rules))
        centers.append(t0 + win // 2)
    return np.array(centers), np.array(accs)


# ---------------------------------------------------------------- subspace
def _traj_points(act, idx, t0, t1):
    """Trajectory points: stack (trial, timestep) as samples -> (ntr, n_samp_pts, nfeat)
    kept per-trial so shuffles respect trial identity.
    Returns array (ntr, (t1-t0)*len(idx))? No: returns (ntr, ntime, nfeat)."""
    return act[:, t0:t1, :][:, :, idx]  # (ntr, ntime, nfeat)


def _subspace_from_points(P, d='pr'):
    """P: (n_points, n_feat). Caps PCA dims below n_samples."""
    from sklearn.decomposition import PCA
    n_samp, n_feat = P.shape
    kmax = min(n_samp - 1, n_feat)
    pca = PCA(n_components=kmax)
    pca.fit(P)
    evr = pca.explained_variance_ratio_
    pr = int(np.round(participation_ratio(evr)))
    pr = max(1, min(pr, kmax))
    return pca.components_[:pr]


def rule_subspace_angle(act, rules, idx, t0, t1, d='pr'):
    """Largest principal angle (deg) between the two rules' activity subspaces
    for population idx over window [t0,t1), using trajectory points as samples."""
    Pr = _traj_points(act, idx, t0, t1)
    # drop trials whose trajectory contains non-finite values (network diverged under silencing)
    finite_tr = np.isfinite(Pr.reshape(Pr.shape[0], -1)).all(axis=1)
    rules = np.asarray(rules)[finite_tr]
    Pr = Pr[finite_tr]
    P0 = Pr[rules == 0].reshape(-1, Pr.shape[-1])
    P1 = Pr[rules == 1].reshape(-1, Pr.shape[-1])
    if P0.shape[0] < 3 or P1.shape[0] < 3:
        return np.nan
    s0 = _subspace_from_points(P0, d=d)
    s1 = _subspace_from_points(P1, d=d)
    ang = subspace_angles(s0.T, s1.T)
    return float(np.degrees(np.max(ang)))


def rule_subspace_angle_shuffle(act, rules, idx, t0, t1, n_shuffle=100, d='pr'):
    """Shuffle null: randomly split ALL trials in half (ignoring rule), angle between halves."""
    Pr = _traj_points(act, idx, t0, t1)
    ntr = Pr.shape[0]; nfeat = Pr.shape[-1]
    angs = []
    rng = np.random.default_rng(0)
    for _ in range(n_shuffle):
        perm = rng.permutation(ntr)
        half = ntr // 2
        A = Pr[perm[:half]].reshape(-1, nfeat)
        B = Pr[perm[half:]].reshape(-1, nfeat)
        if A.shape[0] < 3 or B.shape[0] < 3:
            continue
        sA = _subspace_from_points(A, d=d)
        sB = _subspace_from_points(B, d=d)
        angs.append(np.degrees(np.max(subspace_angles(sA.T, sB.T))))
    return np.array(angs)


# ---------------------------------------------------------------- lesion / silencing
def make_opto(model, cell_groups, value=0.0, t_range=None, hp_task=None):
    """Build an opto spec (silence given cell groups by clamping to `value`).

    The model.forward expects opto as a dict; we pass neuron indices + timesteps.
    Returns dict consumable by test_frozen_weights' opto path."""
    idx = np.concatenate([cg_idx(model, cg) for cg in cell_groups]) if cell_groups else np.array([], int)
    if t_range is None and hp_task is not None:
        t_range = list(range(int(hp_task['trial_end'] / DT)))
    # model.forward expects keys: 't', 'neuron_idx', 'value'
    return {'neuron_idx': idx, 'value': value, 't': t_range}


def make_opto_idx(idx, value=0.0, t_range=None, hp_task=None):
    """Opto spec for explicit neuron indices (e.g. random E-subset controls)."""
    if t_range is None and hp_task is not None:
        t_range = list(range(int(hp_task['trial_end'] / DT)))
    return {'neuron_idx': np.asarray(idx, int), 'value': value, 't': t_range}


def effective_w_rec(model):
    """Effective recurrent weight matrix (Dale + fixed dend coupling)."""
    return model.rnn.effective_weight(
        w=model.rnn.w_rec, mask=model.rnn.mask,
        w_fix=torch.abs(model.rnn.dend2soma) * model.rnn.w_fix).detach().cpu().numpy()


def autonomous_run(model, h_init_vec, hp, n_steps=80, batch=1):
    """Run the network from h_init with ZERO external input & zero trial-history,
    to test whether a rule state persists purely through recurrent dynamics.
    Returns (n_steps, N) activity."""
    model.rnn.batch_size = batch
    input_dim = model.rnn.n['input']
    x = torch.zeros(n_steps, batch, input_dim)
    h_init = torch.tensor(np.tile(h_init_vec, (batch, 1)), dtype=torch.float32)
    th = {'i_prev_rew': torch.zeros(n_steps, batch, 2),
          'i_prev_choice': torch.zeros(n_steps, batch, 3),
          'i_prev_stim': torch.zeros(n_steps, batch, input_dim)}
    with torch.no_grad():
        out, data = model(input=x, init={'h': h_init, 'i_me': None},
                          trial_history=th, hp=hp)
    return torch.stack(data['record']['hiddens'], 0)[:, 0, :].numpy()


def rule_states(sim, model, t_window=3):
    """Mean full hidden state per rule at end of maintenance (just before response).
    Returns (h_rule0, h_rule1) each (N,)."""
    from analysis import windows  # local
    # caller passes sim with 'act','rules'; window computed by caller
    raise NotImplementedError  # use inline; kept for reference


def attractor_retention(model, hp, hp_task, h0, h1, idx, n_steps=80):
    """Autonomous separation retention for population idx:
    ||traj0-traj1||_end / ||traj0-traj1||_start. ~1 = maintained, ~0 = collapsed."""
    tr0 = autonomous_run(model, h0, hp, n_steps=n_steps)
    tr1 = autonomous_run(model, h1, hp, n_steps=n_steps)
    sep = np.linalg.norm(tr0[:, idx] - tr1[:, idx], axis=1)
    return sep, float(sep[-1] / (sep[0] + 1e-9))


def lesion_intermodular(model):
    """Return a deep-copied model with PFC<->SR long-range weights zeroed
    (both w_rec entries and the mask), leaving within-module weights intact."""
    m2 = copy.deepcopy(model)
    sr = [i for cg in CELL_GROUPS if cg.startswith('sr') for i in model.rnn.cg_idx[cg]]
    pf = [i for cg in CELL_GROUPS if cg.startswith('pfc') for i in model.rnn.cg_idx[cg]]
    sr = np.array(sr); pf = np.array(pf)
    with torch.no_grad():
        for a, b in [(sr, pf), (pf, sr)]:
            m2.rnn.w_rec[np.ix_(a, b)] = 0.0
            m2.rnn.mask[np.ix_(a, b)] = 0.0
    return m2, sr, pf


def module_indices(model):
    """Return (sr_all, pfc_all) index arrays over ALL cell groups in each module."""
    sr = np.array([i for cg in CELL_GROUPS if cg.startswith('sr') for i in model.rnn.cg_idx[cg]])
    pf = np.array([i for cg in CELL_GROUPS if cg.startswith('pfc') for i in model.rnn.cg_idx[cg]])
    return sr, pf


def rule_current_decomposition(model, sim, t0, t1):
    """For each esoma population, decompose the rule-discriminating recurrent input
    current into LOCAL (within-module) vs LONG-RANGE (inter-modular) components.

    Returns dict per module: fraction of the rule-difference current carried by
    local vs long-range projections. Uses W_eff and mean per-rule activity.
    """
    W = effective_w_rec(model)  # (N, N): w[i,j] = i sends to j? check convention
    # model.forward uses hidden@w_rec_eff -> current to receiver j = sum_i h_i w[i,j]
    sr_all, pfc_all = module_indices(model)
    act = sim['act']; rules = sim['rules']
    # mean activity per rule over the maintenance window: (N,)
    hbar0 = act[rules == 0][:, t0:t1, :].mean((0, 1))
    hbar1 = act[rules == 1][:, t0:t1, :].mean((0, 1))
    dh = hbar1 - hbar0  # rule-difference activity vector (senders)
    res = {}
    # Long-range projections target DENDRITES (pfc_esoma->sr_edend, sr_esoma->pfc_edend),
    # so measure rule-difference current into each module's excitatory compartments
    # (soma + dendrite), split by whether the sender is local or in the other module.
    for mod in ['sr', 'pfc']:
        rec_idx = np.concatenate([cg_idx(model, f'{mod}_esoma'), cg_idx(model, f'{mod}_edend')])
        local_send = sr_all if mod == 'sr' else pfc_all
        long_send = pfc_all if mod == 'sr' else sr_all
        # per-receiver net rule-difference current from each sender set, then abs-sum over receivers
        cur_local_vec = dh[local_send] @ W[np.ix_(local_send, rec_idx)]
        cur_long_vec = dh[long_send] @ W[np.ix_(long_send, rec_idx)]
        cur_local = np.abs(cur_local_vec).sum()
        cur_long = np.abs(cur_long_vec).sum()
        tot = cur_local + cur_long + 1e-9
        res[mod] = dict(local=float(cur_local), long_range=float(cur_long),
                        frac_long=float(cur_long / tot))
    return res
