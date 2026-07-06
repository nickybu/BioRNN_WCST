"""
Ablation configurations for the BioRNN WCST retraining experiments.

Q(A): drop PV interneurons     -> remove sr_pv / pfc_pv from the architecture and retrain
Q(B): drop dendrites + shrink  -> n_branches=1 (no separate dendritic branches) and
                                   shrink esoma counts to a handful, then retrain
Capstone: transfer             -> train on rule set 1, then measure reuse on a related task

Each builder returns an hp dict compatible with train_local.train_local / make_wcst_hp.
"""
from train_local import make_wcst_hp


def hp_baseline(**ov):
    """Full model, subtractive dendrites, sparsity 0.8 (matches the pretrained main nets)."""
    return make_wcst_hp(dend_nonlinearity='subtractive', sparse_srsst_to_sredend=0.8, **ov)


def hp_drop_pv(**ov):
    """Q(A): remove PV interneurons from both modules entirely, then retrain.

    PV is dropped from cell_group_list and its counts zeroed so the network must
    learn the task with only SST + VIP inhibition."""
    hp = hp_baseline(**ov)
    hp['cell_group_list'] = [cg for cg in hp['cell_group_list'] if not cg.endswith('_pv')]
    hp['n_sr_pv'] = 0
    hp['n_pfc_pv'] = 0
    return hp


def hp_drop_dendrites(n_esoma=5, **ov):
    """Q(B): collapse dendrites (n_branches=1, so edend mirrors esoma with a single
    compartment) and shrink the excitatory populations to a handful of neurons.

    n_esoma controls the number of E cells per module (default 5)."""
    hp = hp_baseline(**ov)
    hp['n_branches'] = 1
    hp['n_sr_esoma'] = n_esoma
    hp['n_pfc_esoma'] = n_esoma
    # edend counts are derived as n_branches * n_esoma inside make_wcst_hp, but since we
    # override after construction, recompute here:
    for area in ['sr', 'pfc']:
        hp['n_{}_edend'.format(area)] = hp['n_branches'] * hp['n_{}_esoma'.format(area)]
    return hp


def hp_shrink_only(n_esoma=5, **ov):
    """Control for Q(B): shrink E populations but KEEP 2 dendritic branches, to separate
    the effect of removing dendrites from the effect of shrinking."""
    hp = hp_baseline(**ov)
    hp['n_sr_esoma'] = n_esoma
    hp['n_pfc_esoma'] = n_esoma
    for area in ['sr', 'pfc']:
        hp['n_{}_edend'.format(area)] = hp['n_branches'] * hp['n_{}_esoma'.format(area)]
    return hp
