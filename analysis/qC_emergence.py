
"""Q(C): emergence of rule properties DURING learning.
For each checkpoint in a training run, rebuild model, simulate, and measure:
  - perf, perf_rule (task behavior)
  - PFC rule decodability, SR rule decodability (population code emergence)
  - SR rule-subspace angle (the paper's Fig-7 geometry)
Tracks WHEN neural rule structure appears relative to behavioral performance.
"""
import sys, os, json, glob
sys.path.insert(0, "/work")
import numpy as np, torch, random
import analysis as A
from model_working import Net_readoutSR_working

def analyze_ckpt(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hp, hp_task = ck["hp"], ck["hp_task"]
    m = Net_readoutSR_working(hp)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    w = A.windows(hp_task)
    # decode window: late delay just before test cards (pre-card memory of rule)
    t0, t1 = w["test_cards_on"]-5, w["test_cards_on"]
    sim = A.simulate(m, hp, hp_task, n_trials=100, switch_every=15, seed=0)
    act, rules = sim["act"], sim["rules"]
    pfc = A.cg_idx(m,"pfc_esoma"); sr = A.cg_idx(m,"sr_esoma")
    out = {"step": int(ck["step"]), "stage": int(ck["stage"]),
           "perf": float(np.mean(sim["perf"])), "perf_rule": float(np.mean(sim["perf_rule"]))}
    try:
        Xp = A._trial_feature(act, pfc, t0, t1); out["dec_pfc"]=float(A.rule_decode_cv(Xp, rules))
    except Exception as e: out["dec_pfc"]=None
    try:
        Xs = A._trial_feature(act, sr, t0, t1); out["dec_sr"]=float(A.rule_decode_cv(Xs, rules))
    except Exception as e: out["dec_sr"]=None
    try:
        out["sr_angle"]=float(A.rule_subspace_angle(act, rules, sr, t0, t1))
    except Exception as e: out["sr_angle"]=None
    return out

def run_net(ckdir, stride):
    fs = sorted(glob.glob(ckdir+"/ckpt_step*.pt"))
    fs = fs[::stride] + ([fs[-1]] if fs and fs[-1] not in fs[::stride] else [])
    res=[]
    for f in fs:
        try:
            res.append(analyze_ckpt(f)); print("  ok", os.path.basename(f), res[-1]["step"], flush=True)
        except Exception as e:
            print("  FAIL", os.path.basename(f), repr(e), flush=True)
    return res

if __name__=="__main__":
    base="/out"
    stride=int(os.environ.get("STRIDE","4"))
    nets = json.loads(os.environ["NETS"])  # list of "batch/net"
    allres={}
    for nk in nets:
        ckdir = base+"/"+nk+"/ckpts"
        print("=== "+nk+" ("+str(len(glob.glob(ckdir+"/ckpt_step*.pt")))+" ckpts, stride="+str(stride)+") ===", flush=True)
        allres[nk]=run_net(ckdir, stride)
    json.dump(allres, open(base+"/qC_emergence.json","w"))
    print("WROTE", base+"/qC_emergence.json", flush=True)

