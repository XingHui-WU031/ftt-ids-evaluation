import os as _os
_H = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.dirname(_H); _OUT = _os.path.join(_H, "out"); _os.makedirs(_OUT, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _ROOT)
from src.thresholds import benign_budget_threshold, labelled_budget_threshold
import numpy as np
R=_ROOT+"/results/runs/"
m=np.load(_OUT+"/unsw_meta.npz",allow_pickle=True)
yv,yt=m["y_val"],m["y_test"]; cat=m["cat_test"]; st=m["sttl_test"]; state=m["state_test"]; svc=m["service_test"]
SEEDS=range(42,47)
def load(ds,mod,s):
    d=np.load(f"{R}{ds}/{mod}/seed{s}/preds.npz"); return d["p_val"],d["p_test"]
def thr(pv,a=0.01):
    return labelled_budget_threshold(yv, pv, a)
def f1(y,pr):
    tp=(pr&(y==1)).sum(); fp=(pr&(y==0)).sum(); fn=((~pr)&(y==1)).sum(); return 2*tp/(2*tp+fp+fn)
P={mod:np.array([load("unsw",mod,s)[1]>=thr(load("unsw",mod,s)[0]) for s in SEEDS]) for mod in ["rf","xgboost","mlp","cnn1d","ft_transformer","proposed","thesis_ftt","single_token"]}
rng=np.random.default_rng(1); n=len(yt)
idx=[rng.integers(0,n,n) for _ in range(500)]
for mod in P:
    if mod=="proposed": continue
    d=[np.mean([f1(yt[i],P["proposed"][s][i])-f1(yt[i],P[mod][s][i]) for s in range(5)]) for i in idx]
    print(mod,"dF1 %.2f [%.2f, %.2f]"%(100*np.mean(d),100*np.percentile(d,2.5),100*np.percentile(d,97.5)))
# FP profile at 0.5
fin=[];nos=[];s254=[];jac=[];s254n=[]
for s in SEEDS:
    pt=load("unsw","proposed",s)[1]; fp=(pt>=.5)&(yt==0)
    fin.append((state[fp]=="FIN").mean()); nos.append((svc[fp]=="-").mean()); s254.append((st[fp]==254).mean())
    pn=load("unsw_nottl","proposed",s)[1]; fpn=(pn>=.5)&(yt==0)
    jac.append((fp&fpn).sum()/(fp|fpn).sum()); s254n.append((st[fpn]==254).mean())
print("FP@0.5: sttl254 %.1f FIN %.1f noservice %.1f | jaccard full vs nottl %.3f | nottl FP sttl254 %.1f"%(100*np.mean(s254),100*np.mean(fin),100*np.mean(nos),np.mean(jac),100*np.mean(s254n)))
# per category
for c in np.unique(cat):
    k=cat==c
    print(c,k.sum()," ".join("%.2f"%(100*np.mean([ (load("unsw",mod,s)[1][k]>=t(load("unsw",mod,s)[0])).mean() for s in SEEDS])) for mod in("proposed","xgboost") for t in (lambda pv:0.5, thr)))
# FP removed
print(np.mean([((load("unsw","proposed",s)[1]>=.5)&(yt==0)).sum() for s in SEEDS]), np.mean([(P["proposed"][i]&(yt==0)).sum() for i in range(5)]))
# seed-ensemble
for mod in ("proposed","xgboost"):
    pv=np.mean([load("unsw",mod,s)[0] for s in SEEDS],0); pt=np.mean([load("unsw",mod,s)[1] for s in SEEDS],0)
    pr=pt>=thr(pv); print(mod,"ensemble tauFPR F1 %.2f FPR %.2f"%(100*f1(yt,pr),100*(pr&(yt==0)).sum()/(yt==0).sum()))
