import os as _os
_H = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.dirname(_H); _OUT = _os.path.join(_H, "out"); _os.makedirs(_OUT, exist_ok=True)
import sys as _sys
_sys.path.insert(0, _ROOT)
from src.thresholds import benign_budget_threshold, labelled_budget_threshold
import numpy as np, json
from scipy.stats import chi2
R=_ROOT+"/results/runs/"; m=np.load(_OUT+"/unsw_meta.npz",allow_pickle=True); yv,yt=m["y_val"],m["y_test"]; S=range(42,47)
def load(mod,s):
    d=np.load(f"{R}unsw/{mod}/seed{s}/preds.npz"); return d["p_val"],d["p_test"]
def thr_fpr(pv,a=0.01):
    return labelled_budget_threshold(yv, pv, a)
def thr_f1(pv):
    from sklearn.metrics import precision_recall_curve
    p,r,t=precision_recall_curve(yv,pv); f=2*p[:-1]*r[:-1]/np.clip(p[:-1]+r[:-1],1e-12,None); return float(t[int(np.argmax(f))])
def f1(y,pr):
    tp=(pr&(y==1)).sum(); fp=(pr&(y==0)).sum(); fn=((~pr)&(y==1)).sum(); return 2*tp/(2*tp+fp+fn)
def mcn(pa,pb):
    ca,cb=pa==yt,pb==yt; b=int((ca&~cb).sum()); c=int((~ca&cb).sum())
    return 1.0 if b+c==0 else float(chi2.sf((abs(b-c)-1)**2/(b+c),1))
rng=np.random.default_rng(0); n=len(yt); idx=[rng.integers(0,n,n) for _ in range(1000)]
out={}
for rule,tf in (("fpr01",thr_fpr),("f1",thr_f1)):
    P={mod:[load(mod,s)[1]>=tf(load(mod,s)[0]) for s in S] for mod in ["rf","xgboost","mlp","cnn1d","ft_transformer","proposed","thesis_ftt","single_token"]}
    for mod in P:
        if mod=="proposed": continue
        ps=[mcn(P["proposed"][i].astype(int),P[mod][i].astype(int)) for i in range(5)]
        d=np.array([np.mean([f1(yt[i],P["proposed"][s][i])-f1(yt[i],P[mod][s][i]) for s in range(5)]) for i in idx])
        pt=np.mean([f1(yt,P["proposed"][s])-f1(yt,P[mod][s]) for s in range(5)])
        out[f"{rule}:{mod}"]=dict(p_med=float(np.median(ps)),n_sig=int(sum(p<0.05 for p in ps)),d=100*pt,lo=100*np.percentile(d,2.5),hi=100*np.percentile(d,97.5))
        print(rule,mod,"McNemar med %.1e sig %d/5 | dF1 %+.2f [%+.2f, %+.2f]"%(np.median(ps),sum(p<0.05 for p in ps),100*pt,100*np.percentile(d,2.5),100*np.percentile(d,97.5)))
json.dump(out,open(_OUT+"/boot1000.json","w"),indent=1)
