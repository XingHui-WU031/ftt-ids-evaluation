import os as _os
_H = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.dirname(_H); _OUT = _os.path.join(_H, "out"); _os.makedirs(_OUT, exist_ok=True)
import numpy as np, json
from sklearn.metrics import roc_auc_score, roc_curve
R3=_ROOT+"/results_r3/"; R1=_ROOT+"/results/runs/"
m=np.load(_OUT+"/unsw_meta.npz",allow_pickle=True); yv,yt=m["y_val"],m["y_test"]
def thr(pv,a=0.01):
    neg=np.sort(pv[yv==0]); k=int(np.floor(a*len(neg))); return float(np.nextafter(neg[len(neg)-k-1],1.0))
def row(root,mod,seeds):
    r=[]
    for s in seeds:
        d=np.load(f"{root}/{mod}/seed{s}/preds.npz"); pv,pt=d["p_val"],d["p_test"]; fpr,tpr,_=roc_curve(yt,pt)
        r.append([roc_auc_score(yt,pt),np.interp(0.01,fpr,tpr),((pt>=.5)&(yt==0)).sum()/(yt==0).sum(),roc_auc_score(yv,pv)])
    r=np.array(r); return r.mean(0),r.std(0,ddof=1)
out={}
print("== D ROAR (seeds 42-44): test AUC, TPR@1%FPR, FPR@0.5, val AUC")
S=(42,43,44)
for mod in ("proposed","xgboost"):
    for tag,root in (("full",R1+"unsw"),("top3 (TTL)",R1+"unsw_nottl"),("top5",R3+"runs/unsw_roar_top5"),("rnd5a",R3+"runs/unsw_roar_rnd5a"),("rnd5b",R3+"runs/unsw_roar_rnd5b"),("top10",R3+"runs/unsw_roar_top10"),("rnd10a",R3+"runs/unsw_roar_rnd10a"),("rnd10b",R3+"runs/unsw_roar_rnd10b")):
        mu,sd=row(root,mod,S); out[f"{mod}:{tag}"]=[mu.tolist(),sd.tolist()]
        print(f"{mod:9s}{tag:12s} AUC {mu[0]:.4f}±{sd[0]:.4f} TPR@1% {100*mu[1]:.2f}±{100*sd[1]:.2f} FPR@.5 {100*mu[2]:.2f} valAUC {mu[3]:.4f}")
print("\n== C explanation")
s=json.load(open(R3+"explain_r3/unsw/proposed/summary_r3.json"))
for k,v in s["ig"].items():
    fa=v["mask_auc"]; print(k,"IG add.rel %.2f%%"%(100*v["additivity_rel"]),"vs EG500",{a:round(b,3) for a,b in v["vs_eg500"].items()},"signed r %.3f"%v["signed_corr_per_record"],"| mask AUC:",{a:round(b,3) for a,b in fa.items() if a in("none","ig_top5","random5","ig_top10","random10")})
print("IG stability across seeds",s["ig_stability_across_seeds"])
g=sorted(s["ig_global_mean"].items(),key=lambda x:-x[1])[:10]; print("IG global top10",[(a,round(b,2)) for a,b in g])
print("EG2000",{k:(round(100*v["additivity_rel"],2),v["vs_eg500"],v["vs_ig"]) for k,v in s["eg"].items()})
import numpy as np
fp=[];tp=[]
for k,v in s["ig"].items():
    b=v["by_outcome"]; print(k,"FP n",b["FP"]["n"],{a:round(x,2) for a,x in list(b["FP"]["mean"].items())[:5]},"| TP",{a:round(x,2) for a,x in list(b["TP"]["mean"].items())[:5]},"| TN",{a:round(x,2) for a,x in list(b["TN"]["mean"].items())[:3]})
m5=np.mean([[v["mask_auc"][f"ig_top{k}"] for k in (1,2,3,5,10)]+[v["mask_auc"][f"random{k}"] for k in (1,2,3,5,10)] for v in s["ig"].values()],0); print("mask AUC mean over seeds: IG top1,2,3,5,10:",m5[:5].round(3),"random:",m5[5:].round(3))
out["ig_rel"]=[v["additivity_rel"] for v in s["ig"].values()]
json.dump(out,open(_OUT+"/r3_cd.json","w"))
