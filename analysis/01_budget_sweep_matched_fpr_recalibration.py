import os as _os
_H = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.dirname(_H); _OUT = _os.path.join(_H, "out"); _os.makedirs(_OUT, exist_ok=True)
import numpy as np, json, sys
from sklearn.metrics import roc_curve, roc_auc_score
R=_ROOT+"/results/runs/"
m=np.load(_OUT+"/unsw_meta.npz",allow_pickle=True)
yv,yt=m["y_val"],m["y_test"]
MODELS=["rf","xgboost","mlp","cnn1d","ft_transformer","proposed","thesis_ftt","single_token"]
SEEDS=range(42,47)
def load(ds,mod,s):
    d=np.load(f"{R}{ds}/{mod}/seed{s}/preds.npz"); return d["p_val"],d["p_test"]
def thr_fpr(pv,yv,a):
    neg=np.sort(pv[yv==0]); k=int(np.floor(a*len(neg)))
    return float(np.nextafter(neg[len(neg)-k-1],1.0)) if k<len(neg) else 0.0
def met(y,p,t):
    pr=p>=t; tp=(pr&(y==1)).sum(); fp=(pr&(y==0)).sum(); fn=((~pr)&(y==1)).sum(); tn=((~pr)&(y==0)).sum()
    prec=tp/max(tp+fp,1); rec=tp/(tp+fn); f1=2*prec*rec/max(prec+rec,1e-12)
    return dict(acc=(tp+tn)/len(y),prec=prec,rec=rec,f1=f1,fpr=fp/(fp+tn))
out={}
# check reproduction
for mod in MODELS:
    r=[met(yt,load("unsw",mod,s)[1],thr_fpr(load("unsw",mod,s)[0],yv,0.01)) for s in SEEDS]
    print(mod,"fpr01: F1 %.2f FPR %.2f"%(100*np.mean([x["f1"] for x in r]),100*np.mean([x["fpr"] for x in r])))
print("\n== A. alpha sweep (mean over seeds): val-budget -> test FPR / recall / F1")
A=[0.001,0.005,0.01,0.02,0.05,0.10]
sweep={}
for mod in MODELS:
    row=[]
    for a in A:
        r=[met(yt,load("unsw",mod,s)[1],thr_fpr(load("unsw",mod,s)[0],yv,a)) for s in SEEDS]
        row.append([100*np.mean([x[k] for x in r]) for k in("fpr","rec","f1")]+[100*np.std([x["fpr"] for x in r],ddof=1)])
    sweep[mod]=row
    print(mod," | ".join("a=%.1f%%: FPR %.2f±%.2f R %.2f F1 %.2f"%(100*a,r[0],r[3],r[1],r[2]) for a,r in zip(A,row)))
out["sweep"]=sweep
print("\n== B. matched test FPR: TPR@FPR=1%,5%,10%; pAUC(FPR<=0.1) standardized")
mt={}
for mod in MODELS:
    rr=[]
    for s in SEEDS:
        pt=load("unsw",mod,s)[1]; fpr,tpr,_=roc_curve(yt,pt)
        rr.append([np.interp(0.01,fpr,tpr),np.interp(0.05,fpr,tpr),np.interp(0.10,fpr,tpr),roc_auc_score(yt,pt,max_fpr=0.1)])
    rr=np.array(rr); mt[mod]=[rr.mean(0).tolist(),rr.std(0,ddof=1).tolist()]
    print(mod," ".join("%.2f±%.2f"%(100*a,100*b) for a,b in zip(rr.mean(0),rr.std(0,ddof=1))))
out["matched"]=mt
print("\n== C. calibration ECE (15 bins) / Brier, val vs test")
def ece(y,p,n=15):
    b=np.minimum((p*n).astype(int),n-1); e=0
    for i in range(n):
        k=b==i
        if k.any(): e+=k.mean()*abs(y[k].mean()-p[k].mean())
    return e
cal={}
for mod in MODELS[:6]:
    r=[]
    for s in SEEDS:
        pv,pt=load("unsw",mod,s); r.append([ece(yv,pv),ece(yt,pt),np.mean((pv-yv)**2),np.mean((pt-yt)**2)])
    r=np.array(r).mean(0); cal[mod]=r.tolist(); print(mod,"ECE val %.3f test %.3f Brier val %.3f test %.3f"%tuple(r))
out["cal"]=cal
print("\n== D. FPR by sttl sub-population at tau_FPR(1%) and 0.5 (proposed, xgboost)")
st=m["sttl_test"]; sv=m["sttl_val"]
for mod in("proposed","xgboost"):
    acc=[]
    for s in SEEDS:
        pv,pt=load("unsw",mod,s); t=thr_fpr(pv,yv,0.01)
        nt=yt==0; nv=yv==0
        acc.append([ (pt[nt&(st==254)]>=t).mean(), (pt[nt&(st!=254)]>=t).mean(), (pv[nv&(sv==254)]>=t).mean(), (pv[nv&(sv!=254)]>=t).mean(),
                     (pt[nt&(st==254)]>=.5).mean(), (pt[nt&(st!=254)]>=.5).mean(), (pv[nv&(sv==254)]>=.5).mean(), (pv[nv&(sv!=254)]>=.5).mean(),
                     ((pt>=t)&nt&(st==254)).sum()/max(((pt>=t)&nt).sum(),1), ((pt>=.5)&nt&(st==254)).sum()/((pt>=.5)&nt).sum()])
    a=100*np.array(acc).mean(0)
    print(mod,"tauFPR: test FPR sttl254 %.2f other %.2f | val sttl254 %.2f other %.2f || 0.5: test %.2f/%.2f val %.2f/%.2f || share of FP with sttl254: tauFPR %.1f, 0.5 %.1f"%tuple(a))
    out["sub_"+mod]=a.tolist()
print("share normal sttl254: val %.1f test %.1f"%(100*(sv[yv==0]==254).mean(),100*(st[yt==0]==254).mean()))
# reweighting prediction: expected test FPR if val subpop FPRs were reweighted to test mix
print("\n== E. benign-only recalibration: threshold = (1-a) quantile of scores on n random benign test flows; evaluate on remaining test")
rng=np.random.default_rng(0)
rec={}
for mod in MODELS[:6]:
    res={}
    for n in (200,500,1000,2000,5000):
        for a in (0.01,0.05):
            r=[]
            for s in SEEDS:
                pv,pt=load("unsw",mod,s); negidx=np.where(yt==0)[0]
                for rep in range(20):
                    cal_i=rng.choice(negidx,n,replace=False); mask=np.ones(len(yt),bool); mask[cal_i]=False
                    neg=np.sort(pt[cal_i]); k=int(np.floor(a*n)); t=float(np.nextafter(neg[n-k-1],1.0))
                    x=met(yt[mask],pt[mask],t); r.append([x["fpr"],x["rec"],x["f1"],x["acc"]])
            r=np.array(r); res[f"{n}_{a}"]=[(100*r.mean(0)).tolist(),(100*r.std(0)).tolist()]
    rec[mod]=res
    print(mod," | ".join("n=%s: FPR %.2f±%.2f R %.2f±%.2f F1 %.2f"%(k,v[0][0],v[1][0],v[0][1],v[1][1],v[0][2]) for k,v in res.items() if k.endswith("0.01")))
    print("   a=5%:"," | ".join("n=%s: FPR %.2f±%.2f R %.2f F1 %.2f"%(k,v[0][0],v[1][0],v[0][1],v[0][2]) for k,v in res.items() if k.endswith("0.05")))
out["recal"]=rec
json.dump(out,open(_OUT+"/out.json","w"),indent=1)
