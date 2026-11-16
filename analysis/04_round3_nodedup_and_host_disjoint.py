import os as _os
_H = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.dirname(_H); _OUT = _os.path.join(_H, "out"); _os.makedirs(_OUT, exist_ok=True)
import numpy as np, json, glob, os
from sklearn.metrics import roc_auc_score, roc_curve
R3=_ROOT+"/results_r3/runs/"; R1=_ROOT+"/results/runs/"
def thr(pv,yv,a=0.01):
    neg=np.sort(pv[yv==0]); k=int(np.floor(a*len(neg))); return float(np.nextafter(neg[len(neg)-k-1],1.0))
def met(y,p,t):
    pr=p>=t; tp=(pr&(y==1)).sum(); fp=(pr&(y==0)).sum(); fn=((~pr)&(y==1)).sum(); tn=((~pr)&(y==0)).sum()
    return dict(f1=2*tp/max(2*tp+fp+fn,1),fpr=fp/max(fp+tn,1),rec=tp/max(tp+fn,1),acc=(tp+tn)/len(y))
def ece(y,p,n=15):
    b=np.minimum((p*n).astype(int),n-1); return sum((b==i).mean()*abs(y[b==i].mean()-p[b==i].mean()) for i in range(n) if (b==i).any())
out={}
print("== A: dedup vs no-dedup (UNSW), mean over 5 seeds")
m1=np.load(_OUT+"/unsw_meta.npz",allow_pickle=True); m2=np.load(R3+"unsw_nodedup/meta.npz",allow_pickle=True)
for mod in ("xgboost","rf","proposed","ft_transformer"):
    for tag,root,mm in (("dedup",R1+"unsw",m1),("nodedup",R3+"unsw_nodedup",m2)):
        yv,yt=mm["y_val"],mm["y_test"]; rows=[]
        for s in range(42,47):
            d=np.load(f"{root}/{mod}/seed{s}/preds.npz"); pv,pt=d["p_val"],d["p_test"]; t=thr(pv,yv)
            a=met(yv,pv,.5); b=met(yt,pt,.5); c=met(yt,pt,t); fpr,tpr,_=roc_curve(yt,pt)
            rows.append([roc_auc_score(yv,pv),a["acc"],a["fpr"],roc_auc_score(yt,pt),b["f1"],b["fpr"],t,c["f1"],c["fpr"],c["rec"],np.interp(0.01,fpr,tpr),ece(yv,pv),ece(yt,pt)])
        r=np.array(rows); mu=r.mean(0); sd=r.std(0,ddof=1); out[f"A:{mod}:{tag}"]=[mu.tolist(),sd.tolist()]
        print(f"{mod:15s}{tag:8s} valAUC {mu[0]:.4f} valAcc {100*mu[1]:.2f} valFPR@.5 {100*mu[2]:.2f} | testAUC {mu[3]:.4f} F1@.5 {100*mu[4]:.2f} FPR@.5 {100*mu[5]:.2f}±{100*sd[5]:.2f} | thr {mu[6]:.3f} F1 {100*mu[7]:.2f} FPR {100*mu[8]:.2f}±{100*sd[8]:.2f} R {100*mu[9]:.2f} | TPR@1% {100*mu[10]:.2f} | ECE val {mu[11]:.3f} test {mu[12]:.3f}")
print("\n== B: TON_IoT host-disjoint")
rng=np.random.default_rng(0); MODS=["rf","xgboost","mlp","ft_transformer","proposed"]
allr={}
for k in range(5):
    mm=np.load(R3+f"toniot_host{k}/meta.npz",allow_pickle=True); yv,yt=mm["y_val"],mm["y_test"]; hosts=mm["host_test"]
    print(f"fold{k}: test n={len(yt)} benign={int((yt==0).sum())} attack={int(yt.sum())} hosts={len(set(hosts))} benign hosts={len(set(hosts[yt==0]))}")
    for mod in MODS:
        rows=[]
        for s in (42,43,44):
            d=np.load(R3+f"toniot_host{k}/{mod}/seed{s}/preds.npz"); pv,pt=d["p_val"],d["p_test"]; t=thr(pv,yv)
            a=met(yv,pv,.5); b=met(yt,pt,.5); c=met(yt,pt,t); neg=np.where(yt==0)[0]; rc=[]
            for rep in range(20):
                n=1000; ci=rng.choice(neg,n,replace=False); mk=np.ones(len(yt),bool); mk[ci]=False
                q=np.sort(pt[ci]); t2=float(np.nextafter(q[n-int(np.floor(0.01*n))-1],1.0)); x=met(yt[mk],pt[mk],t2); rc.append([x["fpr"],x["rec"],x["f1"]])
            rc=np.mean(rc,0)
            rows.append([a["fpr"],roc_auc_score(yt,pt),b["f1"],b["fpr"],b["rec"],c["f1"],c["fpr"],c["rec"],rc[0],rc[1],rc[2],ece(yv,pv),ece(yt,pt),a["f1"]])
        mu=np.mean(rows,0); allr[(k,mod)]=mu
        print(f"   {mod:15s} valFPR@.5 {100*mu[0]:.2f} valF1 {100*mu[13]:.2f} | AUC {mu[1]:.4f} F1@.5 {100*mu[2]:.2f} FPR@.5 {100*mu[3]:.2f} R {100*mu[4]:.2f} | budget: F1 {100*mu[5]:.2f} FPR {100*mu[6]:.2f} R {100*mu[7]:.2f} | recal: FPR {100*mu[8]:.2f} R {100*mu[9]:.2f} F1 {100*mu[10]:.2f} | ECE {mu[11]:.3f}/{mu[12]:.3f}")
print("\nmean over folds / median / max of test FPR")
for mod in MODS:
    a=np.array([allr[(k,mod)] for k in range(5)]); out["B:"+mod]=a.tolist()
    print(f"{mod:15s} valFPR@.5 {100*a[:,0].mean():.2f} | AUC {a[:,1].mean():.4f} F1@.5 {100*a[:,2].mean():.2f} FPR@.5 mean {100*a[:,3].mean():.2f} med {100*np.median(a[:,3]):.2f} max {100*a[:,3].max():.2f} | budget FPR mean {100*a[:,6].mean():.2f} med {100*np.median(a[:,6]):.2f} R {100*a[:,7].mean():.2f} | recal FPR mean {100*a[:,8].mean():.2f} max {100*a[:,8].max():.2f} R mean {100*a[:,9].mean():.2f}")
json.dump(out,open(_OUT+"/r3_ab.json","w"))
