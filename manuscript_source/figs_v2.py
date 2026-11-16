import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve
R="/home/claude/bundle"; F="/home/claude/paper/figs"
C=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#6250d6","#e34948"]
INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3df"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":INK2,"axes.labelcolor":INK,
  "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,
  "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,"legend.frameon":False,"savefig.dpi":300})
LAB={"rf":"RF","xgboost":"XGBoost","mlp":"MLP","cnn1d":"1D-CNN","ft_transformer":"FT-Transformer","proposed":"FTT-IDS"}
MAIN=["rf","xgboost","mlp","cnn1d","ft_transformer","proposed"]
def box(ax,x,y,w,h,t,fc,ec,fs=6.2):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.005,rounding_size=0.015",fc=fc,ec=ec,lw=1))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=INK)
fig,ax=plt.subplots(figsize=(7.6,3.3)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
stages=[("Stage I\nData protocol",["Exact-duplicate removal\nbefore splitting","Stratified train/validation\nsplit; official test set\nkept intact","Preprocessing statistics\nfitted on the training\nsplit only"]),
        ("Stage II\nModels (5 seeds)",["Feature-token Transformer\n(FTT-IDS) and\nfive ablations","Baselines: RF, XGBoost,\nMLP, 1D-CNN,\nFT-Transformer","Validation-only tuning\nand early stopping"]),
        ("Stage III\nOperating point",["Threshold rules set on\nvalidation data\n(0.5, F1, Youden, budget)","Budget sweep and\nmatched-FPR comparison","Benign-only\nrecalibration\nunder shift"]),
        ("Stage IV\nExplanation audit",["Expected-gradients\nSHAP on the\nattack margin","Stability, faithfulness,\npermutation agreement","Error analysis by\noutcome; TTL-removal\nexperiment"])]
W=0.215; G=0.04
for i,(h,items) in enumerate(stages):
    x=0.005+i*(W+G)
    ax.text(x+W/2,0.97,h,ha="center",va="top",fontsize=8,fontweight="bold",color=INK)
    for j,t in enumerate(items): box(ax,x,0.60-j*0.27,W,0.22,t,fc=["#eef4fc","#fdf0ea","#e9f7f1","#fdf5e3"][i],ec=C[i])
    if i<3: ax.annotate("",xy=(x+W+G-0.008,0.44),xytext=(x+W+0.008,0.44),arrowprops=dict(arrowstyle="-|>",color=INK2,lw=1.2))
fig.savefig(f"{F}/fig1_framework.png",bbox_inches="tight"); plt.close()

L=pd.read_csv(f"{R}/tables/unsw/all_runs_long.csv")
rules=[("default","0.5"),("val_f1","Val. max-F1"),("val_youden","Val. Youden"),("val_fpr01","Val. FPR ≤ 1%")]
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
for k,(met,lab) in enumerate([("fpr","Test false-positive rate (%)"),("f1","Test F1 (%)")]):
    ax=axs[k]; w=0.2
    for i,(r,rl) in enumerate(rules):
        g=L[L.threshold_rule==r].groupby("model")[met]
        mu=[100*g.mean()[m] for m in MAIN]; sd=[100*g.std()[m] for m in MAIN]
        ax.bar(np.arange(6)+(i-1.5)*w,mu,w*0.9,yerr=sd,color=C[i],label=rl,error_kw=dict(lw=0.6,ecolor=INK2,capsize=1.5))
    ax.set_xticks(range(6)); ax.set_xticklabels([LAB[m] for m in MAIN],rotation=30,ha="right"); ax.set_ylabel(lab); ax.grid(axis="x",visible=False)
    if met=="f1": ax.set_ylim(84,96)
fig.legend(*axs[0].get_legend_handles_labels(),fontsize=7.5,ncol=4,loc="upper center",bbox_to_anchor=(0.5,1.06))
fig.tight_layout(); fig.savefig(f"{F}/fig3_threshold_rules.png",bbox_inches="tight"); plt.close()

y=np.load(f"{R}/runs/unsw/y_test.npy")
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0))
for i,m in enumerate(MAIN):
    p=np.load(f"{R}/runs/unsw/{m}/seed42/preds.npz")["p_test"]
    fpr,tpr,_=roc_curve(y,p); pr,rc,_=precision_recall_curve(y,p)
    axs[0].plot(fpr,tpr,color=C[i],lw=1.6 if m=="proposed" else 1,label=LAB[m])
    axs[1].plot(rc,pr,color=C[i],lw=1.6 if m=="proposed" else 1)
for rule,mk,lab in [("default","o","threshold 0.5"),("val_fpr01","s","validation budget 1%")]:
    r=L[(L.model=="proposed")&(L.seed==42)&(L.threshold_rule==rule)].iloc[0]
    axs[0].plot(r.fpr,r.recall,mk,ms=7,mfc="white",mec=INK,mew=1.2,label=f"FTT-IDS @ {lab}")
    axs[1].plot(r.recall,r.precision,mk,ms=7,mfc="white",mec=INK,mew=1.2)
axs[0].set_xlabel("False-positive rate"); axs[0].set_ylabel("True-positive rate"); axs[0].set_xlim(0,0.5); axs[0].set_ylim(0.5,1.005)
axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_ylim(0.75,1.005); axs[1].set_xlim(0.5,1.0)
axs[0].legend(fontsize=6.5,loc="lower right")
fig.tight_layout(); fig.savefig(f"{F}/fig4_roc_pr.png",bbox_inches="tight"); plt.close()

# ---- new: budget transfer + recalibration
o=json.load(open("/home/claude/analysis/out.json")); A=[0.1,0.5,1,2,5,10]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0))
ax=axs[0]
ax.plot([0.08,40],[0.08,40],color=INK2,lw=0.9,ls="--"); ax.text(11,8.2,"test = budget",fontsize=6.5,color=INK2,rotation=37)
for i,m in enumerate(MAIN):
    v=[r[0] for r in o["sweep"][m]]
    ax.plot(A,v,color=C[i],marker="o",ms=3.5,lw=1.6 if m=="proposed" else 1,label=LAB[m])
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks(A); ax.set_xticklabels([str(a) for a in A]); ax.set_yticks([0.1,0.5,1,2,5,10,30]); ax.set_yticklabels(["0.1","0.5","1","2","5","10","30"])
ax.minorticks_off(); ax.set_xlim(0.08,13); ax.set_ylim(0.08,40)
ax.set_xlabel("Validation false-alarm budget α (%)"); ax.set_ylabel("Realised test FPR (%)"); ax.legend(fontsize=6.5,loc="lower right")
ax=axs[1]; N=[200,500,1000,2000,5000]
ax.axhline(1.0,color=INK2,lw=0.9,ls="--"); ax.text(3300,0.42,"target 1%",fontsize=6.5,color=INK2,ha="center",va="center")
for m,i,dx in (("xgboost",1,0.96),("proposed",5,1.04)):
    mu=[o["recal"][m][f"{n}_0.01"][0][0] for n in N]; sd=[o["recal"][m][f"{n}_0.01"][1][0] for n in N]
    ax.errorbar([n*dx for n in N],mu,yerr=sd,color=C[i],marker="o",ms=4,lw=1.4,capsize=2,label=LAB[m]+", recalibrated")
ax.axhline(4.44,color=C[5],lw=1,ls=":"); ax.text(210,4.5,"FTT-IDS, validation budget (4.44%)",fontsize=6.5,color=INK2,va="bottom")
ax.axhline(5.81,color=C[1],lw=1,ls=":"); ax.text(210,5.87,"XGBoost, validation budget (5.81%)",fontsize=6.5,color=INK2,va="bottom")
ax.set_xscale("log"); ax.set_xticks(N); ax.set_xticklabels([str(n) for n in N]); ax.minorticks_off(); ax.set_ylim(0,6.8)
ax.set_xlabel("Benign target-network flows used for recalibration (n)"); ax.set_ylabel("Realised test FPR (%)"); ax.legend(fontsize=6.5,loc="center right")
fig.tight_layout(); fig.savefig(f"{F}/fig5_budget_transfer.png",bbox_inches="tight"); plt.close()
print("ok")
