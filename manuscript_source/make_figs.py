import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, precision_recall_curve
R="/home/claude/paper/res"; F="/home/claude/paper/figs"
C=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#6250d6","#e34948"]
INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3df"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":INK2,"axes.labelcolor":INK,
  "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,
  "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,"legend.frameon":False,"savefig.dpi":300})
LAB={"rf":"RF","xgboost":"XGBoost","mlp":"MLP","cnn1d":"1D-CNN","ft_transformer":"FT-Transformer","proposed":"FTT-IDS (ours)"}
MAIN=["rf","xgboost","mlp","cnn1d","ft_transformer","proposed"]

# ---------- Fig: framework ----------
def box(ax,x,y,w,h,t,fc="#eef4fc",ec=C[0],fs=6.8):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.01,rounding_size=0.02",fc=fc,ec=ec,lw=1))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=INK)
fig,ax=plt.subplots(figsize=(7.6,3.2)); ax.axis("off"); ax.set_xlim(0,1); ax.set_ylim(0,1)
stages=[("Stage I\nData protocol",["Exact-duplicate removal\n(training file only)","Stratified train/val\nsplit (official test\nkept intact)","log1p + standardisation;\ncategorical index coding\n(fit on training split)"]),
        ("Stage II\nModels (5 seeds)",["Feature-token Transformer\n(per-feature + categorical\nembeddings)","Baselines: RF, XGBoost,\nMLP, 1D-CNN,\nFT-Transformer","Validation-only tuning\n& early stopping"]),
        ("Stage III\nDecision rule",["Scores on validation set","Threshold at a\nvalidation false-alarm\nbudget (1%)","Test metrics +\nMcNemar / paired\nbootstrap"]),
        ("Stage IV\nExplanation audit",["Expected-gradients\nSHAP on the\nattack margin","Stability, faithfulness,\npermutation agreement","Error analysis by\nTP / TN / FP / FN"])]
for i,(h,items) in enumerate(stages):
    x=0.01+i*0.25
    ax.text(x+0.115,0.95,h,ha="center",va="top",fontsize=8,fontweight="bold",color=INK)
    for j,t in enumerate(items): box(ax,x,0.62-j*0.27,0.23,0.21,t,fc=["#eef4fc","#fdf0ea","#e9f7f1","#fdf5e3"][i],ec=C[[0,1,2,3][i]])
    if i<3: ax.annotate("",xy=(x+0.255,0.45),xytext=(x+0.235,0.45),arrowprops=dict(arrowstyle="-|>",color=INK2,lw=1))
fig.savefig(f"{F}/fig1_framework.png",bbox_inches="tight"); plt.close()

# ---------- Fig: architecture ----------
fig,ax=plt.subplots(figsize=(8.0,2.6)); ax.axis("off"); ax.set_xlim(-0.01,1.02); ax.set_ylim(0,1)
box(ax,0.0,0.55,0.13,0.3,"Numeric\nfeatures\nx₁…x_m")
box(ax,0.0,0.12,0.13,0.3,"Categorical\nfeatures\nc₁…c_k")
box(ax,0.17,0.55,0.15,0.3,"Per-feature\nlinear map\ntᵢ = xᵢwᵢ + bᵢ",fc="#fdf0ea",ec=C[1])
box(ax,0.17,0.12,0.15,0.3,"Embedding\ntables\ntⱼ = Eⱼ[cⱼ]",fc="#fdf0ea",ec=C[1])
box(ax,0.36,0.3,0.13,0.4,"Token\nsequence\n(m+k)×d",fc="#e9f7f1",ec=C[2])
box(ax,0.53,0.3,0.17,0.4,"Transformer\nencoder\n2 layers,\n4 heads, d=64",fc="#eef4fc",ec=C[0])
box(ax,0.74,0.3,0.1,0.4,"Mean\npooling",fc="#e9f7f1",ec=C[2])
box(ax,0.88,0.3,0.12,0.4,"Linear\nhead →\np(attack)",fc="#fdf5e3",ec=C[3])
for (x1,y1,x2,y2) in [(0.13,0.7,0.17,0.7),(0.13,0.27,0.17,0.27),(0.32,0.7,0.36,0.55),(0.32,0.27,0.36,0.45),(0.49,0.5,0.53,0.5),(0.70,0.5,0.74,0.5),(0.84,0.5,0.88,0.5)]:
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="-|>",color=INK2,lw=1))
fig.savefig(f"{F}/fig2_architecture.png",bbox_inches="tight"); plt.close()

# ---------- Fig: threshold rules (UNSW) ----------
L=pd.read_csv(f"{R}/tables/unsw/all_runs_long.csv")
rules=[("default","0.5"),("val_f1","Val. max-F1"),("val_youden","Val. Youden"),("val_fpr01","Val. FPR ≤ 1%")]
fig,axs=plt.subplots(1,2,figsize=(7.2,2.8))
for k,(met,lab) in enumerate([("fpr","Test false-positive rate (%)"),("f1","Test F1 (%)")]):
    ax=axs[k]; w=0.2
    for i,(r,rl) in enumerate(rules):
        g=L[L.threshold_rule==r].groupby("model")[met]
        mu=[100*g.mean()[m] for m in MAIN]; sd=[100*g.std()[m] for m in MAIN]
        ax.bar(np.arange(6)+(i-1.5)*w,mu,w*0.9,yerr=sd,color=C[i],label=rl,error_kw=dict(lw=0.6,ecolor=INK2,capsize=1.5))
    ax.set_xticks(range(6)); ax.set_xticklabels([LAB[m] for m in MAIN],rotation=30,ha="right"); ax.set_ylabel(lab)
    if met=="f1": ax.set_ylim(84,96)
fig.legend(*axs[0].get_legend_handles_labels(),fontsize=7.5,ncol=4,loc="upper center",bbox_to_anchor=(0.5,1.06))
fig.tight_layout(); fig.savefig(f"{F}/fig3_threshold_rules.png",bbox_inches="tight"); plt.close()

# ---------- Fig: ROC/PR with operating points ----------
y=np.load(f"{R}/runs/unsw/y_test.npy")
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0))
for i,m in enumerate(MAIN):
    p=np.load(f"{R}/runs/unsw/{m}/seed42/preds.npz")["p_test"]
    fpr,tpr,_=roc_curve(y,p); pr,rc,_=precision_recall_curve(y,p)
    axs[0].plot(fpr,tpr,color=C[i],lw=1.4 if m=="proposed" else 1,label=LAB[m])
    axs[1].plot(rc,pr,color=C[i],lw=1.4 if m=="proposed" else 1)
met=json.load(open(f"{R}/runs/unsw/proposed/seed42/metrics.json")) if False else None
for rule,mk,lab in [("default","o","threshold 0.5"),("val_fpr01","s","validation FPR ≤ 1%")]:
    r=L[(L.model=="proposed")&(L.seed==42)&(L.threshold_rule==rule)].iloc[0]
    axs[0].plot(r.fpr,r.recall,mk,ms=7,mfc="white",mec=INK,mew=1.2,label=f"FTT-IDS @ {lab}")
    axs[1].plot(r.recall,r.precision,mk,ms=7,mfc="white",mec=INK,mew=1.2)
axs[0].set_xlabel("False-positive rate"); axs[0].set_ylabel("True-positive rate"); axs[0].set_xlim(0,0.5); axs[0].set_ylim(0.5,1.005)
axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[1].set_ylim(0.75,1.005); axs[1].set_xlim(0.5,1.0)
axs[0].legend(fontsize=6.5,loc="lower right")
fig.tight_layout(); fig.savefig(f"{F}/fig4_roc_pr.png",bbox_inches="tight"); plt.close()

# ---------- Fig: TTL shift ----------
tr=pd.read_csv("/mnt/user-data/uploads/Project/数据集/UNSW_NB15_training-set.csv").drop(columns=["id"]).drop_duplicates()
te=pd.read_csv("/mnt/user-data/uploads/Project/数据集/UNSW_NB15_testing-set.csv")
vals=[31,62,254]; fig,ax=plt.subplots(figsize=(7.2,2.4)); w=0.2
groups=[("Train, normal",tr[tr.label==0]),("Train, attack",tr[tr.label==1]),("Test, normal",te[te.label==0]),("Test, attack",te[te.label==1])]
for i,(g,d) in enumerate(groups):
    sh=[100*(d.sttl==v).mean() for v in vals]+[100*(~d.sttl.isin(vals)).mean()]
    ax.bar(np.arange(4)+(i-1.5)*w,sh,w*0.9,color=C[i],label=g)
ax.set_xticks(range(4)); ax.set_xticklabels(["sttl = 31","sttl = 62","sttl = 254","other"]); ax.set_ylabel("Share of records (%)"); ax.set_ylim(0,105); ax.legend(ncol=4,fontsize=7,loc="upper center",bbox_to_anchor=(0.5,1.15))
fig.tight_layout(); fig.savefig(f"{F}/fig5_sttl_shift.png",bbox_inches="tight"); plt.close()

# ---------- Fig: global SHAP both datasets ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,3.4))
for ax,(d,t) in zip(axs,[("unsw","UNSW-NB15"),("toniot","TON_IoT")]):
    g=pd.read_csv(f"{R}/explain/{d}/proposed/global_importance.csv").head(12)[::-1]
    pm=pd.read_csv(f"{R}/explain/{d}/proposed/permutation_importance.csv").set_index("feature")
    ax.barh(g.feature,g.mean_abs_shap,xerr=g.std_over_seeds,color=C[0],height=0.65,error_kw=dict(lw=0.6,ecolor=INK2,capsize=1.5))
    ax.set_title(t,fontsize=9,color=INK); ax.set_xlabel("Mean |SHAP| on attack margin")
    ax.grid(axis="y",visible=False)
fig.tight_layout(); fig.savefig(f"{F}/fig6_global_shap.png",bbox_inches="tight"); plt.close()

# ---------- Fig: faithfulness ----------
fig,axs=plt.subplots(1,2,figsize=(7.2,2.7))
for ax,(d,t) in zip(axs,[("unsw","UNSW-NB15"),("toniot","TON_IoT")]):
    s=pd.read_csv(f"{R}/explain/{d}/proposed/faithfulness_global_summary.csv")
    base=s[s.kind=="none"]
    for i,(k,lab) in enumerate([("shap_topk","Mask SHAP top-k"),("random_k","Mask random k")]):
        q=pd.concat([base,s[s.kind==k]]).sort_values("k")
        ax.errorbar(q.k,q.auc_mean,yerr=q.auc_std,color=C[i],marker="o",ms=4,lw=1.4,capsize=2,label=lab)
    ax.set_title(t,fontsize=9); ax.set_xlabel("Number of features masked (k)"); ax.set_ylabel("Test ROC-AUC")
axs[0].legend(fontsize=7,loc="lower left")
fig.tight_layout(); fig.savefig(f"{F}/fig7_faithfulness.png",bbox_inches="tight"); plt.close()

# ---------- Fig: SHAP by outcome (UNSW) ----------
oc=pd.read_csv(f"{R}/explain/unsw/proposed/shap_by_outcome.csv").set_index("group").loc[["TP","FN","FP","TN"]]
hm=oc[[c for c in oc.columns if c.startswith("meanSHAP_")]]; hm.columns=[c[9:] for c in hm.columns]
import matplotlib.colors as mc
cmap=mc.LinearSegmentedColormap.from_list("div",["#2a78d6","#f2f1ee","#e34948"])
fig,ax=plt.subplots(figsize=(7.2,2.3)); v=np.abs(hm.values).max()
im=ax.imshow(hm.values,cmap=cmap,vmin=-v,vmax=v,aspect="auto"); ax.grid(False)
ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels(hm.columns,rotation=35,ha="right")
ax.set_yticks(range(4)); ax.set_yticklabels([f"{g} (n={int(oc.loc[g,'n_total'])})" for g in hm.index])
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]): ax.text(j,i,f"{hm.values[i,j]:+.2f}",ha="center",va="center",fontsize=6.5,color=INK)
cb=fig.colorbar(im,ax=ax,fraction=0.025); cb.set_label("Mean SHAP (+ → attack)",fontsize=7)
fig.tight_layout(); fig.savefig(f"{F}/fig8_shap_by_outcome.png",bbox_inches="tight"); plt.close()
print("ok")
