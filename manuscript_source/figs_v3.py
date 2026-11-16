import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
F="/home/claude/paper/figs"
C=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#6250d6","#e34948"]
INK="#0b0b0b"; INK2="#52514e"; GRID="#e4e3df"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":INK2,"axes.labelcolor":INK,
  "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,
  "axes.grid":True,"grid.color":GRID,"grid.linewidth":0.6,"axes.axisbelow":True,"legend.frameon":False,"savefig.dpi":300})
o=json.load(open("/home/claude/analysis/r3_ab.json"))
M=[("rf","RF",0),("xgboost","XGBoost",1),("mlp","MLP",2),("ft_transformer","FT-Transformer",4),("proposed","FTT-IDS",5)]
fig,axs=plt.subplots(1,3,figsize=(7.4,2.9)); w=0.16
for ax,(col,lab) in zip(axs,[(3,"Test FPR at τ = 0.5 (%)"),(8,"Test FPR after recalibration (%)"),(9,"Recall after recalibration (%)")]):
    for i,(k,name,ci) in enumerate(M):
        a=np.array(o["B:"+k]); ax.bar(np.arange(5)+(i-2)*w,100*a[:,col],w*0.88,color=C[ci],label=name)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{j+1}" for j in range(5)]); ax.set_xlabel("Held-out host group"); ax.set_ylabel(lab); ax.grid(axis="x",visible=False)
axs[0].axhline(1.0,color=INK2,lw=0.9,ls="--")
axs[1].axhline(1.0,color=INK2,lw=0.9,ls="--"); axs[1].set_ylim(0,2.4); axs[1].text(4.45,1.05,"target 1%",fontsize=6.3,color=INK2,ha="right",va="bottom")
axs[2].set_ylim(0,105)
fig.legend(*axs[0].get_legend_handles_labels(),fontsize=7.2,ncol=5,loc="upper center",bbox_to_anchor=(0.5,1.07))
fig.tight_layout(); fig.savefig(f"{F}/fig6_host_disjoint.png",bbox_inches="tight"); plt.close(); print("ok")
