from pathlib import Path
import json,sys
import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--formal-outputs',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
R=Path(__file__).resolve().parents[1];F=args.formal_outputs.resolve();O=args.out.resolve();O.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R))
from src.thresholds import benign_budget_threshold
models=['rf','xgboost','mlp','ft_transformer','proposed'];allfold=[];seenrows=[];hostcal=[];design=[];attack=[]
for pol in ['within_host','none']:
 df=pd.read_csv(F/pol/'summary/strict_host_metrics.csv');g=df.groupby(['fold','model','rule'])[['fpr','recall','auc']].mean().reset_index();g['policy']=pol;allfold.append(g)
 a=pd.read_csv(F/pol/'summary/attack_coverage_and_host_fpr.csv');a=a[~a.group.str.startswith('benign_host:')];ag=a.groupby(['fold','model','group','attack_type_seen_in_training'])[['n','rate']].mean().reset_index();ag['policy']=pol;attack.append(ag)
 for f in range(5):
  ds=f'toniot_strict_host{f}'+('_nodedup' if pol=='none' else '');b=F/pol/'runs'/ds;z=np.load(b/'meta.npz');yt,yv,gt,ht=[z[k] for k in ['y_test','y_val','g_test','host_test']];au=json.loads((F/pol/'protocol'/f'{ds}_audit.json').read_text());seen=np.isin(gt,list(au['attack_counts']['train']))
  # Select a host by benign sample size only and deterministic lexical tie order, before inspecting scores.
  counts={h:int(((ht==h)&(yt==0)).sum()) for h in sorted(set(ht))};eligible=[h for h,n in counts.items() if n>=1000]
  other='none' if pol=='within_host' else 'within_host';ods=f'toniot_strict_host{f}'+('_nodedup' if other=='none' else '');oz=np.load(F/other/'runs'/ods/'meta.npz');oh,oy=oz['host_test'],oz['y_test'];eligible=[h for h in eligible if ((oh==h)&(oy==0)).sum()>=1000]
  ch=eligible[0] if eligible else None;evalmask=ht!=ch if ch else np.zeros(len(ht),bool)
  valid=ch is not None and len(np.unique(yt[evalmask]))==2
  design.append(dict(policy=pol,fold=f+1,calibration_host=ch,n_cal_host_benign=counts.get(ch,0),n_eval_benign=int(((yt==0)&evalmask).sum()),n_eval_attack=int(((yt==1)&evalmask).sum()),valid=valid))
  for mod in models:
   for seed in [42,43,44]:
    pr=np.load(b/mod/f'seed{seed}'/'preds.npz');p=pr['p_test'];sub=df[(df.fold==f+1)&(df.model==mod)&(df.seed==seed)&(df.rule=='target_benign')]
    for _,q in sub.iterrows():
     for lab,mask in [('seen',(yt==1)&seen),('unseen',(yt==1)&~seen)]:
      if mask.any():seenrows.append(dict(policy=pol,fold=f+1,model=mod,seed=seed,draw=int(q.draw),coverage=lab,n=int(mask.sum()),recall=float((p[mask]>=q.threshold).mean())))
    if valid:
     aucc=float(roc_auc_score(yt[evalmask],p[evalmask]))
     neg=np.flatnonzero((ht==ch)&(yt==0));rng=np.random.default_rng(17000+f)
     for draw in range(20):
      ci=rng.choice(neg,1000,replace=False);t=benign_budget_threshold(p[ci]);bm=(yt==0)&evalmask;am=(yt==1)&evalmask
      hostcal.append(dict(policy=pol,fold=f+1,model=mod,seed=seed,draw=draw,threshold=t,fpr=float((p[bm]>=t).mean()),recall=float((p[am]>=t).mean()),auc=aucc))
allfold=pd.concat(allfold);allfold.to_csv(O/'model_fold_policy_metrics.csv',index=False)
pd.concat(attack).to_csv(O/'attack_type_recall.csv',index=False)
pd.DataFrame(design).to_csv(O/'host_calibration_design.csv',index=False)
pd.DataFrame(seenrows).to_csv(O/'seen_unseen_attack_recall.csv',index=False)
h=pd.DataFrame(hostcal);h.to_csv(O/'separate_host_calibration_draws.csv',index=False)
hf=h.groupby(['policy','fold','model'])[['fpr','recall','auc']].mean().reset_index();hf.to_csv(O/'separate_host_calibration_fold_means.csv',index=False)
sm=pd.DataFrame(seenrows).groupby(['policy','fold','model','coverage'])[['n','recall']].mean().reset_index();sm.to_csv(O/'seen_unseen_fold_means.csv',index=False)
# Compare model differences by fold rather than treating seeds and draws as independent samples.
pair=[]
for (pol,rule),g in allfold.groupby(['policy','rule']):
 for metric in ['fpr','recall','auc']:
  v=g.pivot(index='fold',columns='model',values=metric);delta=v.proposed-v.xgboost
  pair.append(dict(policy=pol,rule=rule,metric=metric,mean_difference=float(delta.mean()),sd_across_5_folds=float(delta.std(ddof=1)),minimum=float(delta.min()),maximum=float(delta.max()),folds=5))
pd.DataFrame(pair).to_csv(O/'paired_model_fold_differences.csv',index=False)
print('Host calibration design:',pd.DataFrame(design).to_string(index=False))
print('Separate-host macro means:',hf.groupby(['policy','model'])[['fpr','recall']].mean().to_string())
print('Seen:',sm[sm.coverage=='seen'].groupby(['policy','model']).recall.mean().to_string())
