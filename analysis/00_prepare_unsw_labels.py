"""Rebuild the UNSW-NB15 split and save validation/test labels and test-set attributes.
usage: python 00_prepare_unsw_labels.py <folder with the two UNSW-NB15 csv files>"""
import os as _os
_H = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.dirname(_H); _OUT = _os.path.join(_H, "out"); _os.makedirs(_OUT, exist_ok=True)
import sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
D=sys.argv[1]; out=_os.path.join(_OUT,"unsw_meta.npz")
tr=pd.read_csv(D+"/UNSW_NB15_training-set.csv").drop(columns=["id"])
te=pd.read_csv(D+"/UNSW_NB15_testing-set.csv").drop(columns=["id"])
n0=len(tr); tr=tr.drop_duplicates().reset_index(drop=True)
print("train raw",n0,"dedup",len(tr),"removed",n0-len(tr), "attack share raw->dedup", tr.label.mean())
trp,vap=train_test_split(tr,test_size=0.2,random_state=42,stratify=tr["label"])
print("train",len(trp),trp.label.value_counts().to_dict(),"val",len(vap),vap.label.value_counts().to_dict())
print("test",len(te),te.label.value_counts().to_dict())
feat=[c for c in te.columns if c not in("label","attack_cat")]
dup=te.duplicated().sum(); print("test dups",dup)
trkeys=set(map(tuple,tr[feat].astype(str).values))
print("feat n",len(feat))
np.savez(out,y_val=vap.label.values,y_test=te.label.values,cat_test=te.attack_cat.values.astype(str),
 sttl_test=te.sttl.values,state_test=te.state.values.astype(str),service_test=te.service.values.astype(str),
 sttl_val=vap.sttl.values,sttl_tr=trp.sttl.values,y_tr=trp.label.values, dup_test=te.duplicated().values)
for nm,df in (("train",trp),("val",vap),("test",te)):
    for lab in (0,1):
        s=df[df.label==lab].sttl
        print(nm,lab,{v:round((s==v).mean()*100,1) for v in (31,254,62,0)})
