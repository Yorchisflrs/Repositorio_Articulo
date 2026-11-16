"""E6 ablation without schooling-trajectory features; E0 design-based prevalence (Taylor linearization, strata=estrato, PSU=conglome)."""
import pandas as pd, numpy as np, json, warnings; warnings.filterwarnings('ignore')
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
d=pd.read_parquet('analytic.parquet'); XP=json.load(open('models/xgb_params.json'))
FEATS=['age','female','indig_tongue','school_years','overage','child_of_head','hh_mobile','hh_computer','poor','extreme_poor','exp_quintile','head_educ_years','head_female','head_age','hh_size','n_kids','rural','inadequate_housing']+[f'dom_{k}' for k in range(1,9)]
sets={'full':FEATS,'no_trajectory':[f for f in FEATS if f not in ('school_years','overage')],
      'no_digital':[f for f in FEATS if f not in ('hh_mobile','hh_computer')],'plus_internet_raw':FEATS+['hh_internet_raw'],'plus_internet_harmonized':FEATS+['hh_internet'],
      'only_socioecon_digital':[f for f in FEATS if f not in ('school_years','overage','age','female','child_of_head')]}
tr=d[d.year.isin([2017,2018,2019])]; rows=[]
for k,fs in sets.items():
    m=xgb.XGBClassifier(**XP,objective='binary:logistic',tree_method='hist',random_state=2026,n_jobs=-1).fit(tr[fs],tr.y)
    for yr in range(2020,2026):
        te=d[d.year==yr]; p=m.predict_proba(te[fs])[:,1]
        rows.append(dict(set=k,year=yr,auc=roc_auc_score(te.y,p,sample_weight=te.factora07),prauc=average_precision_score(te.y,p,sample_weight=te.factora07)))
a=pd.DataFrame(rows); a.to_csv('tables/E6_ablation.csv',index=False)
print(a.pivot(index='set',columns='year',values='auc').round(3)); print(a.pivot(index='set',columns='year',values='prauc').round(3))
def svy_mean(df,var='y'):
    w=df.factora07.values; y=df[var].values; W=w.sum(); m=(w*y).sum()/W
    z=w*(y-m)/W; df=df.assign(z=z); v=0
    for _,s in df.groupby(['year','estrato']):
        t=s.groupby('conglome').z.sum(); n=len(t)
        if n>1: v+=n/(n-1)*((t-t.mean())**2).sum()
    return m, np.sqrt(v)
rows=[]
for yr in range(2017,2026):
    for area,mask in [('National',None),('Urban',0),('Rural',1)]:
        s=d[d.year==yr] if mask is None else d[(d.year==yr)&(d.rural==mask)]
        m,se=svy_mean(s); rows.append(dict(year=yr,area=area,prev=m,lo=m-1.96*se,hi=m+1.96*se,n=len(s)))
    for v in ['hh_internet_raw','hh_internet','hh_computer']:
        s=d[d.year==yr].dropna(subset=[v]); m,se=svy_mean(s,v); rows.append(dict(year=yr,area=v,prev=m,lo=m-1.96*se,hi=m+1.96*se,n=len(s)))
p=pd.DataFrame(rows); p.to_csv('tables/E0_prevalence_design_based.csv',index=False)
print((p.pivot(index='year',columns='area',values='prev')*100).round(2))
print((p.assign(ci=lambda x:(x.hi-x.lo)*50).pivot(index='year',columns='area',values='ci')).round(2))
