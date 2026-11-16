"""Experiments E1-E5: in-period CV, temporal transfer, retraining/recalibration, subgroup audit, SHAP drift."""
import pandas as pd, numpy as np, json, sys, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb, statsmodels.api as sm
SEED=2026; rng=np.random.default_rng(SEED)
d=pd.read_parquet('analytic.parquet')
FEATS=['age','female','indig_tongue','school_years','overage','child_of_head','hh_mobile','hh_computer','poor','extreme_poor','exp_quintile','head_educ_years','head_female','head_age','hh_size','n_kids','rural','inadequate_housing']+[f'dom_{k}' for k in range(1,9)]
PRE=[2017,2018,2019]
def calib(y,p,w):
    lp=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))); ww=w/w.mean()
    try:
        slope=sm.GLM(y,sm.add_constant(lp),family=sm.families.Binomial(),freq_weights=ww).fit().params[1]
        citl=sm.GLM(y,np.ones_like(lp),family=sm.families.Binomial(),offset=lp,freq_weights=ww).fit().params[0]
    except Exception: slope,citl=np.nan,np.nan
    return citl,slope
def metrics(y,p,w):
    citl,slope=calib(y,p,w)
    return dict(n=len(y),pos=int(y.sum()),prev=np.average(y,weights=w),mean_pred=np.average(p,weights=w),
      auc=roc_auc_score(y,p,sample_weight=w),prauc=average_precision_score(y,p,sample_weight=w),
      brier=brier_score_loss(y,p,sample_weight=w),citl=citl,slope=slope)
def boot_ci(df,p,B=300):
    psus=df.psu.unique(); idx=df.groupby('psu').indices; a=[];b=[]
    for _ in range(B):
        s=rng.choice(psus,len(psus)); ii=np.concatenate([idx[k] for k in s])
        yy,pp,ww=df.y.values[ii],p[ii],df.factora07.values[ii]
        if yy.min()==yy.max(): continue
        a.append(roc_auc_score(yy,pp,sample_weight=ww)); b.append(average_precision_score(yy,pp,sample_weight=ww))
    return np.percentile(a,[2.5,97.5]),np.percentile(b,[2.5,97.5])
def make(name,params=None):
    if name=='LR': return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=1.0,max_iter=2000))
    if name=='RF': return make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=500,min_samples_leaf=20,max_features='sqrt',n_jobs=-1,random_state=SEED))
    if name=='XGB': return xgb.XGBClassifier(**params,objective='binary:logistic',eval_metric='logloss',tree_method='hist',random_state=SEED,n_jobs=-1)
tr=d[d.year.isin(PRE)].reset_index(drop=True); X,Y=tr[FEATS],tr.y.values
gkf=GroupKFold(n_splits=5)
# --- tune XGB on grouped CV (PR-AUC)
grid=[dict(max_depth=md,min_child_weight=mc,n_estimators=ne,learning_rate=0.03,subsample=0.8,colsample_bytree=0.8,reg_lambda=1.0)
      for md in [3,4,5] for mc in [5,20] for ne in [300,600]]
best=None
for g in grid:
    oof=np.zeros(len(tr))
    for a,b in gkf.split(X,Y,tr.psu): oof[b]=make('XGB',g).fit(X.iloc[a],Y[a]).predict_proba(X.iloc[b])[:,1]
    s=average_precision_score(Y,oof,sample_weight=tr.factora07)
    if best is None or s>best[0]: best=(s,g)
print('best xgb',best)
XP=best[1]; json.dump(XP,open('models/xgb_params.json','w'))
# --- E1 in-period grouped CV
rows=[]; oofs={}
for m in ['LR','RF','XGB']:
    oof=np.zeros(len(tr))
    for a,b in gkf.split(X,Y,tr.psu): oof[b]=make(m,XP).fit(X.iloc[a],Y[a]).predict_proba(X.iloc[b])[:,1]
    oofs[m]=oof; r=metrics(Y,oof,tr.factora07.values); r.update(model=m,exp='E1_CV_2017-19'); rows.append(r)
# --- E2 temporal transfer
fits={m:make(m,XP).fit(X,Y) for m in ['LR','RF','XGB']}
preds={}
for yr in range(2020,2026):
    te=d[d.year==yr].reset_index(drop=True)
    for m,f in fits.items():
        p=f.predict_proba(te[FEATS])[:,1]; preds[(m,yr)]=p
        r=metrics(te.y.values,p,te.factora07.values); (al,ah),(bl,bh)=boot_ci(te,p,200)
        r.update(model=m,exp=f'E2_pre->{yr}',auc_lo=al,auc_hi=ah,pr_lo=bl,pr_hi=bh); rows.append(r)
# --- E3 retrain on 2022-2024 vs recalibrate pre-model on 2023-2024, test 2025
te=d[d.year==2025].reset_index(drop=True); rc=d[d.year.isin([2023,2024])].reset_index(drop=True)
for m in ['LR','XGB']:
    post=d[d.year.isin([2022,2023,2024])]
    f2=make(m,XP).fit(post[FEATS],post.y.values); p=f2.predict_proba(te[FEATS])[:,1]
    r=metrics(te.y.values,p,te.factora07.values); r.update(model=m,exp='E3_retrain22-24->2025'); rows.append(r)
    lp=lambda q: np.log(np.clip(q,1e-6,1-1e-6)/(1-np.clip(q,1e-6,1-1e-6)))
    prc=fits[m].predict_proba(rc[FEATS])[:,1]
    cal=LogisticRegression(C=1e6).fit(lp(prc).reshape(-1,1),rc.y.values,sample_weight=rc.factora07.values)
    p=cal.predict_proba(lp(fits[m].predict_proba(te[FEATS])[:,1]).reshape(-1,1))[:,1]
    r=metrics(te.y.values,p,te.factora07.values); r.update(model=m,exp='E3_recalib23-24->2025'); rows.append(r)
res=pd.DataFrame(rows); res.to_csv('tables/E1_E3_performance.csv',index=False)
pd.set_option('display.width',250); print(res[['exp','model','n','pos','prev','mean_pred','auc','auc_lo','auc_hi','prauc','brier','citl','slope']].round(4).to_string())
# --- E4 subgroup audit on 2022-2024 (pre-trained XGB and LR)
post=d[d.year.isin([2022,2023,2024,2025])].reset_index(drop=True); sub=[]
for m in ['LR','XGB']:
    p=fits[m].predict_proba(post[FEATS])[:,1]
    thr=np.quantile(fits[m].predict_proba(X)[:,1],0.90)  # top-10% risk threshold fixed on training data
    for gname,mask in [('urban',post.rural==0),('rural',post.rural==1),('non-poor',post.poor==0),('poor',post.poor==1),
                       ('Spanish',post.indig_tongue==0),('Indigenous tongue',post.indig_tongue==1),('male',post.female==0),('female',post.female==1),('all',post.y>=0)]:
        q=post[mask]; pp=p[mask.values]; r=metrics(q.y.values,pp,q.factora07.values)
        flag=pp>=thr; r['recall_top10']=np.average(flag[q.y.values==1],weights=q.factora07.values[q.y.values==1]); r['flag_rate']=np.average(flag,weights=q.factora07.values)
        r.update(model=m,group=gname); sub.append(r)
sub=pd.DataFrame(sub); sub.to_csv('tables/E4_subgroups.csv',index=False)
print(sub[['model','group','n','pos','prev','mean_pred','auc','prauc','citl','slope','recall_top10','flag_rate']].round(3).to_string())
# --- E5 SHAP drift for pre-trained XGB
import shap
ex=shap.TreeExplainer(fits['XGB'])
sv_pre=ex.shap_values(X); sv_post=ex.shap_values(post[FEATS])
imp=pd.DataFrame({'feature':FEATS,'pre_2017_19':np.abs(sv_pre).mean(0),'post_2022_25':np.abs(sv_post).mean(0)}).sort_values('pre_2017_19',ascending=False)
imp.to_csv('tables/E5_shap_drift.csv',index=False); print(imp.round(4).to_string())
np.save('models/shap_pre.npy',sv_pre); np.save('models/shap_post.npy',sv_post)
X.to_parquet('models/X_pre.parquet'); post[FEATS].to_parquet('models/X_post.parquet')
import pickle; pickle.dump(fits,open('models/fits_pre.pkl','wb'))
pd.DataFrame({'year':np.concatenate([d[d.year==yr].year.values for yr in range(2020,2026)]),
 'y':np.concatenate([d[d.year==yr].y.values for yr in range(2020,2026)]),
 'w':np.concatenate([d[d.year==yr].factora07.values for yr in range(2020,2026)]),
 **{m:np.concatenate([preds[(m,yr)] for yr in range(2020,2026)]) for m in ['LR','RF','XGB']}}).to_parquet('models/test_preds.parquet')
pd.DataFrame({'y':Y,'w':tr.factora07.values,**oofs}).to_parquet('models/oof_pre.parquet')
