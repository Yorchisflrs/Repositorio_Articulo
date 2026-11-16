"""Calendar artefact (non-enrollment by interview month, ages 12-17) and 2020 performance by interview mode."""
import pandas as pd, numpy as np, pyreadstat, glob, pickle
from sklearn.metrics import roc_auc_score, average_precision_score
b=pd.read_parquet('base_6_17.parquet'); b['mes']=b.mes.astype(int)
p=b[(b.p204==1)&(b.p205!=1)&(b.p301a<6)&b.p306.notna()&b.p208a.between(12,17)].copy(); p['t']=(p.p306==2).astype(int)
mm=p.groupby('mes').apply(lambda g: round(100*np.average(g['t'],weights=g['factora07']),1)); mm.to_csv('tables/month_artefact_12_17.csv'); print(mm.to_dict())
fits=pickle.load(open('models/fits_pre.pkl','rb')); FEATS=list(pd.read_parquet('models/X_pre.parquet').columns)
t,_=pyreadstat.read_dta(glob.glob('/home/claude/enaho_raw/2020/enaho01-*-100.dta')[0],usecols=['conglome','vivienda','hogar','tipoentrevista']); t.columns=[c.lower() for c in t.columns]
t['hhid']='2020_'+t.conglome.astype(str)+'_'+t.vivienda.astype(str)+'_'+t.hogar.astype(str)
d=pd.read_parquet('analytic.parquet'); d=d[d.year==2020].merge(t[['hhid','tipoentrevista']],on='hhid')
print('telephone share',round((d.tipoentrevista==2).mean(),3), d.groupby('mes').apply(lambda g:round((g.tipoentrevista==2).mean(),2)).to_dict())
d['p']=fits['XGB'].predict_proba(d[FEATS])[:,1]; rows=[]
for k,g in d.groupby('tipoentrevista'):
    rows.append(dict(mode={1:'face-to-face',2:'telephone'}[int(k)],n=len(g),pos=int(g.y.sum()),obs=np.average(g.y,weights=g.factora07),pred=np.average(g.p,weights=g.factora07),
      auc=roc_auc_score(g.y,g.p,sample_weight=g.factora07),prauc=average_precision_score(g.y,g.p,sample_weight=g.factora07)))
r=pd.DataFrame(rows); r.to_csv('tables/E2b_2020_by_mode.csv',index=False); print(r.round(4))
