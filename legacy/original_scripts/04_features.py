"""Analytic sample and features. Population: ages 12-17, usual residents, secondary not completed,
interviewed April-December (avoids school-vacation artefact in p306)."""
import pandas as pd, numpy as np, sys
d=pd.read_parquet(sys.argv[1]); d['mes']=d.mes.astype(int)
d=d[(d.p204==1)&(d.p205!=1)&(d.p301a<6)&d.p306.notna()&(d.mes>=4)&d.p208a.between(12,17)].copy()
d['y']=(d.p306==2).astype(int)
d['age']=d.p208a.astype(int)
d['female']=(d.p207==2).astype(int)
d['indig_tongue']=d.p300a.isin([1,2,3,10,11,12,13,14,15]).astype(int)
# years of schooling approved
# ENAHO records the last primary grade in p301c (grado) and the last secondary year in p301b (año)
gp=d.p301c.fillna(0).clip(0,5); gs=d.p301b.fillna(0).clip(0,4)
yrs=np.select([d.p301a<=2, d.p301a==3, d.p301a==4, d.p301a==5],[0, gp, 6, 6+gs],np.nan)
d['school_years']=yrs
d['overage']=(d.age-6-d.school_years).clip(-2,12)
d['child_of_head']=d.p203.isin([3]).astype(int)
d['hh_internet']=(d.p1144==1).astype(int)
d['hh_mobile']=(d.p1142==1).astype(int)
d['hh_computer']=(d.computer==1).astype(float); d.loc[d.computer.isna(),'hh_computer']=np.nan
d['poor']=d.pobreza.isin([1,2]).astype(int)
d['extreme_poor']=(d.pobreza==1).astype(int)
# within-year weighted quintile of per-capita expenditure (avoids inflation drift)
d['pcexp']=d.gashog2d/d.mieperho
def wq(gr):
    o=gr.pcexp.argsort().values; cw=np.cumsum(gr.factora07.values[o])/gr.factora07.sum()
    q=np.empty(len(gr)); q[o]=np.minimum((cw*5).astype(int)+1,5); return pd.Series(q,index=gr.index)
d['exp_quintile']=d.groupby('year',group_keys=False).apply(wq)
d['head_educ_years']=d.head_educ.map({1:0,2:0,3:3,4:6,5:8,6:11,7:13,8:14,9:14,10:16,11:18,12:np.nan})
d['head_female']=(d.head_sex==2).astype(int)
d['head_age']=d.head_age
d['hh_size']=d.mieperho; d['n_kids']=d.n_kids
d['rural']=(d.estrato>=6).astype(int)
d['inadequate_housing']=(d.nbi1==1).astype(int)
d['dominio']=d.dominio.astype(int)
for k in range(1,9): d[f'dom_{k}']=(d.dominio==k).astype(int)
d['dept']=d.ubigeo.str[:2]
d['psu']=d.year.astype(str)+'_'+d.conglome.astype(str)
d['hhid']=d.psu+'_'+d.vivienda.astype(str)+'_'+d.hogar.astype(str)
FEATS=['age','female','indig_tongue','school_years','overage','child_of_head','hh_internet','hh_mobile','hh_computer',
 'poor','extreme_poor','exp_quintile','head_educ_years','head_female','head_age','hh_size','n_kids','rural','inadequate_housing']+[f'dom_{k}' for k in range(1,9)]
keep=FEATS+['y','year','mes','factora07','estrato','conglome','psu','hhid','dept','dominio','p314a']
d[keep].to_parquet(sys.argv[2])
print(d.shape, d[FEATS].isna().mean().round(4)[lambda s:s>0].to_dict())
print(d.groupby('year').y.agg(['size','sum','mean']).round(4))
