"""Harmonize household internet. From 2023 the ENAHO item p1144 also counts mobile-data plans
(new items p114b1 fixed, p114b2 postpaid mobile, p114b3 prepaid mobile). hh_internet (harmonized) =
p1144 for 2017-2022 and p114b1 (fixed connection) for 2023-2024. Original kept as hh_internet_raw."""
import pyreadstat, glob, pandas as pd, numpy as np
d=pd.read_parquet('analytic.parquet')
if 'hh_internet_raw' not in d: d['hh_internet_raw']=d.hh_internet
fx=[]
for y in [2023,2024,2025]:
    f=glob.glob(f'/home/claude/enaho_raw/{y}/enaho01-*-100.dta')[0]
    t,_=pyreadstat.read_dta(f,usecols=['conglome','vivienda','hogar','p114b1','p114b2','p114b3']); t.columns=[c.lower() for c in t.columns]
    t['hhid']=str(y)+'_'+t.conglome.astype(str)+'_'+t.vivienda.astype(str)+'_'+t.hogar.astype(str); fx.append(t[['hhid','p114b1','p114b2','p114b3']])
fx=pd.concat(fx).drop_duplicates('hhid')
d=d.drop(columns=[c for c in ['p114b1','p114b2','p114b3'] if c in d]).merge(fx,on='hhid',how='left')
m=d.year>=2023
print('match rate 2023-24', d.loc[m,'p114b1'].notna().mean())
d['hh_internet']=np.where(m,(d.p114b1==1).astype(int),d.hh_internet_raw)
w=lambda g: round(100*np.average(g,weights=d.loc[g.index,'factora07']),1)
print(d.groupby('year').agg(raw=('hh_internet_raw',w),harmonized=('hh_internet',w)))
print('2023-24 mobile only share', d[m].assign(mob=lambda x:((x.p114b2==1)|(x.p114b3==1))&(x.p114b1!=1)).groupby('year').mob.mean().round(3).to_dict())
d.to_parquet('analytic.parquet')
