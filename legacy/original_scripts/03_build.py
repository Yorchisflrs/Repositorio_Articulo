"""Build person-level analysis file (ages 6-17) from ENAHO modules 300, 200, 100, 612 and sumaria."""
import pyreadstat, glob, sys, numpy as np, pandas as pd
root, out = sys.argv[1], sys.argv[2]
K=['conglome','vivienda','hogar']; KP=K+['codperso']
def rd(pat,y,cols):
    f=glob.glob(f'{root}/{y}/{pat}')[0]
    _,m=pyreadstat.read_dta(f,metadataonly=True)
    real={c.lower():c for c in m.column_names}
    d,_=pyreadstat.read_dta(f,usecols=[real[c] for c in cols if c in real])
    d.columns=[c.lower() for c in d.columns]; return d
frames=[]
for y in range(2017,2026):
    e=rd('enaho01a-*-300.dta',y,KP+['ubigeo','dominio','estrato','mes','p203','p207','p208a','p300a','p301a','p301b','p301c','p303','p306','p307','p308a','p314a','factora07','factor07'])
    e=e.rename(columns={'factor07':'factor07_300'})
    # household head education
    hd=e[e.p203==1][K+['p301a','p207','p208a']].rename(columns={'p301a':'head_educ','p207':'head_sex','p208a':'head_age'}).drop_duplicates(K)
    m2=rd('enaho01-*-200.dta',y,KP+['p204','p205','p206'])
    h=rd('enaho01-*-100.dta',y,K+['result','p1142','p1144','nbi1'])
    eq=rd('enaho01-*-612.dta',y,K+['p612n','p612'])
    pc=eq[eq.p612n==7].groupby(K).p612.min().rename('computer').reset_index()
    s=rd('sumaria-*[0-9][0-9][0-9][0-9].dta',y,K+['mieperho','pobreza','gashog2d','factor07','percepho','totmieho'])
    d=e.merge(m2,on=KP,how='left').merge(h,on=K,how='left').merge(pc,on=K,how='left').merge(s,on=K,how='left').merge(hd,on=K,how='left')
    d['year']=y
    # children 0-17 in household
    kids=m2.merge(rd('enaho01-*-200.dta',y,KP+['p208a']),on=KP); kids=kids[(kids.p208a<18)&(kids.p204==1)].groupby(K).size().rename('n_kids').reset_index()
    d=d.merge(kids,on=K,how='left')
    frames.append(d[(d.p208a>=6)&(d.p208a<=17)])
    print(y,len(frames[-1]))
df=pd.concat(frames,ignore_index=True)
df.to_parquet(out); print(df.shape)
