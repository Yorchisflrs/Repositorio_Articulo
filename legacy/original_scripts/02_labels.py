import pyreadstat, glob, sys
root=sys.argv[1]
for y in ['2017','2020','2024']:
    f=glob.glob(f'{root}/{y}/enaho01a-*-300.dta')[0]
    _,m=pyreadstat.read_dta(f,metadataonly=True)
    lab={c.lower():l for c,l in zip(m.column_names,m.column_labels)}
    vl={c.lower():m.variable_value_labels.get(c) for c in m.column_names}
    print('=====',y)
    for v in ['p306','p307','p303','p308a','p301a','p314a','p208a','factor07','factora07','codinfor','p300a','p316_1']:
        print(v,'|',lab.get(v),'|',vl.get(v))
f=glob.glob(f'{root}/2024/enaho01-*-612.dta')[0]
d,m=pyreadstat.read_dta(f,usecols=['p612n','p612'],apply_value_formats=True)
print(d['p612n'].drop_duplicates().tolist()[:30]); print(d['p612'].value_counts().head())
f=glob.glob(f'{root}/2017/enaho01-*-612.dta')[0]
d,m=pyreadstat.read_dta(f,usecols=['p612n'],apply_value_formats=True)
print(d['p612n'].drop_duplicates().tolist()[:30])
for y in ['2017','2024']:
  f=glob.glob(f'{root}/{y}/enaho01-*-100.dta')[0]
  _,m=pyreadstat.read_dta(f,metadataonly=True)
  lab={c.lower():l for c,l in zip(m.column_names,m.column_labels)}
  print(y,[(v,lab.get(v)) for v in ['p1141','p1142','p1143','p1144','p1145','result','nbi1']])
  print(m.variable_value_labels.get('result'))
