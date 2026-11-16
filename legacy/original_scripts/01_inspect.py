import pyreadstat, glob, os, sys
root=sys.argv[1]
want={'300':['conglome','vivienda','hogar','codperso','ubigeo','dominio','estrato','mes','p203','p207','p208a','p300a','p301a','p301b','p301c','p303','p304a','p305','p306','p307','p308a','p308b','p308c','p308d','p310','p311','p313','p314a','p314b_1','p314d','p316_1','p316a1','p316b','p316c','factor07','factora07','facpob07','codinfor','ticuest01a','p209'],
'200':['p203','p207','p208a','p209','facpob07','factor07','mes'],
'100':['p1141','p1142','p1143','p1144','p1145','result','factor07','nbi1','p110','p111a','p1121'],
'612':['p612n','p612','factor07'],
'sumaria':['mieperho','pobreza','gashog2d','inghog2d','linea','factor07','estrsocial','percepho']}
for y in sorted(os.listdir(root)):
    d=os.path.join(root,y)
    for key,pat in [('300','enaho01a-*-300.dta'),('200','enaho01-*-200.dta'),('100','enaho01-*-100.dta'),('612','enaho01-*-612.dta'),('sumaria','sumaria-*[0-9][0-9][0-9][0-9].dta')]:
        f=glob.glob(os.path.join(d,pat))
        if not f: print(y,key,'MISSING'); continue
        _,m=pyreadstat.read_dta(f[0],metadataonly=True)
        cols=[c.lower() for c in m.column_names]
        lab={c.lower():l for c,l in zip(m.column_names,m.column_labels)}
        res=[]
        for v in want[key]:
            if v in cols: res.append(v)
        miss=[v for v in want[key] if v not in cols]
        print(f"{y} {key} n={m.number_rows} ncol={len(cols)} file={os.path.basename(f[0])}\n   present={res}\n   missing={miss}")
