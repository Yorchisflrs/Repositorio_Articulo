"""Figures for the IEEE manuscript (single-column width 3.5 in, 300 dpi, PDF + PNG)."""
import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, shap, pickle
plt.rcParams.update({'font.family':'serif','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,
  'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#52514e','axes.grid':True,'grid.color':'#e5e4e0','grid.linewidth':0.5,'lines.linewidth':1.6})
C={'XGB':'#2a78d6','RF':'#1baf7a','LR':'#eb6834','National':'#0b0b0b','Urban':'#2a78d6','Rural':'#eb6834'}
MK={'XGB':'o','RF':'s','LR':'^','National':'o','Urban':'s','Rural':'^'}
def save(fig,name): fig.tight_layout(pad=0.3); fig.savefig(f'figures/{name}.pdf'); fig.savefig(f'figures/{name}.png',dpi=300); plt.close(fig)
# Fig 1 prevalence
p=pd.read_csv('tables/E0_prevalence_design_based.csv'); fig,ax=plt.subplots(figsize=(3.5,2.2))
ax.axvspan(2019.5,2021.5,color='#f0efec',zorder=0); ax.text(2020.5,8.1,'COVID-19\nyears',ha='center',va='top',fontsize=6.5,color='#52514e')
for a in ['Rural','National','Urban']:
    s=p[p.area==a]; ax.fill_between(s.year,s.lo*100,s.hi*100,color=C[a],alpha=0.12,lw=0)
    ax.plot(s.year,s.prev*100,marker=MK[a],ms=4,color=C[a],label=a)
ax.set_ylabel('Not enrolled (%)'); ax.set_ylim(0,8.3); ax.set_xticks(range(2017,2026)); ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(0.5,-0.02))
save(fig,'fig1_prevalence')
# Fig 2 temporal transfer
r=pd.read_csv('tables/E1_E3_performance.csv'); fig,axs=plt.subplots(1,2,figsize=(3.5,2.35))
for m in ['XGB','RF','LR']:
    cv=r[(r.exp=='E1_CV_2017-19')&(r.model==m)].iloc[0]; s=r[r.exp.str.startswith('E2')&(r.model==m)].copy(); s['year']=s.exp.str[-4:].astype(int)
    axs[0].errorbar(s.year,s.auc,yerr=[s.auc-s.auc_lo,s.auc_hi-s.auc],marker=MK[m],ms=3.5,color=C[m],capsize=1.5,lw=1.2,elinewidth=0.7,label=m)
    axs[0].axhline(cv.auc,color=C[m],ls=':',lw=0.8)
    axs[1].errorbar(s.year,s.prauc,yerr=[s.prauc-s.pr_lo,s.pr_hi-s.prauc],marker=MK[m],ms=3.5,color=C[m],capsize=1.5,lw=1.2,elinewidth=0.7)
    axs[1].axhline(cv.prauc,color=C[m],ls=':',lw=0.8)
axs[0].set_title('(a) ROC-AUC'); axs[1].set_title('(b) PR-AUC')
for a in axs: a.set_xticks(range(2020,2026)); a.set_xticklabels(['20','21','22','23','24','25'])
axs[0].set_xlabel('Test year (20xx)'); axs[1].set_xlabel('Test year (20xx)')
h,l=axs[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.02))
fig.tight_layout(pad=0.3,rect=(0,0,1,0.9)); fig.savefig('figures/fig2_transfer.pdf'); fig.savefig('figures/fig2_transfer.png',dpi=300); plt.close(fig)
# Fig 3 reliability (XGB) by test year
tp=pd.read_parquet('models/test_preds.parquet'); fig,ax=plt.subplots(figsize=(3.5,2.4))
ax.plot([0,0.35],[0,0.35],color='#52514e',lw=0.8,ls='--',label='Perfect')
for yr,col,mk in [(2020,'#eb6834','^'),(2022,'#1baf7a','s'),(2025,'#2a78d6','o')]:
    s=tp[tp.year==yr].copy(); s['bin']=pd.qcut(s.XGB.rank(method='first'),10,labels=False)
    g=s.groupby('bin').apply(lambda x: pd.Series({'p':np.average(x.XGB,weights=x.w),'o':np.average(x.y,weights=x.w)}))
    ax.plot(g.p,g.o,marker=mk,ms=3.5,color=col,label=str(yr))
ax.set_xlabel('Mean predicted risk (decile)'); ax.set_ylabel('Observed non-enrollment'); ax.legend(frameon=False,loc='upper left')
save(fig,'fig3_calibration')
# Fig 4 SHAP beeswarm (pre-trained XGB, training data)
sv=np.load('models/shap_pre.npy'); X=pd.read_parquet('models/X_pre.parquet')
names={'dom_6':'Domain: South highlands','dom_5':'Domain: Central highlands','overage':'Age-grade distortion','age':'Age','school_years':'Years of schooling','head_educ_years':'Head education (yrs)','hh_computer':'Computer at home',
 'child_of_head':'Child of household head','n_kids':'Children in household','female':'Female','indig_tongue':'Indigenous mother tongue','hh_size':'Household size',
 'dom_7':'Domain: Selva','dom_1':'Domain: North coast','dom_6':'Domain: South highlands','dom_5':'Domain: Central highlands','exp_quintile':'Expenditure quintile','extreme_poor':'Extreme poverty'}
rng=np.random.default_rng(2026); idx=rng.choice(len(X),6000,replace=False)
plt.figure(figsize=(3.5,3.0)); shap.summary_plot(sv[idx],X.iloc[idx].rename(columns=names),max_display=12,show=False,plot_size=None,color_bar_label='Feature value',alpha=0.6)
f=plt.gcf(); f.set_size_inches(3.5,3.0)
for a in f.axes: a.tick_params(labelsize=6.5); a.xaxis.label.set_size(7); a.yaxis.label.set_size(7)
f.axes[0].set_xlabel('SHAP value (log-odds)'); save(f,'fig4_shap')
print('ok')
# Fig 5 SHAP importance drift (pre vs post)
imp=pd.read_csv('tables/E5_shap_drift.csv').head(10).iloc[::-1]
fig,ax=plt.subplots(figsize=(3.5,2.5)); yy=np.arange(len(imp))
ax.barh(yy+0.2,imp.pre_2017_19,height=0.38,color='#2a78d6',label='2017-2019 (training)')
ax.barh(yy-0.2,imp.post_2022_25,height=0.38,color='#eb6834',label='2022-2025 (test)')
ax.set_yticks(yy); ax.set_yticklabels([names.get(f,f) for f in imp.feature],fontsize=6.5); ax.set_xlabel('Mean |SHAP| (log-odds)')
ax.grid(axis='y',visible=False); ax.legend(frameon=False,loc='lower right',fontsize=6.5); save(fig,'fig5_shap_drift')
print('fig5 ok')
