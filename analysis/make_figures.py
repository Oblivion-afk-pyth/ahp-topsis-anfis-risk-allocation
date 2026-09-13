from pathlib import Path
import json,numpy as np
S=Path(__file__).resolve().parent
r=json.loads((S/'audit_results.json').read_text())
m=json.loads((S/'part2_results.json').read_text())
alts=['Client','Contractor','Consultant','Shared']
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
figdir=S/'figures';figdir.mkdir(exist_ok=True)
def save(fig,name):fig.savefig(figdir/name,dpi=190,bbox_inches='tight',facecolor='white');plt.close(fig)
fig,ax=plt.subplots(figsize=(7,3.4));ax.axis('off')
boxes=[(.02,.67,.28,.25,'Define risk and alternatives\nSeven capability\ncriteria'),(.36,.67,.28,.25,'AHP weights and\ncomplete 4 × 7 scores'),(.70,.67,.28,.25,'Direct TOPSIS ranking\nOperational output'),(.36,.15,.28,.28,'24 profiles and labels\nSix scenario groups'),(.70,.15,.28,.28,'Exploratory comparison\nANFIS and\nnon-adaptive FIS')]
from matplotlib.patches import Rectangle
for x,y,ww,hh,tx in boxes:
    ax.add_patch(Rectangle((x,y),ww,hh,facecolor='#eff3f5',edgecolor='#555555'));ax.text(x+ww/2,y+hh/2,tx,ha='center',va='center',fontsize=8)
for start,end in [((.30,.795),(.36,.795)),((.64,.795),(.70,.795)),((.84,.67),(.5,.43)),((.64,.29),(.70,.29))]:ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#555555'})
ax.text(.02,.38,'Separate checks',weight='bold');ax.text(.02,.17,'Weight sensitivity\nExpert dispersion for R1\nPaired expert resampling',fontsize=9)
save(fig,'figure1.png')
fig,ax=plt.subplots(figsize=(6.5,2.8));v=r['topsis'][0]['CC'];ax.bar(alts,v,color=['#a4b5c2']*3+['#38566a']);ax.set_ylim(0,1.04);ax.set_ylabel('TOPSIS coefficient')
for a,z in enumerate(v):ax.text(a,z+.025,f'{z:.3f}',ha='center')
save(fig,'figure2.png')
fig,ax=plt.subplots(figsize=(6.5,2.8));vv=[x['top1_retention']*100 for x in r['monte_carlo'][1:]];ax.bar([f'R{i+1}' for i in range(6)],vv,color='#536e80');ax.set_ylim(0,110);ax.set_ylabel('Original leader retained (%)')
for a,z in enumerate(vv):ax.text(a,z+2,f'{z:.2f}%' if a==4 else '100%',ha='center',fontsize=9)
save(fig,'figure3.png')
cur=m['curves'];a=r['matched_ablation'][-1]
# Initial training error calculated from the same R6 inner split.
preds=np.loadtxt(S/'all_predictions.csv',delimiter=',',skiprows=1,usecols=[7]) if False else None
epochs=np.arange(1,len(cur['train'])+1)
fig,ax=plt.subplots(figsize=(6.5,2.8));ax.plot(np.r_[0,epochs],np.r_[a['initial_training_RMSE'],cur['train']],label='Training');ax.plot(np.r_[0,epochs],np.r_[a['initial_validation_RMSE'],cur['val']],label='Inner validation');ax.axvline(a['selected_updates'],color='#666666',ls='--',label='Retained update');ax.set_xlabel('Completed premise updates');ax.set_ylabel('RMSE');ax.legend(fontsize=8);save(fig,'figure4.png')
fig,ax=plt.subplots(figsize=(6.5,2.8));x=np.arange(6)
for j,(name,label) in enumerate([('anfis','ANFIS 16'),('fis','FIS 20'),('mlr','MLR 20')]):ax.bar(x+(j-1)*.24,[f[name]['rmse'] for f in m['folds']],width=.24,label=label)
ax.set_xticks(x,[f'R{i+1}' for i in range(6)]);ax.set_ylabel('Held-out RMSE');ax.legend(fontsize=8);save(fig,'figure5.png')
