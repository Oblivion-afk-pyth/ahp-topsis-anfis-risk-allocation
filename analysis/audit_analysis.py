"""Reproduce aggregate analyses and audit the adaptive contribution.

Inputs are numerical manuscript matrices, not authenticated expert returns.
Run: python audit_analysis.py. Requires NumPy. Outputs are written beside this file.
"""
from pathlib import Path
import os, json, csv, runpy, platform
import numpy as np

BASE = Path(__file__).resolve().parent
os.chdir(BASE)
inp = json.loads((BASE/'aggregate_inputs.json').read_text())
expert = json.loads((BASE/'expert_inputs.json').read_text())
E = np.array(expert['AHP'],float)
EX = np.array(expert['R1_scores'],float)
A = np.exp(np.log(E).mean(axis=0))
Xs = np.array(inp['risk_matrices'], float)
Xs[0] = np.median(EX,axis=0)

def ahp(a):
    val, vec = np.linalg.eig(a)
    k = np.argmax(val.real)
    w = abs(vec[:,k].real); w /= w.sum()
    lm = float(val[k].real)
    return w, lm, (lm-7)/6/1.32

def topsis(x,w):
    n=x/np.sqrt((x*x).sum(axis=0)); v=n*w
    pos=v.max(axis=0); neg=v.min(axis=0)
    sp=np.linalg.norm(v-pos,axis=1); sm=np.linalg.norm(v-neg,axis=1)
    return {'normalised':n.tolist(),'weighted':v.tolist(),'ideal':pos.tolist(),
            'anti_ideal':neg.tolist(),'Splus':sp.tolist(),'Sminus':sm.tolist(),
            'CC':(sm/(sp+sm)).tolist()}

w,lm,cr=ahp(A)
names=['R1_PaymentDelay','R2_DesignChange','R3_SiteConditions','R4_PriceEscalation','R5_ForceMajeure','R6_RegulatoryChange']
res=[topsis(x,w) for x in Xs]
# Regenerate the exact input layout expected by the retained model implementation.
p1={'weights':w.tolist(),'lmax':lm,'CI':(lm-7)/6,'CR':cr,
    'results':{names[i]:dict(X=Xs[i].tolist(),Sp=r['Splus'],Sm=r['Sminus'],CC=r['CC']) for i,r in enumerate(res)}}
(BASE/'part1_results.json').write_text(json.dumps(p1,indent=2))
(BASE/'r2r6.json').write_text(json.dumps({k:v for k,v in p1['results'].items() if k!=names[0]},indent=2))
model=runpy.run_path(str(BASE/'reproduce_anfis.py'))
model_results=json.loads((BASE/'part2_results.json').read_text())

# Matched ablation: identical 16 rows, starting centres, widths and ridge estimate.
# Retain the original RNG call sequence, including the 20-row FIS initialisation.
g=model['train_anfis'].__globals__
g['rng']=np.random.default_rng(7)
original_init=g['init_premise']
initial={}
def capture_init(x):
    c,s=original_init(x)
    initial['C']=c.copy(); initial['S']=s.copy()
    return c,s
g['init_premise']=capture_init
rows=[]; initial_preds=[]
X=Xs.reshape(24,7)/9; y=np.array([r['CC'] for r in res]).ravel()
rid=np.repeat(np.arange(6),4)
for ti in range(6):
    te=rid==ti; va=rid==(ti+1)%6; tr=~(te|va)
    c,s,th,curves,updates=g['train_anfis'](X[tr],y[tr],X[va],y[va])
    ci,si=initial['C'],initial['S']
    th0,_=g['lse_consequents'](X[tr],y[tr],g['firing'](X[tr],ci,si)[1],g['LAM'])
    yp0=g['predict'](X[te],ci,si,th0)
    yp=g['predict'](X[te],c,s,th)
    assert np.allclose(yp,np.array(model_results['preds']['anfis'])[te],atol=1e-11)
    initial_preds.extend(yp0.tolist())
    rows.append({'risk':names[ti],'selected_updates':updates,'attempted_updates':len(curves['val']),
                 'initial_RMSE':g['rmse'](y[te],yp0),'adaptive_RMSE':g['rmse'](y[te],yp),
                 'initial_training_RMSE':g['rmse'](y[tr],g['predict'](X[tr],ci,si,th0)),
                 'initial_pair':g['pairwise_acc'](y[te],yp0),
                 'initial_top1':int(np.argmax(yp0)==np.argmax(y[te])),
                 'initial_validation_RMSE':g['rmse'](y[va],g['predict'](X[va],ci,si,th0)),
                 'selected_validation_RMSE':g['rmse'](y[va],g['predict'](X[va],c,s,th))})
    original_init(X[~te])
ip=np.array(initial_preds)
initial_summary={'mean_rmse':np.mean([r['initial_RMSE'] for r in rows]),
 'mean_mae':float(np.abs(ip-y).mean()),'pooled_rmse':g['rmse'](ip,y),
 'pooled_r2':float(1-np.sum((ip-y)**2)/np.sum((y-y.mean())**2)),
 'pair_correct':sum(r['initial_pair'][0] for r in rows),'top1':sum(r['initial_top1'] for r in rows)}

oat=[]
for j in range(7):
 for d in [.1,-.1,.2,-.2]:
    nw=w.copy(); nw[j]=w[j]*(1+d); oth=np.arange(7)!=j
    nw[oth]=w[oth]*(1-nw[j])/w[oth].sum()
    cc=topsis(Xs[0],nw)['CC']
    oat.append({'criterion':j+1,'fraction':d,'weight':nw[j],'CC_shared':cc[3],'weights':nw.tolist(),'CC':cc})
rng=np.random.default_rng(42)
mc=[]
for i,n in [(0,10000)]+[(i,2000) for i in range(6)]:
    weights=rng.uniform(.8,1.2,(n,7))*w; weights/=weights.sum(axis=1,keepdims=True)
    cc=np.array([topsis(Xs[i],nw)['CC'] for nw in weights]); best=np.argmax(res[i]['CC'])
    mc.append({'risk':names[i],'N':n,'top1_retention':float(np.mean(cc.argmax(1)==best)),
               'best_cc_mean':float(cc[:,best].mean()),'best_cc_sd':float(cc[:,best].std()),
               'best_cc_min':float(cc[:,best].min()),'best_cc_max':float(cc[:,best].max())})
# A displayed rounded matrix is not exactly reciprocal: quantify rather than hide.
Ar=A.copy()
for i in range(7):
 for j in range(i+1,7): Ar[j,i]=1/Ar[i,j]
wr,lmr,crr=ahp(Ar)
rec={'weights':wr.tolist(),'lambda':lmr,'CR':crr,'max_weight_change':float(abs(w-wr).max()),
     'CC':[topsis(x,wr)['CC'] for x in Xs]}
assert all(np.argmax(res[i]['CC'])==np.argmax(rec['CC'][i]) for i in range(6))
cx=np.repeat(np.array([[5],[1],[3],[9]]),7,axis=1)
cz=np.repeat(np.array([[5],[1],[3],[6]]),7,axis=1)
counter=[topsis(cx,w)['CC'][0],topsis(cz,w)['CC'][0]]
assert np.allclose(counter,[.5,.8])
audit={'weights':w.tolist(),'lambda':lm,'CR':cr,'topsis':res,'oat':oat,'monte_carlo':mc,
       'matched_ablation':rows,'matched_initial_summary':initial_summary,
       'reciprocal_check':rec,'counterexample':counter,
       'environment':{'python':platform.python_version(),'numpy':np.__version__}}
# Empirical dispersion and paired expert-resampling for the available R1 records.
ew=np.array([ahp(a)[0] for a in E]); ecr=np.array([ahp(a)[2] for a in E])
quart=np.quantile(EX,[.25,.5,.75],axis=0,method='linear')
wquart=np.quantile(ew,[.25,.5,.75],axis=0,method='linear')
ec=np.array([topsis(x,wi)['CC'] for x,wi in zip(EX,ew)])
ranks=(-ec).argsort(axis=1).argsort(axis=1)+1
# All rankings are strict in these data; standard untied Kendall W.
assert all(len(set(row))==4 for row in ec)
rs=ranks.sum(axis=0); m=37; n=4
kendall_W=float(12*np.sum((rs-m*(n+1)/2)**2)/(m*m*(n**3-n)))
brng=np.random.default_rng(20260912); boots=[]
for _ in range(10000):
    ids=brng.integers(0,37,37)
    aw=ahp(np.exp(np.log(E[ids]).mean(axis=0)))[0]
    bx=np.median(EX[ids],axis=0)
    boots.append(topsis(bx,aw)['CC'])
boots=np.array(boots)
loo=[]
for k in range(37):
    ids=np.arange(37)!=k
    aw=ahp(np.exp(np.log(E[ids]).mean(axis=0)))[0]
    loo.append(topsis(np.median(EX[ids],axis=0),aw)['CC'])
audit['experts']={'N':37,'aggregate_AHP':A.tolist(),'R1_medians':Xs[0].tolist(),
 'individual_CR':ecr.tolist(),'CR_min':float(ecr.min()),'CR_max':float(ecr.max()),
 'weight_quartiles':wquart.tolist(),'score_quartiles':quart.tolist(),
 'individual_R1_CC':ec.tolist(),'rank_sums':rs.tolist(),'Kendall_W':kendall_W,
 'individual_top1_counts':np.bincount(ec.argmax(1),minlength=4).tolist(),
 'bootstrap_top1_counts':np.bincount(boots.argmax(1),minlength=4).tolist(),
 'bootstrap_CC_interval':np.quantile(boots,[.025,.975],axis=0).tolist(),
 'bootstrap_margin_interval':np.quantile(boots[:,3]-boots[:,:3].max(axis=1),[.025,.975]).tolist(),
 'leave_one_expert_top1_counts':np.bincount(np.array(loo).argmax(1),minlength=4).tolist()}
(BASE/'audit_results.json').write_text(json.dumps(audit,indent=2))
with (BASE/'aggregate_scores.csv').open('w',newline='') as f:
    z=csv.writer(f);z.writerow(['risk','alternative']+[f'C{j+1}' for j in range(7)]+['TOPSIS_CC'])
    for i,x in enumerate(Xs):
        for a,row in enumerate(x):z.writerow([f'R{i+1}',['Client','Contractor','Consultant','Shared'][a]]+row.tolist()+[res[i]['CC'][a]])
with (BASE/'all_predictions.csv').open('w',newline='') as f:
    z=csv.writer(f);z.writerow(['risk','alternative','target','ANFIS','FIS_20','MLR_20','FIS_16_matched'])
    for k in range(24):z.writerow([f'R{k//4+1}',k%4+1,y[k],model_results['preds']['anfis'][k],model_results['preds']['fis'][k],model_results['preds']['mlr'][k],ip[k]])
print('Selected updates:',[r['selected_updates'] for r in rows])
print('Matched initial summary:',initial_summary)
print('Reciprocity sensitivity:',rec['max_weight_change'])
print('Expert results:',{k:audit['experts'][k] for k in ['Kendall_W','individual_top1_counts','bootstrap_top1_counts','bootstrap_margin_interval','leave_one_expert_top1_counts']})
print('Audit complete. Expert workbook identity confirmed by author; R2-R6 remain reconstructed.')
