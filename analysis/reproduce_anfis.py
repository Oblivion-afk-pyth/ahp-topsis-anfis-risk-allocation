"""Part 2: Redesigned ANFIS (2-rule, cluster-initialised, regularised) + baselines, LORO CV."""
import numpy as np, json

rng = np.random.default_rng(7)
p1 = json.load(open("part1_results.json"))
r26 = json.load(open("r2r6.json"))
w = np.array(p1["weights"])

risks = ["R1_PaymentDelay","R2_DesignChange","R3_SiteConditions","R4_PriceEscalation","R5_ForceMajeure","R6_RegulatoryChange"]
data = {}
data["R1_PaymentDelay"] = p1["results"]["R1_PaymentDelay"]
for k in risks[1:]:
    data[k] = r26[k]

X_all, y_all, risk_id = [], [], []
for ri, k in enumerate(risks):
    X = np.array(data[k]["X"], float) / 9.0   # normalise to (0,1]
    CC = np.array(data[k]["CC"], float)
    for a in range(4):
        X_all.append(X[a]); y_all.append(CC[a]); risk_id.append(ri)
X_all = np.array(X_all); y_all = np.array(y_all); risk_id = np.array(risk_id)
print("dataset:", X_all.shape, y_all.shape)

K = 2      # rules
LAM = 0.05 # ridge
EPOCHS = 200
LR = 0.02
PATIENCE = 10

def init_premise(Xtr):
    # k-means (2 clusters) init
    idx = rng.choice(len(Xtr), K, replace=False)
    C = Xtr[idx].copy()
    for _ in range(50):
        d = ((Xtr[:,None,:]-C[None,:,:])**2).sum(-1)
        lab = d.argmin(1)
        for k in range(K):
            if (lab==k).any(): C[k] = Xtr[lab==k].mean(0)
    S = np.full((K,7), 0.35)
    return C, S

def firing(X, C, S):
    # product of Gaussians per rule
    d2 = (X[:,None,:]-C[None,:,:])**2 / (2*S[None,:,:]**2)
    lw = -d2.sum(-1)
    W = np.exp(lw - lw.max(1, keepdims=True))
    Wn = W / W.sum(1, keepdims=True)   # normalised firing strengths
    return W, Wn

def lse_consequents(X, y, Wn, lam):
    # design: for each rule, [x,1]*wn
    N = len(X)
    Phi = np.zeros((N, K*8))
    for k in range(K):
        Phi[:, k*8:k*8+7] = X * Wn[:,[k]]
        Phi[:, k*8+7] = Wn[:,k]
    A = Phi.T@Phi + lam*np.eye(K*8)
    theta = np.linalg.solve(A, Phi.T@y)
    return theta, Phi

def predict(X, C, S, theta):
    _, Wn = firing(X, C, S)
    N = len(X)
    Phi = np.zeros((N, K*8))
    for k in range(K):
        Phi[:, k*8:k*8+7] = X * Wn[:,[k]]
        Phi[:, k*8+7] = Wn[:,k]
    return Phi@theta

def rmse(a,b): return float(np.sqrt(((a-b)**2).mean()))

def train_anfis(Xtr, ytr, Xval, yval):
    C, S = init_premise(Xtr)
    theta, _ = lse_consequents(Xtr, ytr, firing(Xtr,C,S)[1], LAM)
    best = (rmse(predict(Xval,C,S,theta), yval), C.copy(), S.copy(), theta.copy(), 0)
    curves = {"train":[], "val":[]}
    wait = 0
    for ep in range(EPOCHS):
        # forward pass: LSE for consequents
        _, Wn = firing(Xtr, C, S)
        theta, _ = lse_consequents(Xtr, ytr, Wn, LAM)
        # backward pass: numerical gradient descent on premise params
        def loss(Cc, Ss):
            return ((predict(Xtr, Cc, Ss, theta) - ytr)**2).mean()
        eps = 1e-4
        gC = np.zeros_like(C); gS = np.zeros_like(S)
        base = loss(C, S)
        for k in range(K):
            for j in range(7):
                C2 = C.copy(); C2[k,j] += eps
                gC[k,j] = (loss(C2,S)-base)/eps
                S2 = S.copy(); S2[k,j] += eps
                gS[k,j] = (loss(C,S2)-base)/eps
        C -= LR*gC/ (np.abs(gC).max()+1e-9)
        S -= LR*gS/ (np.abs(gS).max()+1e-9)
        S = np.clip(S, 0.05, 2.0)
        tr = rmse(predict(Xtr,C,S,theta), ytr)
        vl = rmse(predict(Xval,C,S,theta), yval)
        curves["train"].append(tr); curves["val"].append(vl)
        if vl < best[0] - 1e-5:
            best = (vl, C.copy(), S.copy(), theta.copy(), ep + 1); wait = 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    _, C, S, theta, bep = best
    return C, S, theta, curves, bep

def pairwise_acc(yt, yp):
    ok = tot = 0
    for i in range(4):
        for j in range(i+1,4):
            if abs(yt[i]-yt[j]) < 1e-9: continue
            tot += 1
            if (yt[i]-yt[j])*(yp[i]-yp[j]) > 0: ok += 1
    return ok, tot

folds = []
preds_all = {"anfis":np.zeros(24), "mlr":np.zeros(24), "fis":np.zeros(24)}
curves_saved = None
for ti, k in enumerate(risks):
    te = risk_id == ti
    tr = ~te
    # inner validation: one training risk held out (next risk cyclically)
    vi = (ti+1) % 6
    val = risk_id == vi
    tr_in = tr & ~val
    Xtr, ytr = X_all[tr_in], y_all[tr_in]
    Xval, yval = X_all[val], y_all[val]
    Xte, yte = X_all[te], y_all[te]
    # ANFIS
    C, S, theta, curves, bep = train_anfis(Xtr, ytr, Xval, yval)
    yp = predict(Xte, C, S, theta)
    preds_all["anfis"][te] = yp
    if ti == 5: curves_saved = curves
    # MLR on full training (tr) via OLS
    Xd = np.hstack([X_all[tr], np.ones((tr.sum(),1))])
    beta, *_ = np.linalg.lstsq(Xd, y_all[tr], rcond=None)
    yp_mlr = np.hstack([Xte, np.ones((4,1))]) @ beta
    preds_all["mlr"][te] = yp_mlr
    # FIS: cluster-initialised, consequents LSE once, NO premise adaptation
    C0, S0 = init_premise(X_all[tr])
    th0, _ = lse_consequents(X_all[tr], y_all[tr], firing(X_all[tr],C0,S0)[1], LAM)
    yp_fis = predict(Xte, C0, S0, th0)
    preds_all["fis"][te] = yp_fis
    row = dict(fold=k, best_epoch=int(bep),
        anfis=dict(rmse=rmse(yte,yp), mae=float(np.abs(yte-yp).mean()),
                   pred=[round(float(v),3) for v in yp], obs=[round(float(v),3) for v in yte],
                   top1=int(np.argmax(yp)==np.argmax(yte)), pair=pairwise_acc(yte,yp)),
        mlr=dict(rmse=rmse(yte,yp_mlr), mae=float(np.abs(yte-yp_mlr).mean()),
                 top1=int(np.argmax(yp_mlr)==np.argmax(yte)), pair=pairwise_acc(yte,yp_mlr)),
        fis=dict(rmse=rmse(yte,yp_fis), mae=float(np.abs(yte-yp_fis).mean()),
                 top1=int(np.argmax(yp_fis)==np.argmax(yte)), pair=pairwise_acc(yte,yp_fis)))
    folds.append(row)
    print(k, "ANFIS rmse=%.3f" % row["anfis"]["rmse"], "MLR rmse=%.3f" % row["mlr"]["rmse"],
          "FIS rmse=%.3f" % row["fis"]["rmse"], "pair", row["anfis"]["pair"], "top1", row["anfis"]["top1"])

def summarize(name):
    r = [f[name]["rmse"] for f in folds]; m = [f[name]["mae"] for f in folds]
    pa = [f[name]["pair"] for f in folds]
    ok = sum(p[0] for p in pa); tot = sum(p[1] for p in pa)
    top1 = sum(f[name]["top1"] for f in folds)
    yp = preds_all[name]
    ss_res = ((y_all-yp)**2).sum(); ss_tot = ((y_all-y_all.mean())**2).sum()
    return dict(rmse_mean=float(np.mean(r)), rmse_sd=float(np.std(r, ddof=1)),
                mae_mean=float(np.mean(m)), mae_sd=float(np.std(m, ddof=1)),
                pooled_rmse=rmse(y_all, yp),
                pair_acc=ok/tot*100, top1=f"{top1}/6", r2=float(1-ss_res/ss_tot))
summary = {n: summarize(n) for n in ["anfis","mlr","fis"]}
for n,s in summary.items(): print(n, {k2: (round(v2,4) if isinstance(v2,float) else v2) for k2,v2 in s.items()})

json.dump(dict(folds=folds, summary=summary, curves=curves_saved,
               preds=({k2:v2.tolist() for k2,v2 in preds_all.items()}), y=y_all.tolist()),
          open("part2_results.json","w"), indent=1)
print("saved part2_results.json")
