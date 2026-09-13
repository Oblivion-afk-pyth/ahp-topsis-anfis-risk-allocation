"""Part 1: AHP + TOPSIS + sensitivity + Monte Carlo recomputation."""
import numpy as np
import json

np.set_printoptions(precision=4, suppress=True)

# ---------- AHP (Table 2) ----------
A = np.array([
    [1.00, 1.62, 2.14, 2.83, 3.47, 1.91, 2.45],
    [0.62, 1.00, 1.73, 2.21, 2.89, 1.58, 2.07],
    [0.47, 0.58, 1.00, 1.51, 2.03, 1.22, 1.67],
    [0.35, 0.45, 0.66, 1.00, 1.62, 0.89, 1.41],
    [0.29, 0.35, 0.49, 0.62, 1.00, 0.73, 1.12],
    [0.52, 0.63, 0.82, 1.12, 1.37, 1.00, 1.53],
    [0.41, 0.48, 0.60, 0.71, 0.89, 0.65, 1.00],
])
n = 7
eigvals, eigvecs = np.linalg.eig(A)
imax = np.argmax(eigvals.real)
lmax = eigvals.real[imax]
w = np.abs(eigvecs[:, imax].real)
w = w / w.sum()
CI = (lmax - n) / (n - 1)
RI = 1.32  # Saaty RI for n=7
CR = CI / RI
print("AHP weights (eigenvector):", np.round(w, 4))
print("lambda_max=%.4f CI=%.5f CR=%.5f" % (lmax, CI, CR))

# Paper's Table 3 weights
w_paper = np.array([0.27, 0.20, 0.14, 0.10, 0.08, 0.12, 0.09])
print("Paper weights:", w_paper, "diff:", np.round(w - w_paper, 3))

def topsis(X, w):
    Xn = X / np.sqrt((X**2).sum(axis=0))
    V = Xn * w
    vp = V.max(axis=0); vm = V.min(axis=0)
    Sp = np.sqrt(((V - vp)**2).sum(axis=1))
    Sm = np.sqrt(((V - vm)**2).sum(axis=1))
    CC = Sm / (Sp + Sm)
    return Sp, Sm, CC

# ---------- R1 (Table 4) ----------
X1 = np.array([
    [6,5,9,8,7,6,5],
    [8,8,7,6,6,7,7],
    [5,6,5,7,8,5,4],
    [9,9,8,8,9,8,9],
], dtype=float)
Sp, Sm, CC = topsis(X1, w)
print("\nR1 with eigenvector weights: Sp", np.round(Sp,3), "Sm", np.round(Sm,3), "CC", np.round(CC,3))
Sp2, Sm2, CC2 = topsis(X1, w_paper)
print("R1 with paper weights:      Sp", np.round(Sp2,3), "Sm", np.round(Sm2,3), "CC", np.round(CC2,3))

# ---------- R2-R6 reconstructed matrices ----------
# rows: Client A1, Contractor A2, Consultant A3, Shared A4
matrices = {
 "R2_DesignChange": np.array([
    [4,4,7,7,5,5,4],
    [9,8,7,6,8,8,8],
    [6,6,4,6,7,6,5],
    [7,7,7,7,7,7,7],
 ], dtype=float),
 "R3_SiteConditions": np.array([
    [7,6,8,8,6,7,6],
    [6,7,6,5,7,6,6],
    [4,5,4,5,6,4,4],
    [8,8,8,8,8,8,8],
 ], dtype=float),
 "R4_PriceEscalation": np.array([
    [7,6,8,7,6,7,6],
    [5,6,6,5,6,6,6],
    [3,4,4,4,5,4,3],
    [8,8,8,8,7,8,8],
 ], dtype=float),
 "R5_ForceMajeure": np.array([
    [8,7,9,8,7,8,7],
    [5,6,6,5,6,5,5],
    [4,4,4,5,5,4,4],
    [7,7,8,7,7,8,7],
 ], dtype=float),
 "R6_RegulatoryChange": np.array([
    [7,6,8,8,7,7,6],
    [5,5,6,5,5,5,5],
    [4,5,4,6,6,4,4],
    [8,8,8,8,8,8,8],
 ], dtype=float),
}
all_results = {"R1_PaymentDelay": dict(X=X1.tolist(), Sp=Sp.tolist(), Sm=Sm.tolist(), CC=CC.tolist())}
print()
for k, X in matrices.items():
    Sp, Sm, CC = topsis(X, w)
    rank = (-CC).argsort().argsort() + 1
    print(k, "CC", np.round(CC,3), "ranks", rank)
    all_results[k] = dict(X=X.tolist(), Sp=Sp.tolist(), Sm=Sm.tolist(), CC=CC.tolist())

# ---------- OAT sensitivity (all 28 scenarios), R1 ----------
print("\nOAT sensitivity R1:")
sens = []
for j in range(7):
    for d in (0.10, -0.10, 0.20, -0.20):
        w2 = w.copy()
        w2[j] = w[j] * (1 + d)
        others = [i for i in range(7) if i != j]
        w2[others] = w[others] * (1 - w2[j]) / w[others].sum()
        _,_,CCs = topsis(X1, w2)
        rank_a4 = int((-CCs).argsort().argsort()[3] + 1)
        sens.append(dict(crit=j+1, pct=d, wj=round(float(w2[j]),4), CC_A4=round(float(CCs[3]),3),
                         CC_all=[round(float(c),3) for c in CCs], rank_A4=rank_a4))
        print(f"C{j+1} {d:+.0%} w={w2[j]:.3f} CC_A4={CCs[3]:.3f} rank={rank_a4}")

# ---------- Monte Carlo global sensitivity ----------
rng = np.random.default_rng(42)
N = 10000
stay_first = 0
cc_a4 = []
for _ in range(N):
    pert = rng.uniform(0.8, 1.2, size=7)
    w2 = w * pert; w2 = w2 / w2.sum()
    _,_,CCs = topsis(X1, w2)
    cc_a4.append(CCs[3])
    if CCs.argmax() == 3: stay_first += 1
cc_a4 = np.array(cc_a4)
mc = dict(N=N, pct_first=stay_first/N*100, cc_mean=float(cc_a4.mean()), cc_std=float(cc_a4.std()),
          cc_min=float(cc_a4.min()), cc_max=float(cc_a4.max()))
print("\nMonte Carlo:", mc)

# also MC for all six risks
mc_all = {}
for k, res in all_results.items():
    X = np.array(res["X"]); base_best = int(np.argmax(res["CC"]))
    cnt = 0
    for _ in range(2000):
        pert = rng.uniform(0.8, 1.2, size=7)
        w2 = w * pert; w2 = w2/w2.sum()
        _,_,CCs = topsis(X, w2)
        if CCs.argmax() == base_best: cnt += 1
    mc_all[k] = cnt/2000*100
print("MC all risks (% best unchanged):", mc_all)

out = dict(weights=w.tolist(), lmax=float(lmax), CI=float(CI), CR=float(CR),
           results=all_results, sens=sens, mc=mc, mc_all=mc_all)
json.dump(out, open("part1_results.json","w"), indent=1)
print("\nsaved part1_results.json")
