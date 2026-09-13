import numpy as np, json
w = np.array(json.load(open("part1_results.json"))["weights"])
def topsis(X, w):
    Xn = X / np.sqrt((X**2).sum(axis=0)); V = Xn * w
    vp = V.max(axis=0); vm = V.min(axis=0)
    Sp = np.sqrt(((V-vp)**2).sum(axis=1)); Sm = np.sqrt(((V-vm)**2).sum(axis=1))
    return Sp, Sm, Sm/(Sp+Sm)
mats = {
 "R2_DesignChange": [
    [4,4,8,8,5,6,4],
    [9,8,7,6,8,7,8],
    [6,7,4,7,8,4,5],
    [7,7,7,7,7,6,6]],
 "R3_SiteConditions": [
    [7,6,9,8,8,7,5],
    [6,8,6,5,7,6,7],
    [4,5,4,6,6,4,4],
    [8,8,8,7,8,8,8]],
 "R4_PriceEscalation": [
    [7,5,9,8,6,7,5],
    [5,7,6,5,8,6,7],
    [3,4,4,6,5,4,4],
    [8,8,8,7,7,8,8]],
 "R5_ForceMajeure": [
    [5,6,9,8,7,8,7],
    [4,7,6,5,6,5,5],
    [3,4,4,6,5,4,4],
    [5,7,8,7,7,7,6]],
 "R6_RegulatoryChange": [
    [6,6,9,8,7,7,6],
    [4,6,6,4,5,5,6],
    [4,5,4,6,8,4,4],
    [7,8,8,7,8,8,7]],
}
out = {}
for k,m in mats.items():
    X = np.array(m,float)
    Sp,Sm,CC = topsis(X,w)
    rank = (-CC).argsort().argsort()+1
    print(k, "CC", np.round(CC,3), "rank", rank)
    out[k] = dict(X=m, Sp=[round(float(v),4) for v in Sp], Sm=[round(float(v),4) for v in Sm], CC=[round(float(v),4) for v in CC])
json.dump(out, open("r2r6.json","w"), indent=1)
