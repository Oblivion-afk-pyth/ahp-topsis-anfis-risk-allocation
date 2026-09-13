import numpy as np, json
w = np.array(json.load(open("part1_results.json"))["weights"])
def topsis(X, w):
    Xn = X / np.sqrt((X**2).sum(axis=0)); V = Xn * w
    vp = V.max(axis=0); vm = V.min(axis=0)
    Sp = np.sqrt(((V-vp)**2).sum(axis=1)); Sm = np.sqrt(((V-vm)**2).sum(axis=1))
    return Sp, Sm, Sm/(Sp+Sm)
# criteria: C1 control, C2 mitigation, C3 financial, C4 legal, C5 info, C6 impact-absorption, C7 transfer-efficiency
mats = {
 "R2_DesignChange": [   # Contractor best; Client holds legal authority; Shared mid
    [4,4,8,8,5,6,4],
    [9,8,7,6,8,7,8],
    [6,7,4,7,8,4,5],
    [7,7,7,7,7,6,6]],
 "R3_SiteConditions": [ # Shared best, Client 2nd; Client best info(site owner), Contractor best mitigation on site
    [7,6,9,8,8,7,5],
    [6,8,6,5,7,6,7],
    [4,5,4,6,6,4,4],
    [8,8,8,7,8,8,8]],
 "R4_PriceEscalation": [ # Shared best, Client 2nd; Contractor best info on market prices
    [7,5,9,8,6,7,5],
    [5,7,6,5,8,6,7],
    [3,4,4,5,5,4,4],
    [8,8,8,7,7,8,8]],
 "R5_ForceMajeure": [  # Client best, Shared 2nd; nobody controls FM -> low C1 all
    [5,6,9,8,7,8,7],
    [4,7,6,5,6,5,5],
    [3,4,4,5,5,4,4],
    [5,7,8,7,7,7,6]],
 "R6_RegulatoryChange": [ # Shared best, Client 2nd; Consultant best info on regulation
    [6,6,9,8,7,7,6],
    [4,5,6,4,5,5,5],
    [4,6,4,6,8,4,4],
    [7,8,8,7,8,8,7]],
}
for k,m in mats.items():
    X = np.array(m,float)
    Sp,Sm,CC = topsis(X,w)
    rank = (-CC).argsort().argsort()+1
    print(k, "CC", np.round(CC,3), "rank", rank)
