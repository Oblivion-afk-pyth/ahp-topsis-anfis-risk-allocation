"""Optional audit: check the workbook against the extracted expert inputs.
Requires openpyxl in addition to the NumPy requirement for the analysis.
Does not alter the source workbook or fill missing values.
"""
from pathlib import Path
import json
import numpy as np
import openpyxl

base=Path(__file__).resolve().parent
w=openpyxl.load_workbook(base/'37_Experts_Data.xlsx',data_only=True)
A=np.ones((37,7,7)); X=np.full((37,4,7),np.nan); pairs=set(); scores=set()
for expert,pair,value in list(w.worksheets[0].values)[1:]:
    e=int(expert.split('_')[1])-1
    j,k=[int(c[1:])-1 for c in pair.split(' vs ')]
    assert (e,j,k) not in pairs
    pairs.add((e,j,k)); assert value>0
    A[e,j,k]=value;A[e,k,j]=1/value
for expert,alternative,*values in list(w.worksheets[1].values)[1:]:
    e=int(expert.split('_')[1])-1
    a=['Client (A1)','Contractor (A2)','Consultant (A3)','Shared (A4)'].index(alternative)
    assert (e,a) not in scores
    scores.add((e,a));X[e,a]=values
assert len(pairs)==777 and len(scores)==148
assert np.isfinite(X).all() and (X>=1).all() and (X<=9).all()
data=json.loads((base/'expert_inputs.json').read_text())
assert np.array_equal(A,np.array(data['AHP']))
assert np.array_equal(X,np.array(data['R1_scores']))
print('PASS: all 777 pairwise comparisons and 1,036 R1 scores match the unmodified workbook.')
