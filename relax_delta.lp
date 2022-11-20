\* Relaxation_Delta_LP *\
Minimize
Minimize_the_Proportion_of_Uncertainty_Removed: delta
Subject To
_C1: - 7839.85593816 delta + eps_2 = 0
_C2: - 11759.7839072 delta + eps_4 = 0
_C3: eps_2 + eps_4 >= 4599.6398454
Bounds
 delta <= 1
 eps_2 <= 7839.85593816
 eps_4 <= 11759.7839072
End
