"""
=====================================================================
Test2
=====================================================================
"""
print(__doc__)

from neurocombat_sklearn import CombatTransformer
import pandas as pd
import numpy as np

# Loading data
data = np.load('examples/data/bladder-expr.npy')
covars = pd.read_csv('examples/data/bladder-pheno.txt', delimiter='\t')

# Creating model
model = CombatTransformer()

# Fitting model
# make sure that your inputs are 2D, e.g. shape [n_samples, n_discrete_covariates]
model.fit(data,
          covars[['batch']],
          covars[['cancer']],
          covars[['age']])

# Harmonize data
# could be performed together with fitt by using .fit_transform method
data_combat = model.transform(data,
                              covars[['batch']],
                              covars[['cancer']],
                              covars[['age']])