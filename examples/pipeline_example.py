import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from neurocombat_sklearn import CombatHarmonizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold

# Load data and covariates
covars = pd.read_csv(r'D:\repositories\neurocombat_sklearn\examples\data\bladder-pheno.txt', delimiter='\t', index_col='cel')
data = pd.DataFrame(np.load(r'D:\repositories\neurocombat_sklearn\examples\data\bladder-expr.npy'), index=covars.index)

# Set a random binary label
y = covars.pop('outcome') == 'sTCC-CIS'

# We can not work from a single data set
X = pd.concat([covars[['batch', 'cancer', 'age']], data], axis=1)

# Define our harmonizer
harmonizer = CombatHarmonizer(sites='batch', discrete_covariates=['cancer'], continuous_covariates=['age'], retain=True)

# The Combat harmonizer still works as stand alone
X_hat = harmonizer.fit_transform(X)

harmonizer.retain = False

# Create the machine learning pipeline
pipeline = make_pipeline(
    harmonizer,
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)

# Since we need to make sure all sites and labels are always present we keep splits low.
# Alternatively, you need more samples, less classes or built your own stratifier.
cv = KFold(n_splits=2, shuffle=True, random_state=0)

# Since our labels are made up, the LogisticRegressor wont fit well, but it runs!
score = cross_val_score(pipeline, X, y, cv=cv)

# Report performance
print(f"Mean cross-validation accuracy is: {score.mean():.1%}")
