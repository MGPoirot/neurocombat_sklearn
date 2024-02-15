import pytest

from sklearn.utils.estimator_checks import check_estimator

from neurocombat_sklearn import CombatTransformer

@pytest.mark.parametrize(
    "Estimator", [CombatTransformer]
)
def test_all_transformers(Estimator):
    return check_estimator(Estimator)
