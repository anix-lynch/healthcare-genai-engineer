"""Offline eval — golden set runner + regression gate."""
from .ragas_runner import run_eval, score_query
from .regression_gate import check_regression, RegressionViolation
