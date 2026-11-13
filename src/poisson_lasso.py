"""Poisson L1 fit: mean(exp(eta) - y*eta) + alpha*sum(abs(beta)).

The intercept is unpenalized. beta = positive - negative with nonnegative
parts permits a smooth bound-constrained solve and exact zero coefficients.
Every returned solution is checked against the original L1 KKT conditions.
"""
import numpy as np
from scipy.optimize import minimize


def kkt_error(x, y, intercept, coef, alpha):
    residual = np.exp(intercept + x @ coef) - y
    gradient = x.T @ residual / len(y)
    active = np.abs(coef) > 1e-8
    violation = np.maximum(np.abs(gradient) - alpha, 0)
    violation[active] = np.abs(gradient[active] + alpha * np.sign(coef[active]))
    return float(max(abs(residual.mean()), violation.max(initial=0)))


def fit_poisson_lasso(x, y, alpha, warm=None, tolerance=1e-5):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 2 or y.shape != (len(x),) or len(y) == 0
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or (y < 0).any() or y.mean() <= 0 or not np.isfinite(alpha) or alpha < 0):
        raise ValueError('Finite design, nonnegative target with positive mean, and alpha >= 0 required')
    p = x.shape[1]
    coef = np.zeros(p) if warm is None else np.asarray(warm['coef'], float)
    intercept = np.log(y.mean()) if warm is None else float(warm['intercept'])
    start = np.r_[intercept, np.maximum(coef, 0), np.maximum(-coef, 0)]

    def objective(theta):
        beta = theta[1:p+1] - theta[p+1:]
        eta = theta[0] + x @ beta
        if np.max(eta) > 500:
            return np.inf, np.zeros_like(theta)
        mu = np.exp(eta)
        residual = (mu - y) / len(y)
        gradient = x.T @ residual
        value = np.mean(mu - y * eta) + alpha * theta[1:].sum()
        return value, np.r_[residual.sum(), gradient + alpha, -gradient + alpha]

    iterations = 0
    # Very correlated columns can meet the function-change stopping rule before
    # the L1 conditions. In that case polish with the function-change rule off.
    for attempt in range(2):
        result = minimize(objective, start, method='L-BFGS-B', jac=True,
            bounds=[(None, None)] + [(0, None)] * (2*p),
            options=dict(maxiter=12000, maxfun=50000, maxls=50,
                         maxcor=20 if attempt == 0 else 50,
                         ftol=1e-14 if attempt == 0 else 0, gtol=1e-9))
        iterations += result.nit
        coef = result.x[1:p+1] - result.x[p+1:]
        coef[np.abs(coef) < 1e-8] = 0
        error = kkt_error(x, y, result.x[0], coef, alpha)
        if np.isfinite(error) and error <= tolerance:
            break
        start = np.r_[result.x[0], np.maximum(coef, 0), np.maximum(-coef, 0)]
    if not np.isfinite(error) or error > tolerance:
        raise RuntimeError(f'L1 Poisson failed KKT check: {error:g}; {result.message}')
    return dict(intercept=float(result.x[0]), coef=coef, alpha=float(alpha),
                kkt_error=error, iterations=int(iterations), polishing_passes=attempt,
                objective=float(result.fun), optimizer_message=str(result.message),
                converged=True, nonzero=int(np.count_nonzero(coef)))


def predict_poisson_lasso(model, x):
    mu = np.exp(model['intercept'] + np.asarray(x) @ model['coef'])
    if not np.isfinite(mu).all() or (mu <= 0).any():
        raise ValueError('Invalid predicted goal mean')
    return mu
