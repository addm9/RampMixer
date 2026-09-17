"""Contribution-Guided Ramp Decomposition on historical windows only."""
import numpy as np
from numba import njit, prange

EPS = 1e-6


@njit(cache=True)
def _centers(u, freq, q):
    centers = np.zeros(u.shape[0])
    for i in range(u.shape[0]):
        numerator, denominator = 0., 0.
        for k in range(u.shape[1]):
            energy = q[k]*(u[i, k].real**2+u[i, k].imag**2)
            numerator += energy*freq[k]
            denominator += energy
        centers[i] = numerator/max(denominator, 1e-30)
    return centers


@njit(cache=True)
def _objective(u, p, c, alpha, centers, freq, q, overlap, fidelity=1.):
    value = 0.
    for k in range(len(freq)):
        total = 0j
        for i in range(len(u)):
            total += u[i, k]
            e = c[i, k]-u[i, k]
            value += fidelity*q[k]*(e.real**2+e.imag**2)
            value += alpha[i]*q[k]*(freq[k]-centers[i])**2*abs(u[i, k])**2
        value += q[k]*abs(p[k]-total)**2
    for i in range(len(u)):
        for j in range(i+1, len(u)):
            gram = 0.
            for k in range(len(freq)):
                gram += q[k]*(u[i, k]*np.conj(u[j, k])).real
            value += overlap*abs(gram)
    return value


@njit(cache=True)
def _solve(p, c, alpha, freq, q, overlap, max_iter, tol, fidelity):
    u = c.copy()
    centers = _centers(u, freq, q)
    initial = _objective(u, p, c, alpha, centers, freq, q, overlap, fidelity)
    accepted_increase = 0.
    stalled = 0
    delta = 1e30
    dnu = 1e30
    for iteration in range(max_iter):
        previous = u.copy()
        old_centers = centers.copy()
        for i in range(len(u)):
            others = u.sum(axis=0)-u[i]
            correction = np.zeros(len(freq), dtype=np.complex128)
            for j in range(len(u)):
                if i != j:
                    gram = 0.
                    for k in range(len(freq)):
                        gram += q[k]*(u[i, k]*np.conj(u[j, k])).real
                    correction += np.sign(gram)*u[j]
            candidate = (p-others+fidelity*c[i]-.5*overlap*correction)/(1+fidelity+alpha[i]*(freq-centers[i])**2)
            before = u[i].copy()
            baseline = _objective(u, p, c, alpha, centers, freq, q, overlap, fidelity)
            accepted = False
            for attempt in range(24):
                u[i] = before+(candidate-before)*(.5**attempt)
                new = _objective(u, p, c, alpha, centers, freq, q, overlap, fidelity)
                if new <= baseline + 1e-12*max(1., baseline):
                    accepted_increase = max(accepted_increase, new-baseline)
                    accepted = True
                    break
            if not accepted:
                u[i] = before
                stalled += 1
        centers = _centers(u, freq, q)
        delta = np.sqrt(np.sum(q*np.abs(u-previous)**2))/max(np.sqrt(np.sum(q*np.abs(previous)**2)), EPS)
        dnu = np.max(np.abs(centers-old_centers))
        if delta < tol and dnu < tol:
            break
    final = _objective(u, p, c, alpha, centers, freq, q, overlap, fidelity)
    stats = np.array([iteration+1, delta, dnu, 1. if delta < tol and dnu < tol else 0.,
                      initial, final, accepted_increase, stalled])
    return u, centers, stats


@njit(cache=True, parallel=True)
def _solve_batch(p, c, alpha, freq, q, overlap, max_iter, tol, fidelity=1.):
    u = np.empty_like(c)
    centers = np.zeros((len(c), c.shape[1]))
    stats = np.zeros((len(c), 8))
    for n in prange(len(c)):
        u[n], centers[n], stats[n] = _solve(p[n], c[n], alpha, freq, q, overlap, max_iter, tol, fidelity)
    return u, centers, stats


def decompose(power, weights, alpha, overlap=.1, max_iter=500, tol=1e-6, fidelity=1.):
    """Batched causal-window CRD; real modes via a one-sided spectral surrogate.

    Reflection pads historical values only. Frequencies are radians/retained
    observation. All outputs and termination diagnostics are returned.
    """
    power = np.asarray(power, dtype=float)
    weights = np.asarray(weights, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if power.ndim != 2 or not len(power) or power.shape[1] < 2:
        raise ValueError('Power must have shape [batch, history_length >= 2]')
    if weights.ndim != 3 or weights.shape[:2] != power.shape or not weights.shape[2]:
        raise ValueError('CAM must have shape [batch, history_length, drivers]')
    if alpha.shape != (weights.shape[2],) or np.any(alpha < 0):
        raise ValueError('Provide one nonnegative bandwidth per driver')
    if not all(np.isfinite(a).all() for a in (power, weights, alpha)):
        raise ValueError('Decomposition inputs must be finite')
    if np.any(weights < 0) or np.any(weights.sum(-1) > 1+1e-6):
        raise ValueError('CAM entries must be nonnegative with row sums at most one')
    if max_iter < 1 or tol <= 0 or overlap < 0 or fidelity <= 0:
        raise ValueError('Invalid decomposition solver settings')
    n = power.shape[1]
    a = n//2
    p = np.pad(power, ((0, 0), (a, a)), mode='reflect')
    targets = weights*power[..., None]
    c = np.pad(targets.transpose(0, 2, 1), ((0, 0), (0, 0), (a, a)), mode='reflect')
    freq = 2*np.pi*np.fft.rfftfreq(p.shape[1])
    q = np.full(len(freq), 2.)
    q[0] = 1.
    if p.shape[1] % 2 == 0:
        q[-1] = 1.
    ps = np.fft.rfft(p, norm='ortho')
    cs = np.fft.rfft(c, norm='ortho')
    us, centers, stats = _solve_batch(ps, cs, np.asarray(alpha), freq, q, overlap, max_iter, tol, fidelity)
    modes = np.fft.irfft(us, n=p.shape[1], norm='ortho')[..., a:a+n].transpose(0, 2, 1)
    residual = power-modes.sum(-1)
    return np.concatenate([modes, residual[..., None]], axis=-1), centers, stats
