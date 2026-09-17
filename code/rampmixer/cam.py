"""Lagged environmental response fitting and Coupling Attribution Matrix.

See code/IMPLEMENTATION.md for the reference measure and fitting choices.
"""
import numpy as np
from scipy.signal import savgol_coeffs
from scipy.linalg import svd
from sklearn.linear_model import Ridge

EPS = 1e-6


def lag_states(env, lag=3):
    """Left replication stays inside the supplied historical window."""
    n = len(env)
    return np.stack([env[np.maximum(np.arange(n)-k, 0)]
                     for k in range(lag+1)], axis=-1)


def causal_trend(power, width=13):
    """One-sided local quadratic SG endpoint, never centered in future data."""
    power = np.asarray(power)
    out = np.empty_like(power, dtype=float)
    coeffs = {}
    for t in range(len(power)):
        w = min(t+1, width)
        if w < 3:
            out[t] = power[t]
        else:
            if w not in coeffs:
                coeffs[w] = savgol_coeffs(w, 2, pos=w-1, use='dot')
            out[t] = coeffs[w] @ power[t-w+1:t+1]
    return out


class ANOVAResponse:
    """Centered grouped polynomial main terms + selected tensor-product pairs.

    Functional ANOVA is defined under the product of empirical driver-group
    marginals, NOT orthogonality under the correlated joint distribution.
    Basis, rank and regularization are explicit new implementation choices.
    """
    def __init__(self, gamma=.4, rank=4, ridge=100.):
        self.gamma, self.rank, self.ridge = gamma, rank, ridge

    def bases(self, states, fit=False):
        if fit:
            self.transforms = []
        out = []
        for i in range(states.shape[1]):
            if fit:
                mean = states[:, i].mean(0)
                scale = states[:, i].std(0)
                scale[scale < 1e-8] = 1.
            else:
                mean, scale, center, rotation = self.transforms[i]
            z = (states[:, i]-mean)/scale
            cols = [z, z*z]
            for a in range(z.shape[1]):
                for b in range(a+1, z.shape[1]):
                    cols.append((z[:, a]*z[:, b])[:, None])
            poly = np.concatenate(cols, axis=1)
            if fit:
                center = poly.mean(0)
                _, singular, vt = svd(poly-center, full_matrices=False,
                                     check_finite=False)
                rank = min(self.rank, int(np.sum(singular > 1e-7*max(singular[0], 1.))))
                rotation = vt[:rank].T/np.maximum(singular[:rank]/np.sqrt(len(z)), 1e-8)
                self.transforms.append((mean, scale, center, rotation))
            out.append((poly-center) @ rotation)
        return out

    def design(self, bases):
        blocks = list(bases)
        for i, j in self.pairs:
            blocks.append(np.einsum('ni,nj->nij', bases[i], bases[j]).reshape(len(bases[i]), -1))
        return np.concatenate(blocks, axis=1), [b.shape[1] for b in blocks]

    def fit(self, states, fluctuation):
        if not 0 <= self.gamma <= 1 or self.rank < 1 or self.ridge <= 0:
            raise ValueError('Invalid response-model settings')
        bases = self.bases(states, True)
        with np.errstate(invalid='ignore', divide='ignore'):
            self.correlation = np.nan_to_num(np.corrcoef(states[:, :, 0].T))
        self.pairs = [(i, j) for i in range(len(bases)) for j in range(i+1, len(bases))
                      if abs(self.correlation[i, j]) > self.gamma
                      and bases[i].shape[1] and bases[j].shape[1]]
        x, self.widths = self.design(bases)
        if not x.shape[1]:
            raise ValueError('Environmental observations contain no varying response basis')
        self.model = Ridge(alpha=self.ridge, solver='cholesky').fit(x, fluctuation)
        self.train_r2 = float(self.model.score(x, fluctuation))
        return self

    def effects(self, states):
        x, widths = self.design(self.bases(states))
        offsets = np.cumsum([0]+widths)
        terms = np.column_stack([x[:, a:b] @ self.model.coef_[a:b]
                                 for a, b in zip(offsets[:-1], offsets[1:])])
        m = states.shape[1]
        return terms[:, :m], terms[:, m:]

    def attribute(self, states):
        main, pair = self.effects(states)
        strength = np.abs(main).copy()
        for k, (i, j) in enumerate(self.pairs):
            wi = (np.abs(main[:, i])+EPS)/(np.abs(main[:, i])+np.abs(main[:, j])+2*EPS)
            strength[:, i] += wi*np.abs(pair[:, k])
            strength[:, j] += (1-wi)*np.abs(pair[:, k])
        return strength/(strength.sum(1, keepdims=True)+EPS)
