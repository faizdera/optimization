import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# =====================================================================
# SOLVERS
#   x is a vector of any length. Stopping test: |x_{k+1} - x_k| < tol.
#   grad_f may be supplied; if omitted, jax.grad(f) is used.
# =====================================================================

def gradient_descent(f, x0, lr=0.1, tol=1e-8, max_iter=40000, grad_f=None):
    if grad_f is None:
        grad_f = jax.grad(f)
    x = jnp.array(x0, dtype=float)
    hist = [np.array(x)]
    for i in range(1, max_iter + 1):
        x_new = x - lr * grad_f(x)
        hist.append(np.array(x_new))
        if jnp.linalg.norm(x_new - x) < tol:
            return np.array(x_new), i, np.array(hist), True
        x = x_new
    return np.array(x), max_iter, np.array(hist), False


def gradient_descent_momentum(f, x0, lr=0.1, beta=0.85, ema=False,
                              tol=1e-8, max_iter=40000, grad_f=None):
    """ema=False : v = beta*v + g            classical heavy ball
       ema=True  : v = beta*v + (1-beta)*g   exponential moving average, as in adam"""
    if grad_f is None:
        grad_f = jax.grad(f)
    x = jnp.array(x0, dtype=float)
    v = jnp.zeros_like(x)
    hist = [np.array(x)]
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        v = beta*v + ((1 - beta)*g if ema else g)
        x_new = x - lr * v
        hist.append(np.array(x_new))
        if jnp.linalg.norm(x_new - x) < tol:
            return np.array(x_new), i, np.array(hist), True
        x = x_new
    return np.array(x), max_iter, np.array(hist), False


def gradient_descent_adam(f, x0, lr=0.05, b1=0.9, b2=0.999, eps=1e-8,
                          tol=1e-8, max_iter=40000, grad_f=None):
    """Returns the usual four values plus the internal m_hat and v_hat sequences."""
    if grad_f is None:
        grad_f = jax.grad(f)
    x = jnp.array(x0, dtype=float)
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    hist = [np.array(x)]
    m_rec, v_rec = [], []
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g**2
        m_hat = m / (1 - b1**i)
        v_hat = v / (1 - b2**i)
        m_rec.append(np.array(m_hat)); v_rec.append(np.array(v_hat))
        x_new = x - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        hist.append(np.array(x_new))
        if jnp.linalg.norm(x_new - x) < tol:
            return (np.array(x_new), i, np.array(hist), True,
                    np.array(m_rec), np.array(v_rec))
        x = x_new
    return np.array(x), max_iter, np.array(hist), False, np.array(m_rec), np.array(v_rec)


def steps(h):
    d = np.diff(h, axis=0)
    return np.linalg.norm(d.reshape(len(d), -1), axis=1)


def overshoot(h):
    xs_ = h[:, 0]
    flips = np.sign(xs_[:-1]) * np.sign(xs_[1:]) < 0
    sc = int(np.sum(flips))
    if sc == 0:
        return 0, 0.0
    first = int(np.argmax(flips)) + 1
    return sc, float(np.max(np.abs(xs_[first:])))


# =====================================================================
# PROBLEM 1
# =====================================================================

def f1(x):
    return x[0]**2

def f1_grad(x):
    return jnp.array([2*x[0]])


plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 9})
C = {'plain': '#1f4e79', 'momentum': '#c0504d', 'adam': '#4f8a10'}
TOL = 1e-8
STAT = 0.0

print("=" * 78)
print("PROBLEM 1     f(x) = x^2     unconstrained")
print("=" * 78)
print("df/dx = 2x = 0  ->  one stationary point at x = 0")
print()
print("Gradient check, analytic 2x against jax.grad:")
for p in ([2.0], [-7.0], [0.35]):
    a = float(f1_grad(jnp.array(p))[0]); j = float(jax.grad(f1)(jnp.array(p))[0])
    print(f"   x = {p[0]:6.2f}   analytic {a:11.6f}   jax {j:11.6f}   difference {abs(a-j):.1e}")
print()


# ---------------------------------------------------------------------
# 1. ITERATION COUNT AGAINST STARTING POINT
# ---------------------------------------------------------------------

starts = [2, 10, 50, 100, 200, 500, 1000]

print("=" * 78)
print("1. ITERATION COUNT AGAINST STARTING POINT")
print("=" * 78)
print(f"{'x0':>7} {'plain':>8} {'momentum':>10} {'adam':>8}")
print("-" * 78)
it_p, it_m, it_a = [], [], []
for x0 in starts:
    _, a_, _, _ = gradient_descent(f1, [float(x0)], lr=0.1, grad_f=f1_grad)
    _, b_, _, _ = gradient_descent_momentum(f1, [float(x0)], lr=0.1, beta=0.85, grad_f=f1_grad)
    _, c_, _, _, _, _ = gradient_descent_adam(f1, [float(x0)], lr=0.05, grad_f=f1_grad)
    it_p.append(a_); it_m.append(b_); it_a.append(c_)
    print(f"{x0:>7} {a_:8d} {b_:10d} {c_:8d}")
print()
print("plain and momentum multiply the remaining distance by a constant factor each")
print("step, so the count grows with log(x0): a 500-fold increase in x0 costs plain")
print("gradient descent 27 extra iterations.")
print("adam holds its step near lr because of the division by sqrt(v_hat), so the")
print("distance has to be walked rather than contracted and the count grows in")
print("proportion to x0.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(starts, it_p, 'o-', color=C['plain'], lw=1.6, label='plain')
ax[0].plot(starts, it_m, 'o-', color=C['momentum'], lw=1.6, label='momentum')
ax[0].set_xlabel('starting point x₀'); ax[0].set_ylabel('iterations')
ax[0].set_title('Iteration count against starting point')
ax[0].legend(fontsize=8)

ax[1].plot(starts, it_p, 'o-', color=C['plain'], lw=1.6, label='plain')
ax[1].plot(starts, it_m, 'o-', color=C['momentum'], lw=1.6, label='momentum')
ax[1].plot(starts, it_a, 'o-', color=C['adam'], lw=1.6, label='adam')
ax[1].set_xlabel('starting point x₀'); ax[1].set_ylabel('iterations')
ax[1].set_title('Same axes with adam included')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------
# 2. THE THREE METHODS FROM ONE STARTING POINT
# ---------------------------------------------------------------------

_, n_p, h_p, _ = gradient_descent(f1, [10.0], lr=0.1, grad_f=f1_grad)
_, n_m, h_m, _ = gradient_descent_momentum(f1, [10.0], lr=0.1, beta=0.85, grad_f=f1_grad)
_, n_a, h_a, _, m_a, v_a = gradient_descent_adam(f1, [10.0], lr=0.05, grad_f=f1_grad)
runs = {'plain': (h_p, n_p), 'momentum': (h_m, n_m), 'adam': (h_a, n_a)}

xg = np.linspace(-12, 12, 400)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(xg, xg**2, 'k-', lw=1.2, label='f(x)')
ax[0].axvline(STAT, color='gray', ls=':', lw=1.2)
ax[0].text(STAT + 0.3, 138, 'stationary point', fontsize=7.5, color='gray', va='top')
for k, (h, n) in runs.items():
    hh = h[:, 0]
    ax[0].plot(hh, hh**2, 'o-', ms=3.5, lw=1.2, color=C[k],
               markevery=max(1, len(hh)//25), label=f'{k} ({n})')
ax[0].plot(10.0, 100.0, 'ks', ms=7, label='x₀')
ax[0].set_xlabel('x'); ax[0].set_ylabel('f(x)')
ax[0].set_title('Path from x₀ = 10')
ax[0].legend(fontsize=8)

for k, (h, n) in runs.items():
    ax[1].plot(h[:, 0], lw=1.5, color=C[k], label=k)
ax[1].axhline(STAT, color='gray', ls=':', lw=1.2)
ax[1].set_xlim(0, 300); ax[1].set_xlabel('iteration'); ax[1].set_ylabel('x')
ax[1].set_title('Position against iteration')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

cps = (0, 1, 2, 5, 10, 20, 50, 100, 200, 500)
print("=" * 98)
print("2. POSITION AT SELECTED ITERATIONS     x0 = 10")
print("=" * 98)
print(f"{'method':<10}" + "".join(f"{('k='+str(c)):>9}" for c in cps) + f"{'final k':>10}")
print("-" * 98)
for k, (h, n) in runs.items():
    cells = "".join((f"{h[c,0]:9.4f}" if c < len(h) else f"{'-':>9}") for c in cps)
    print(f"{k:<10}{cells}{n:>10}")
print()


# ---------------------------------------------------------------------
# 3. MOMENTUM
# ---------------------------------------------------------------------

print("=" * 78)
print("3. MOMENTUM")
print("=" * 78)
print("Reason for trying it: the update keeps a fraction beta of the previous step, so")
print("the iterate accumulates velocity along a consistent downhill direction and")
print("should reach the bottom in fewer iterations than a step set by the local")
print("gradient alone.")
print()
print("Two forms were implemented:")
print("   classical   v = beta*v + g")
print("   averaged    v = beta*v + (1-beta)*g      the form used for m inside adam")
print()
print(f"{'lr':>7} {'beta':>6} {'form':>11} {'iters':>8} {'sign changes':>14} {'peak |x| after':>16}")
print("-" * 78)

betas = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95]
mh_cls, mh_ema = {}, {}
for lr in (0.1, 0.5):
    for b in betas:
        _, nc, hc, _ = gradient_descent_momentum(f1, [10.0], lr=lr, beta=b,
                                                 ema=False, grad_f=f1_grad)
        sc, pk = overshoot(hc)
        if lr == 0.1:
            mh_cls[b] = hc
        print(f"{lr:7.2f} {b:6.2f} {'classical':>11} {nc:8d} {sc:14d} {pk:16.5f}")
    print("-" * 78)
for b in betas:
    _, ne, he, _ = gradient_descent_momentum(f1, [10.0], lr=0.1, beta=b,
                                             ema=True, grad_f=f1_grad)
    mh_ema[b] = he
    sc, pk = overshoot(he)
    print(f"{0.1:7.2f} {b:6.2f} {'averaged':>11} {ne:8d} {sc:14d} {pk:16.5f}")
print()
print("classical form at lr = 0.1: beta = 0.3 converges in 42 iterations against 87 for")
print("beta = 0, with no overshoot at all, which is what the reasoning above predicts.")
print("beta = 0.5 is still faster than beta = 0 but already crosses x = 0 seven times.")
print("From beta = 0.7 upwards the accumulated velocity dominates, the iterate swings")
print("well past the stationary point and the count rises: 103, 224 and 606 iterations")
print("with peak excursions of 3.0, 5.9 and 8.5 from a start at x = 10.")
print()
print("classical form at lr = 0.5: beta = 0 converges in 2 iterations, because")
print("lr = 1/f'' makes the contraction factor |1 - 2*lr| exactly zero, and every")
print("beta > 0 is far slower. The advantage seen at lr = 0.1 is therefore a")
print("correction for an undersized step rather than an improvement on a well")
print("chosen one, and the two parameters have to be varied together.")
print()
print("averaged form: the factor (1-beta) scales the effective step down by that same")
print("factor, so at fixed lr it is uniformly slower and produces no overshoot. The")
print("two forms are equivalent under lr_averaged = lr_classical / (1-beta).")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for b in betas:
    ax[0].plot(mh_cls[b][:, 0], lw=1.4, label=f'β = {b}')
ax[0].axhline(STAT, color='gray', ls=':', lw=1.2)
ax[0].set_xlim(0, 120); ax[0].set_ylim(-5, 11)
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('x')
ax[0].set_title('Momentum, classical form  v = βv + g,  lr = 0.1')
ax[0].legend(fontsize=8, ncol=2)

for b in betas:
    ax[1].plot(mh_ema[b][:, 0], lw=1.4, label=f'β = {b}')
ax[1].axhline(STAT, color='gray', ls=':', lw=1.2)
ax[1].set_xlim(0, 400); ax[1].set_ylim(-1, 11)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('x')
ax[1].set_title('Momentum, averaged form  v = βv + (1−β)g,  lr = 0.1')
ax[1].legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for b in betas:
    ax[0].semilogy(steps(mh_cls[b]), lw=1.3, label=f'β = {b}')
ax[0].axhline(TOL, color='gray', ls='--', lw=1)
ax[0].set_xlim(0, 700); ax[0].set_ylim(1e-10, 1e1)
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('|x$_{k+1}$ − x$_k$|')
ax[0].set_title('Step size, classical form')
ax[0].legend(fontsize=8, ncol=2)

for b in betas:
    ax[1].semilogy(steps(mh_ema[b]), lw=1.3, label=f'β = {b}')
ax[1].axhline(TOL, color='gray', ls='--', lw=1)
ax[1].set_xlim(0, 700); ax[1].set_ylim(1e-10, 1e1)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('|x$_{k+1}$ − x$_k$|')
ax[1].set_title('Step size, averaged form')
ax[1].legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------
# 4. ADAM
# ---------------------------------------------------------------------

print("=" * 78)
print("4. ADAM")
print("=" * 78)
print("Reason for trying it: plain gradient descent is stable only for lr < 2/f''(x*),")
print("so the step size cannot be set without knowing the curvature beforehand. Adam")
print("divides the step by sqrt(v_hat), a running estimate of the gradient magnitude,")
print("which removes the scale of f from the update. One value of lr should then work")
print("across problems of very different magnitude, and in more than one dimension")
print("each coordinate receives its own step size with no per-variable tuning.")
print()
print(f"{'lr':>7} {'b1':>6} {'b2':>8} {'iters':>8} {'sign changes':>14} {'peak |x| after':>16}")
print("-" * 78)

adam_cases = [(0.05, 0.9, 0.999), (0.02, 0.9, 0.999), (0.005, 0.9, 0.999),
              (0.20, 0.9, 0.999),
              (0.05, 0.7, 0.999), (0.05, 0.5, 0.999), (0.05, 0.0, 0.999),
              (0.05, 0.9, 0.99),  (0.05, 0.9, 0.90)]
A = {}
for lr, b1, b2 in adam_cases:
    _, na, ha, ca, ma, va = gradient_descent_adam(f1, [1.0], lr=lr, b1=b1, b2=b2,
                                                  grad_f=f1_grad)
    A[(lr, b1, b2)] = (ha, ma, va, na)
    sc, pk = overshoot(ha)
    tag = "" if ca else "   not converged"
    print(f"{lr:7.3f} {b1:6.2f} {b2:8.3f} {na:8d} {sc:14d} {pk:16.6f}{tag}")
print()
print("lr sets the size of the overshoot, since the normalised step stays close to lr")
print("while v_hat is still large.")
print("b1 is the momentum coefficient inside adam and is the main cause of the")
print("oscillation. Reducing it from 0.9 to 0.5 removes the overshoot entirely and")
print("cuts the iteration count, the same effect seen for momentum in section 3.")
print("b2 makes little difference between 0.999 and 0.99. At 0.90 the memory in v_hat")
print("is too short, the estimate becomes noisy and the run does not converge.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for lr, b1, b2 in [(0.20, 0.9, 0.999), (0.05, 0.9, 0.999),
                   (0.02, 0.9, 0.999), (0.005, 0.9, 0.999)]:
    ax[0].semilogy(np.abs(A[(lr, b1, b2)][0][:, 0]), lw=1.4, label=f'lr = {lr}')
    ax[0].axhline(lr, color=ax[0].lines[-1].get_color(), ls=':', lw=1)
ax[0].set_xlim(0, 900); ax[0].set_ylim(1e-10, 3e0)
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('|x|')
ax[0].set_title('Distance from the stationary point, varying lr')
ax[0].legend(fontsize=8)

for lr, b1, b2 in [(0.05, 0.9, 0.999), (0.05, 0.7, 0.999),
                   (0.05, 0.5, 0.999), (0.05, 0.0, 0.999)]:
    ax[1].plot(A[(lr, b1, b2)][0][:, 0], lw=1.4, label=f'β₁ = {b1}')
ax[1].axhline(STAT, color='gray', ls=':', lw=1.2)
ax[1].set_xlim(0, 300); ax[1].set_ylim(-0.25, 1.1)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('x')
ax[1].set_title('Position, varying β₁')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------
# 5. THE INTERNAL QUANTITIES m AND v
# ---------------------------------------------------------------------

print("=" * 82)
print("5. ADAM INTERNALS      lr = 0.05, b1 = 0.9, b2 = 0.999, x0 = 1.0")
print("=" * 82)
ha, ma, va, na = A[(0.05, 0.9, 0.999)]
gg = 2*ha[:-1, 0]
print(f"{'k':>6} {'x':>11} {'g = 2x':>11} {'m_hat':>11} {'v_hat':>12} "
      f"{'sqrt(v_hat)':>12} {'step/lr':>9}")
print("-" * 82)
for k in (0, 1, 2, 5, 10, 30, 60, 100, 200, min(275, na - 1)):
    if k < len(ma):
        sv = float(np.sqrt(va[k, 0]))
        print(f"{k:6d} {ha[k,0]:11.6f} {gg[k]:11.6f} {ma[k,0]:11.6f} "
              f"{va[k,0]:12.3e} {sv:12.3e} {ma[k,0]/(sv+1e-8):9.4f}")
print()
print("m_hat has a memory of roughly 1/(1-b1) = 10 iterations, v_hat of roughly")
print("1/(1-b2) = 1000. The last column, m_hat/sqrt(v_hat), is the step in units of lr,")
print("and the two memories give it two distinct regimes.")
print()
print("Early on m_hat is close to g and sqrt(v_hat) is close to |g|, so the ratio is 1")
print("and the step is exactly lr. This regime dominates any run that starts far out:")
print("from x0 = 10 the position falls by 0.0500 per iteration, exactly lr, for several")
print("hundred iterations, which is the linear growth seen in section 1.")
print()
print("Near the stationary point the two averages separate. m_hat follows the shrinking")
print("gradient closely, while sqrt(v_hat) still holds the magnitude the gradient had")
print("hundreds of iterations earlier: at k = 275 the gradient is 1e-6 but sqrt(v_hat)")
print("is still 0.31. The denominator is stale, so the step is far below lr and shrinks")
print("only as fast as m_hat does. That is the quantity the stopping test waits on, and")
print("it is why adam needs 276 iterations here against 87 for plain descent.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
kk = np.arange(len(ma))
ax[0].semilogy(kk, np.abs(gg), lw=1.4, color='k', label='|g|')
ax[0].semilogy(kk, np.abs(ma[:, 0]), lw=1.4, color=C['momentum'], label='|m̂|')
ax[0].semilogy(kk, np.sqrt(va[:, 0]), lw=1.4, color=C['adam'], label='√v̂')
ax[0].set_xlim(0, 300); ax[0].set_ylim(1e-8, 1e1)
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('magnitude')
ax[0].set_title('Adam internal quantities against iteration')
ax[0].legend(fontsize=8)

ax[1].plot(kk, ma[:, 0]/(np.sqrt(va[:, 0]) + 1e-8), lw=1.4, color='k')
ax[1].axhline(0, color='gray', lw=.8)
ax[1].axhline(1, color='gray', ls=':', lw=1)
ax[1].axhline(-1, color='gray', ls=':', lw=1)
ax[1].set_xlim(0, 300); ax[1].set_ylim(-1.4, 1.4)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('m̂ / √v̂')
ax[1].set_title('Step in units of lr')
plt.tight_layout(); plt.show()
