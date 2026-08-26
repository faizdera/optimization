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

def gradient_descent(f, x0, lr=0.001, tol=1e-8, max_iter=40000, grad_f=None):
    if grad_f is None:
        grad_f = jax.grad(f)
    x = jnp.array(x0, dtype=float)
    hist = [np.array(x)]
    for i in range(1, max_iter + 1):
        x_new = x - lr * grad_f(x)
        if not np.all(np.isfinite(np.array(x_new))):
            return np.array(x), i, np.array(hist), False
        hist.append(np.array(x_new))
        if jnp.linalg.norm(x_new - x) < tol:
            return np.array(x_new), i, np.array(hist), True
        x = x_new
    return np.array(x), max_iter, np.array(hist), False


def gradient_descent_momentum(f, x0, lr=0.001, beta=0.85, ema=False,
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
        if not np.all(np.isfinite(np.array(x_new))):
            return np.array(x), i, np.array(hist), False
        hist.append(np.array(x_new))
        if jnp.linalg.norm(x_new - x) < tol:
            return np.array(x_new), i, np.array(hist), True
        x = x_new
    return np.array(x), max_iter, np.array(hist), False


def gradient_descent_adam(f, x0, lr=0.05, b1=0.9, b2=0.999, eps=1e-8,
                          tol=1e-8, max_iter=40000, grad_f=None):
    if grad_f is None:
        grad_f = jax.grad(f)
    x = jnp.array(x0, dtype=float)
    m = jnp.zeros_like(x); v = jnp.zeros_like(x)
    hist = [np.array(x)]
    m_rec, v_rec = [], []
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g**2
        m_hat = m / (1 - b1**i); v_hat = v / (1 - b2**i)
        m_rec.append(np.array(m_hat)); v_rec.append(np.array(v_hat))
        x_new = x - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        if not np.all(np.isfinite(np.array(x_new))):
            return np.array(x), i, np.array(hist), False, np.array(m_rec), np.array(v_rec)
        hist.append(np.array(x_new))
        if jnp.linalg.norm(x_new - x) < tol:
            return (np.array(x_new), i, np.array(hist), True,
                    np.array(m_rec), np.array(v_rec))
        x = x_new
    return np.array(x), max_iter, np.array(hist), False, np.array(m_rec), np.array(v_rec)


def steps(h):
    d = np.diff(h, axis=0)
    return np.linalg.norm(d.reshape(len(d), -1), axis=1)


# =====================================================================
# PROBLEM 2
# =====================================================================

def f2(x):
    return (6*x[0] - 2)**2 * jnp.sin(12*x[0] - 4)

def f2_grad(x):
    u = 6*x[0] - 2
    return jnp.array([12*u*(jnp.sin(12*x[0]-4) + u*jnp.cos(12*x[0]-4))])

def gval(v):
    return float(f2(jnp.array([v])))

def gnorm(v):
    return float(abs(f2_grad(jnp.array([v]))[0]))


plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 9})
C = {'plain': '#1f4e79', 'momentum': '#c0504d', 'adam': '#4f8a10'}
TOL = 1e-8

# stationary points inside [0,1], from f'(x) = 0
STAT = [0.142589, 1/3, 0.524077, 0.757249]

xs = np.linspace(0, 1, 600)
fx = np.array([gval(v) for v in xs])


def draw_f(ax, xlim=(0, 1), ylim=None, mark=True):
    lo, hi = xlim
    g = np.linspace(lo, hi, 600)
    ax.plot(g, [gval(v) for v in g], 'k-', lw=1.2, label='f(x)')
    if mark:
        for s in STAT:
            if lo <= s <= hi:
                ax.axvline(s, color='gray', ls=':', lw=1)
    ax.set_xlim(lo, hi)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel('x'); ax.set_ylabel('f(x)')


print("=" * 80)
print("PROBLEM 2     f(x) = (6x - 2)^2 sin(12x - 4)     unconstrained")
print("=" * 80)
print("The problem is solved without any bound. Results are reported over the range")
print("x in [0, 1]; a run that leaves that range is recorded as such.")
print()
print("f'(x) = 12(6x - 2)[ sin(12x - 4) + (6x - 2) cos(12x - 4) ]")
print("Setting f' = 0 gives 6x - 2 = 0, that is x = 1/3, plus the roots of the")
print("bracket, which is transcendental and has to be solved numerically. Inside")
print("[0, 1] the stationary points are:")
for s in STAT:
    print(f"   x = {s:.6f}    f = {gval(s):10.6f}")
print()
print("Gradient check, analytic against jax.grad:")
for p in (0.35, 0.90, 0.05):
    a = float(f2_grad(jnp.array([p]))[0]); j = float(jax.grad(f2)(jnp.array([p]))[0])
    print(f"   x = {p:5.2f}   analytic {a:12.6f}   jax {j:12.6f}   difference {abs(a-j):.1e}")
print()


# ---------------------------------------------------------------------
# 1. PLAIN GRADIENT DESCENT FROM SEVERAL STARTING POINTS
# ---------------------------------------------------------------------

starts = [0.05, 0.20, 0.30, 0.45, 0.60, 0.90]

print("=" * 80)
print("1. PLAIN GRADIENT DESCENT,  lr = 0.001")
print("=" * 80)
print(f"{'x0':>6} {'x_final':>11} {'f(x_final)':>12} {'|f'+chr(39)+'| there':>12} {'iters':>8} {'converged':>11}")
print("-" * 80)
base = {}
for x0 in starts:
    xm, n, h, c = gradient_descent(f2, [x0], lr=0.001, grad_f=f2_grad)
    base[x0] = h
    print(f"{x0:6.2f} {xm[0]:11.6f} {gval(xm[0]):12.6f} {gnorm(xm[0]):12.3e} {n:8d} {str(c):>11}")
print()
print("Three different stationary points are reached from six starting points. The run")
print("from x0 = 0.45 stops at x = 0.3334 where |f'| is still around 1e-5, four orders")
print("larger than at the other endpoints. The step there is small because the gradient")
print("is small, not because the iterate has arrived.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
draw_f(ax[0], (0, 1))
for x0 in starts:
    h = base[x0][:, 0]
    ax[0].plot(h, [gval(v) for v in h], 'o-', ms=3.5, lw=1.2,
               markevery=max(1, len(h)//20), label=f'x₀ = {x0}')
ax[0].set_title('Plain gradient descent, lr = 0.001')
ax[0].legend(fontsize=7, loc='upper left', ncol=2)

for x0 in starts:
    ax[1].plot(base[x0][:, 0], lw=1.4, label=f'x₀ = {x0}')
for s in STAT:
    ax[1].axhline(s, color='gray', ls=':', lw=1)
ax[1].set_xscale('log'); ax[1].set_xlim(1, 2e4); ax[1].set_ylim(0, 1)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('x')
ax[1].set_title('Position against iteration')
ax[1].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.show()

cps = (0, 1, 2, 5, 10, 25, 50, 100, 500, 2000)
print("=" * 100)
print("POSITION AT SELECTED ITERATIONS,  plain gradient descent, lr = 0.001")
print("=" * 100)
print(f"{'x0':<7}" + "".join(f"{('k='+str(c)):>9}" for c in cps) + f"{'final k':>10}")
print("-" * 100)
for x0 in starts:
    h = base[x0]
    cells = "".join((f"{h[k,0]:9.5f}" if k < len(h) else f"{'-':>9}") for k in cps)
    print(f"{x0:<7.2f}{cells}{len(h)-1:>10}")
print()


# ---------------------------------------------------------------------
# 2. CAN A LARGER STEP CARRY THE ITERATE ACROSS x = 1/3 ?
# ---------------------------------------------------------------------

print("=" * 80)
print("2. PLAIN GRADIENT DESCENT FROM x0 = 0.45, VARYING lr")
print("=" * 80)
print(f"{'lr':>9} {'x_final':>11} {'f(x_final)':>12} {'|f'+chr(39)+'| there':>12} {'iters':>8} {'converged':>11}")
print("-" * 80)
lr_scan = [0.0005, 0.001, 0.002, 0.005, 0.010, 0.014, 0.016, 0.018, 0.020, 0.050]
esc = {}
for lr in lr_scan:
    xm, n, h, c = gradient_descent(f2, [0.45], lr=lr, grad_f=f2_grad)
    esc[lr] = h
    if c:
        print(f"{lr:9.4f} {xm[0]:11.6f} {gval(xm[0]):12.6f} {gnorm(xm[0]):12.3e} {n:8d} {str(c):>11}")
    else:
        print(f"{lr:9.4f} {'left [0,1]':>11} {'-':>12} {'-':>12} {n:8d} {str(c):>11}")
print()
print("No step size carries the iterate across x = 1/3. Every stable lr from 0.0005 to")
print("0.016 stops in the same place. Above that the update exceeds the stability limit")
print("set by the curvature elsewhere in the domain and the iterate leaves [0, 1)")
print("entirely, where f grows without bound. The gradient near x = 1/3 is genuinely")
print("small, so a step large enough to cross it is also large enough to destabilise")
print("the rest of the function.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
draw_f(ax[0], (0.2, 0.6), ylim=(-1.2, 1.2))
for lr in (0.001, 0.005, 0.010, 0.016):
    h = esc[lr][:, 0]
    ax[0].plot(h, [gval(v) for v in h], 'o-', ms=3.5, lw=1.2,
               markevery=max(1, len(h)//20), label=f'lr = {lr}')
ax[0].plot(0.45, gval(0.45), 'ks', ms=7, label='x₀')
ax[0].set_title('Path from x₀ = 0.45, four step sizes')
ax[0].legend(fontsize=8, loc='lower right')

for lr in (0.001, 0.005, 0.010, 0.016):
    ax[1].plot(esc[lr][:, 0], lw=1.4, label=f'lr = {lr}')
ax[1].axhline(1/3, color='gray', ls=':', lw=1)
ax[1].set_xscale('log'); ax[1].set_xlim(1, 2e4); ax[1].set_ylim(0.30, 0.47)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('x')
ax[1].set_title('All step sizes stop in the same place')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------
# 3. WHERE EACH STARTING POINT ENDS UP, AS lr CHANGES
# ---------------------------------------------------------------------

lrs = [0.001, 0.005, 0.010, 0.014]
print("=" * 88)
print("3. FINAL x AS A FUNCTION OF x0 AND lr,  plain gradient descent")
print("=" * 88)
print(f"{'x0':>6}" + "".join(f"{('lr='+format(l,'g')):>19}" for l in lrs))
print("-" * 88)
for x0 in starts:
    row = ""
    for lr in lrs:
        xm, n, h, c = gradient_descent(f2, [x0], lr=lr, grad_f=f2_grad)
        row += f"{xm[0]:12.5f}({n:5d})" if c else f"{'left [0,1]':>19}"
    print(f"{x0:6.2f}" + row)
print()
print("The step size changes the endpoint, not only the speed. At lr = 0.001 two runs")
print("reach x = 0.7572. At lr = 0.005 none do: the runs from x0 = 0.60 and 0.90 pass")
print("over that region and settle at x = 0.1426 instead. At lr = 0.010 and above most")
print("runs finish near x = 1/3 or leave the range altogether. A large step launched")
print("from anywhere lands near the middle of the domain, which is where the gradient")
print("is smallest, and progress then becomes very slow.")
print()


# ---------------------------------------------------------------------
# 4. MOMENTUM
# ---------------------------------------------------------------------

print("=" * 88)
print("4. MOMENTUM FROM x0 = 0.45")
print("=" * 88)
print(f"{'lr':>8} {'beta':>6} {'form':>11} {'x_final':>11} {'|f'+chr(39)+'| there':>12} {'iters':>8} {'conv':>7}")
print("-" * 88)
mom = {}
for lr, b, ema in [(0.001, 0.30, False), (0.001, 0.50, False), (0.001, 0.70, False),
                   (0.001, 0.85, False), (0.001, 0.95, False),
                   (0.005, 0.85, False), (0.010, 0.85, False),
                   (0.001, 0.85, True), (0.005, 0.85, True), (0.010, 0.95, True)]:
    xm, n, h, c = gradient_descent_momentum(f2, [0.45], lr=lr, beta=b, ema=ema,
                                            grad_f=f2_grad)
    mom[(lr, b, ema)] = h
    fm = 'averaged' if ema else 'classical'
    if c:
        print(f"{lr:8.4f} {b:6.2f} {fm:>11} {xm[0]:11.6f} {gnorm(xm[0]):12.3e} {n:8d} {str(c):>7}")
    else:
        print(f"{lr:8.4f} {b:6.2f} {fm:>11} {'left [0,1]':>11} {'-':>12} {n:8d} {str(c):>7}")
print()
print("Momentum crosses x = 1/3 once beta is large enough. beta = 0.30 and 0.50 stop in")
print("the same place plain descent does; from beta = 0.70 upwards the accumulated")
print("velocity carries the iterate through the flat region to x = 0.1426, where |f'|")
print("is genuinely small. The averaged form needs a correspondingly larger lr to do")
print("the same thing, since the (1-beta) factor scales the step down.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
draw_f(ax[0], (0.05, 0.55), ylim=(-1.3, 1.0))
for key, lab in [((0.001, 0.30, False), 'β = 0.30'), ((0.001, 0.50, False), 'β = 0.50'),
                 ((0.001, 0.85, False), 'β = 0.85'), ((0.001, 0.95, False), 'β = 0.95')]:
    h = mom[key][:, 0]
    ax[0].plot(h, [gval(v) for v in h], 'o-', ms=3.5, lw=1.2,
               markevery=max(1, len(h)//20), label=lab)
ax[0].plot(0.45, gval(0.45), 'ks', ms=7, label='x₀')
ax[0].set_title('Momentum from x₀ = 0.45, classical form, lr = 0.001')
ax[0].legend(fontsize=8, loc='lower right')

for key, lab in [((0.001, 0.30, False), 'β = 0.30'), ((0.001, 0.50, False), 'β = 0.50'),
                 ((0.001, 0.70, False), 'β = 0.70'), ((0.001, 0.85, False), 'β = 0.85'),
                 ((0.001, 0.95, False), 'β = 0.95')]:
    ax[1].plot(mom[key][:, 0], lw=1.4, label=lab)
for s in STAT[:2]:
    ax[1].axhline(s, color='gray', ls=':', lw=1)
ax[1].set_xscale('log'); ax[1].set_xlim(1, 2e4); ax[1].set_ylim(0.05, 0.50)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('x')
ax[1].set_title('Position against iteration')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------
# 5. ADAM
# ---------------------------------------------------------------------

print("=" * 88)
print("5. ADAM FROM x0 = 0.45")
print("=" * 88)
print(f"{'lr':>8} {'b1':>6} {'b2':>8} {'x_final':>11} {'|f'+chr(39)+'| there':>12} {'iters':>8} {'conv':>7}")
print("-" * 88)
ad = {}
for lr, b1, b2 in [(0.05, 0.9, 0.999), (0.02, 0.9, 0.999), (0.01, 0.9, 0.999),
                   (0.10, 0.9, 0.999),
                   (0.05, 0.7, 0.999), (0.05, 0.5, 0.999), (0.05, 0.0, 0.999),
                   (0.05, 0.9, 0.99),  (0.05, 0.9, 0.90)]:
    xm, n, h, c, mm, vv = gradient_descent_adam(f2, [0.45], lr=lr, b1=b1, b2=b2,
                                                grad_f=f2_grad)
    ad[(lr, b1, b2)] = (h, mm, vv)
    print(f"{lr:8.3f} {b1:6.2f} {b2:8.3f} {xm[0]:11.6f} {gnorm(xm[0]):12.3e} {n:8d} {str(c):>7}")
print()
print("Adam crosses the flat region for every setting except b1 = 0, which stops at")
print("x = 0.3334 like plain descent. That single row identifies the mechanism.")
print()
print("With b1 = 0 the numerator is m_hat = g, so the step is lr*g/sqrt(v_hat). Near")
print("x = 1/3 the gradient is small while v_hat still carries the magnitude it had")
print("many iterations earlier, so the ratio is small and the iterate stalls exactly")
print("as plain descent does. The division by sqrt(v_hat) on its own therefore does")
print("not rescale a small gradient back up: v_hat lags too far behind to do that.")
print()
print("What carries the iterate across is the averaging in m_hat, which retains the")
print("larger gradients encountered before the flat region. This is the same mechanism")
print("as momentum in section 4, and it is consistent with the threshold seen there:")
print("beta below 0.7 does not cross, and b1 = 0 does not either.")
print()
print("The lower b1 values that do cross are also much faster, 54 and 72 iterations")
print("against 256 at b1 = 0.9, since less accumulated velocity means less overshoot")
print("once the iterate arrives.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
draw_f(ax[0], (0.05, 0.55), ylim=(-1.3, 1.0))
for key, lab in [((0.10, 0.9, 0.999), 'lr = 0.10'), ((0.05, 0.9, 0.999), 'lr = 0.05'),
                 ((0.02, 0.9, 0.999), 'lr = 0.02'), ((0.01, 0.9, 0.999), 'lr = 0.01')]:
    h = ad[key][0][:, 0]
    ax[0].plot(h, [gval(v) for v in h], 'o-', ms=3.5, lw=1.2,
               markevery=max(1, len(h)//20), label=lab)
ax[0].plot(0.45, gval(0.45), 'ks', ms=7, label='x₀')
ax[0].set_title('Adam from x₀ = 0.45, varying lr')
ax[0].legend(fontsize=8, loc='lower right')

for key, lab in [((0.05, 0.9, 0.999), 'β₁ = 0.9'), ((0.05, 0.7, 0.999), 'β₁ = 0.7'),
                 ((0.05, 0.5, 0.999), 'β₁ = 0.5'), ((0.05, 0.0, 0.999), 'β₁ = 0.0')]:
    ax[1].plot(ad[key][0][:, 0], lw=1.4, label=lab)
for s in STAT[:2]:
    ax[1].axhline(s, color='gray', ls=':', lw=1)
ax[1].set_xlim(0, 400); ax[1].set_ylim(0.05, 0.50)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('x')
ax[1].set_title('Position against iteration, varying β₁')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

h5, m5, v5 = ad[(0.05, 0.9, 0.999)]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
kk = np.arange(len(m5))
gg = np.array([float(f2_grad(jnp.array([v]))[0]) for v in h5[:-1, 0]])
ax[0].semilogy(kk, np.abs(gg), lw=1.4, color='k', label='|g|')
ax[0].semilogy(kk, np.abs(m5[:, 0]), lw=1.4, color=C['momentum'], label='|m̂|')
ax[0].semilogy(kk, np.sqrt(v5[:, 0]), lw=1.4, color=C['adam'], label='√v̂')
ax[0].set_xlim(0, 300); ax[0].set_ylim(1e-8, 1e2)
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('magnitude')
ax[0].set_title('Adam internal quantities, lr = 0.05')
ax[0].legend(fontsize=8)

ax[1].plot(kk, m5[:, 0]/(np.sqrt(v5[:, 0]) + 1e-8), lw=1.4, color='k')
ax[1].axhline(0, color='gray', lw=.8)
ax[1].axhline(1, color='gray', ls=':', lw=1); ax[1].axhline(-1, color='gray', ls=':', lw=1)
ax[1].set_xlim(0, 300); ax[1].set_ylim(-1.4, 1.4)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('m̂ / √v̂')
ax[1].set_title('Step in units of lr')
plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------
# 6. THE THREE METHODS SIDE BY SIDE
# ---------------------------------------------------------------------

print("=" * 88)
print("6. THE THREE METHODS FROM x0 = 0.45")
print("=" * 88)
print(f"{'method':<24} {'x_final':>11} {'f(x_final)':>12} {'|f'+chr(39)+'| there':>12} {'iters':>8}")
print("-" * 88)
sel = [('plain, lr = 0.001', base[0.45]),
       ('momentum, β = 0.85', mom[(0.001, 0.85, False)]),
       ('adam, lr = 0.05', ad[(0.05, 0.9, 0.999)][0])]
for lab, h in sel:
    xe = h[-1, 0]
    print(f"{lab:<24} {xe:11.6f} {gval(xe):12.6f} {gnorm(xe):12.3e} {len(h)-1:8d}")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
draw_f(ax[0], (0.05, 0.55), ylim=(-1.3, 1.0))
for (lab, h), k in zip(sel, ('plain', 'momentum', 'adam')):
    hh = h[:, 0]
    ax[0].plot(hh, [gval(v) for v in hh], 'o-', ms=3.5, lw=1.3, color=C[k],
               markevery=max(1, len(hh)//20), label=lab)
ax[0].plot(0.45, gval(0.45), 'ks', ms=7, label='x₀')
ax[0].set_title('Path from x₀ = 0.45')
ax[0].legend(fontsize=8, loc='lower right')

for (lab, h), k in zip(sel, ('plain', 'momentum', 'adam')):
    ax[1].loglog(steps(h), lw=1.5, color=C[k], label=lab)
ax[1].axhline(TOL, color='gray', ls='--', lw=1)
ax[1].set_xlim(1, 2e4); ax[1].set_ylim(1e-10, 1e0)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('|x$_{k+1}$ − x$_k$|')
ax[1].set_title('Step size against iteration')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
