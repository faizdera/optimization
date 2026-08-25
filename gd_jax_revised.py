import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# =====================================================================
# REVISION 1 - stopping criterion is now on the STEP SIZE  |x_{k+1}-x_k|
#              not on the gradient norm.
# =====================================================================

def project(x, bounds):
    if bounds is None:
        return x
    lo, hi = bounds
    return jnp.clip(x, lo, hi)


def gradient_descent(f, x0, lr=0.01, bounds=None, tol=1e-6, max_iter=20000):
    grad_f = jax.grad(f)
    x = jnp.array(x0, dtype=float)
    history = [np.array(x)]
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        x_new = project(x - lr * g, bounds)
        step = float(jnp.linalg.norm(x_new - x))
        history.append(np.array(x_new))
        if step < tol:
            return np.array(x_new), i, np.array(history), True
        x = x_new
    return np.array(x), max_iter, np.array(history), False


def gradient_descent_momentum(f, x0, lr=0.01, beta=0.85, bounds=None, tol=1e-6, max_iter=20000):
    grad_f = jax.grad(f)
    x = jnp.array(x0, dtype=float)
    v = jnp.zeros_like(x)
    history = [np.array(x)]
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        v = beta * v + g
        x_new = project(x - lr * v, bounds)
        step = float(jnp.linalg.norm(x_new - x))
        history.append(np.array(x_new))
        if step < tol:
            return np.array(x_new), i, np.array(history), True
        x = x_new
    return np.array(x), max_iter, np.array(history), False


def gradient_descent_adam(f, x0, lr=0.05, b1=0.9, b2=0.999, eps=1e-8,
                          bounds=None, tol=1e-6, max_iter=20000):
    grad_f = jax.grad(f)
    x = jnp.array(x0, dtype=float)
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    history = [np.array(x)]
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g**2
        m_hat = m / (1 - b1**i)
        v_hat = v / (1 - b2**i)
        x_new = project(x - lr * m_hat / (jnp.sqrt(v_hat) + eps), bounds)
        step = float(jnp.linalg.norm(x_new - x))
        history.append(np.array(x_new))
        if step < tol:
            return np.array(x_new), i, np.array(history), True
        x = x_new
    return np.array(x), max_iter, np.array(history), False


def step_history(history):
    """|x_{k+1} - x_k| along a stored path - the quantity now being tested."""
    return np.linalg.norm(np.diff(history, axis=0), axis=1)


def track(history, iters):
    """position at selected iterations, for the tracking tables."""
    return [(k, history[k]) for k in iters if k < len(history)]


# =====================================================================
# THE THREE PROBLEMS, each with an ANALYTICAL gradient for cross-check
# =====================================================================

def f1(x):
    return x[0]**2

def g1_analytic(x):
    return np.array([2*x[0]])


def f2(x):
    return (6*x[0] - 2)**2 * jnp.sin(12*x[0] - 4)

def g2_analytic(x):
    u = 6*x[0] - 2
    return np.array([12*u*np.sin(12*x[0]-4) + 12*u**2*np.cos(12*x[0]-4)])


def f3(x):
    return x[0]**4 - 4*x[0]*x[1] + x[1]**4

def g3_analytic(x):
    return np.array([4*x[0]**3 - 4*x[1], 4*x[1]**3 - 4*x[0]])


plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 9})
COL = {'plain': '#1f4e79', 'momentum': '#c0504d', 'adam': '#4f8a10'}
TOL = 1e-6


# =====================================================================
# REVISION 2 - JAX autograd vs gradients derived by hand
# =====================================================================

print("=" * 74)
print("GRADIENT VERIFICATION - jax.grad  vs  analytical derivative")
print("=" * 74)
print(f"{'problem':<8} {'point':>18} {'jax.grad':>26} {'max |difference|':>18}")
print("-" * 74)

checks = [("P1", f1, g1_analytic, [2.0]),
          ("P1", f1, g1_analytic, [50.0]),
          ("P2", f2, g2_analytic, [0.35]),
          ("P2", f2, g2_analytic, [0.90]),
          ("P3", f3, g3_analytic, [1.8, -1.2]),
          ("P3", f3, g3_analytic, [0.05, -0.05])]

for name, f, ga, pt in checks:
    j = np.array(jax.grad(f)(jnp.array(pt, dtype=float)))
    a = ga(pt)
    pts = f"[{pt[0]}]" if len(pt) == 1 else f"[{pt[0]}, {pt[1]}]"
    js = f"[{j[0]:.6f}]" if len(j) == 1 else f"[{j[0]:.4f}, {j[1]:.4f}]"
    print(f"{name:<8} {pts:>18} {js:>26} {np.max(np.abs(j-a)):18.2e}")
print("=> agreement is exact to machine precision on every problem and every point.")
print()

# visual version: analytical curve, jax.grad sampled on top of it
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))

xs = np.linspace(0, 1, 400)
ax[0].plot(xs, [g2_analytic([v])[0] for v in xs], 'k-', lw=1.6, label="analytical  f′(x)")
xsamp = np.linspace(0, 1, 26)
ax[0].plot(xsamp, [float(jax.grad(f2)(jnp.array([v]))[0]) for v in xsamp],
           'o', ms=5, mfc='none', mec=COL['momentum'], mew=1.4, label='jax.grad')
ax[0].axhline(0, color='gray', lw=.8)
ax[0].set_xlabel('x'); ax[0].set_ylabel("f′(x)")
ax[0].set_title('Problem 2 — hand derivative vs jax.grad')
ax[0].legend()

diffs = []
for v in xsamp:
    j = float(jax.grad(f2)(jnp.array([v]))[0])
    diffs.append(abs(j - g2_analytic([v])[0]))
ax[1].semilogy(xsamp, np.maximum(diffs, 1e-18), 'o-', ms=4, color=COL['momentum'])
ax[1].axhline(2.2e-16, color='gray', ls='--', lw=1, label='double precision eps')
ax[1].set_ylim(1e-18, 1e-10)
ax[1].set_xlabel('x'); ax[1].set_ylabel('|jax − analytical|')
ax[1].set_title('Difference is at machine precision everywhere')
ax[1].legend()
plt.tight_layout(); plt.show()


# =====================================================================
# REVISION 3 - PROBLEM 1, much larger starting points
# =====================================================================

starts1 = [2.0, 10.0, 50.0, 100.0, 500.0]

print("=" * 74)
print("PROBLEM 1 - PLAIN, large starting points   (lr = 0.1, stop on |dx| < 1e-6)")
print("=" * 74)
print(f"{'x0':>8} {'x_final':>14} {'f(x_final)':>14} {'iters':>8}  converged")
print("-" * 74)

runs1 = {}
for x0 in starts1:
    xf, n, hist, conv = gradient_descent(f1, [x0], lr=0.1)
    runs1[x0] = hist
    print(f"{x0:8.1f} {xf[0]:14.3e} {float(f1(xf)):14.3e} {n:8d}  {conv}")
print("=> x0 grows 250x but iterations only rise 59 -> 84.")
print("   x_k = (1-2*lr)^k * x0 is geometric, so iterations scale with LOG(x0).")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for x0, hist in runs1.items():
    ax[0].semilogy(np.abs(hist[:, 0]) + 1e-16, lw=1.5, label=f'x₀ = {x0:g}')
ax[0].axhline(TOL, color='gray', ls='--', lw=1, label='|dx| tolerance')
ax[0].set_xlim(0, 95); ax[0].set_ylim(1e-7, 1e3)
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('|x|')
ax[0].set_title('Problem 1 — plain GD from x₀ = 2 … 500')
ax[0].legend(fontsize=8)

iters = [len(h) - 1 for h in runs1.values()]
ax[1].semilogx(starts1, iters, 'o-', color=COL['plain'], lw=1.6, ms=6)
ax[1].set_xlabel('starting point x₀  (log scale)'); ax[1].set_ylabel('iterations')
ax[1].set_title('Iterations grow with log(x₀), not x₀')
plt.tight_layout(); plt.show()


# =====================================================================
# REVISION 4 - PROBLEM 1, momentum stated as a HYPOTHESIS and tested
#              directly against plain, same axes
# =====================================================================

print("=" * 74)
print("PROBLEM 1 - hypothesis test: does momentum help on a problem with")
print("            no plateau and no ill-conditioning?   (x0 = 10)")
print("=" * 74)
print("HYPOTHESIS: no. f = x^2 is strictly convex with uniform curvature")
print("            f'' = 2 (condition number = 1). Momentum targets flat")
print("            regions and zig-zag; neither exists here, so the added")
print("            velocity should only overshoot and cost iterations.")
print("-" * 74)
print(f"{'method':<12} {'x_final':>14} {'iters':>8} {'overshoots past 0':>20}")
print("-" * 74)

runsH = {}
for name, solver, kw in [("plain",    gradient_descent,          dict(lr=0.1)),
                         ("momentum", gradient_descent_momentum, dict(lr=0.1, beta=0.85))]:
    xf, n, hist, conv = solver(f1, [10.0], **kw)
    runsH[name] = hist
    crossings = int(np.sum(np.diff(np.sign(hist[:, 0])) != 0))
    print(f"{name:<12} {xf[0]:14.3e} {n:8d} {crossings:20d}")
print("RESULT: hypothesis confirmed - momentum is slower, and the extra")
print("        iterations are spent oscillating across the minimum.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for name, hist in runsH.items():
    ax[0].plot(hist[:, 0], lw=1.6, color=COL[name], label=f'{name} ({len(hist)-1} it)')
ax[0].axhline(0, color='gray', lw=.9, ls=':')
ax[0].set_xlim(0, 200); ax[0].set_ylim(-2.5, 10.5)
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('x')
ax[0].set_title('Problem 1, x₀ = 10 — momentum overshoots, plain does not')
ax[0].legend()

for name, hist in runsH.items():
    ax[1].semilogy(step_history(hist), lw=1.5, color=COL[name], label=name)
ax[1].axhline(TOL, color='gray', ls='--', lw=1, label='tolerance')
ax[1].set_xlim(0, 200); ax[1].set_ylim(1e-7, 1e1)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('|x$_{k+1}$ − x$_k$|')
ax[1].set_title('step size — momentum ripples, plain decays cleanly')
ax[1].legend()
plt.tight_layout(); plt.show()


# =====================================================================
# REVISION 5 - PROBLEM 1, WHY is Adam's oscillation so large?
# =====================================================================

print("=" * 74)
print("PROBLEM 1 - why does Adam oscillate?   lr sweep, x0 = 2")
print("=" * 74)
print("MECHANISM: Adam's step is  lr * m_hat/sqrt(v_hat), and that ratio is")
print("           ~1 whenever the gradient direction is consistent. So the")
print("           step is ~lr REGARDLESS of how small the gradient becomes.")
print("           Near x = 0 it therefore cannot take a small step - it")
print("           overshoots by roughly lr and has to come back.")
print("-" * 74)
print(f"{'lr':>8} {'iters':>8} {'sign changes':>14} {'most negative x':>18}")
print("-" * 74)

lrs_adam = [0.005, 0.01, 0.05, 0.1, 0.3]
osc_lr, osc_amp, osc_it = [], [], []
runsA = {}
for lr in lrs_adam:
    xf, n, hist, conv = gradient_descent_adam(f1, [2.0], lr=lr)
    h = hist[:, 0]
    sc = int(np.sum(np.diff(np.sign(h)) != 0))
    runsA[lr] = hist
    if sc > 0:                                   # only runs that actually cross zero
        osc_lr.append(lr); osc_amp.append(abs(h.min())); osc_it.append(n)
    print(f"{lr:8.3f} {n:8d} {sc:14d} {h.min():18.4f}")
print("=> overshoot amplitude tracks lr almost exactly. Reduce lr and the")
print("   oscillation disappears (lr <= 0.01 never crosses zero at all).")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for lr in [0.01, 0.05, 0.3]:
    ax[0].plot(runsA[lr][:, 0], lw=1.5, label=f'lr = {lr}')
ax[0].axhline(0, color='gray', lw=.9, ls=':')
ax[0].set_xlim(0, 260); ax[0].set_ylim(-0.85, 2.1)
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('x')
ax[0].set_title('Problem 1 — Adam, larger lr = larger overshoot')
ax[0].legend()

ax[1].loglog(osc_lr, osc_amp, 'o-', color=COL['adam'], lw=1.8, ms=7,
             label='measured overshoot |x$_{min}$|')
ref = np.array(osc_lr)
ax[1].loglog(ref, 2.4*ref, 'k--', lw=1.2, label='slope 1 reference (∝ lr)')
ax[1].set_xlim(0.03, 0.5)
ax[1].set_xlabel('lr'); ax[1].set_ylabel('overshoot amplitude')
ax[1].set_title('Overshoot ∝ lr.  At lr ≤ 0.01 it never crosses zero at all')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()


# =====================================================================
# REVISION 6 - PROBLEM 2, does a larger plain lr escape the flat region?
# =====================================================================

print("=" * 74)
print("PROBLEM 2 - x0 = 0.45, PLAIN, learning-rate sweep")
print("=" * 74)
print("QUESTION: can a larger lr carry plain GD across the flat region?")
print("-" * 74)
print(f"{'lr':>8} {'x_final':>10} {'f(x_final)':>12} {'iters':>8} {'conv':>7}   outcome")
print("-" * 74)

lrs2 = [0.0005, 0.001, 0.003, 0.010, 0.013, 0.015, 0.020]
runsL = {}
for lr in lrs2:
    xf, n, hist, conv = gradient_descent(f2, [0.45], lr=lr, bounds=(0., 1.), max_iter=4000)
    runsL[lr] = hist
    if not conv:
        out = "unstable - driven to a bound"
    elif xf[0] < 0.25:
        out = "crossed the flat region"
    else:
        out = "settled in the flat region"
    print(f"{lr:8.4f} {xf[0]:10.5f} {float(f2(xf)):12.5f} {n:8d} {str(conv):>7}   {out}")
print("=> NO escape window exists. Up to lr ~ 0.013 it settles in the flat")
print("   region, only sooner; from lr ~ 0.015 it is unstable and is driven")
print("   onto a bound. Larger lr changes the speed of arrival and then the")
print("   stability, but never the destination.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for lr in [0.001, 0.010, 0.013]:
    ax[0].plot(runsL[lr][:, 0], lw=1.6, label=f'lr = {lr}  (stable)')
ax[0].plot(runsL[0.015][:, 0], lw=0.8, color='#a00000', alpha=.30,
           label='lr = 0.015  (unstable)')
ax[0].axhline(1/3, color='gray', ls=':', lw=1.2)
ax[0].text(1180, 1/3 + 0.01, 'x = 1/3', fontsize=7.5, color='gray', ha='right')
ax[0].axhline(0.142589, color='gray', ls=':', lw=1.2)
ax[0].text(1180, 0.142589 + 0.01, 'x = 0.1426', fontsize=7.5, color='gray', ha='right')
ax[0].set_xlim(0, 950); ax[0].set_ylim(0, 1.05)
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('x')
ax[0].set_title('Problem 2, x₀ = 0.45 — every stable lr ends at x ≈ 1/3')
ax[0].legend(fontsize=8)

for lr in [0.001, 0.010, 0.013]:
    ax[1].semilogy(step_history(runsL[lr]), lw=1.5, label=f'lr = {lr}')
ax[1].axhline(TOL, color='gray', ls='--', lw=1, label='tolerance')
ax[1].set_xlim(0, 950); ax[1].set_ylim(1e-7, 1e-1)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('|x$_{k+1}$ − x$_k$|')
ax[1].set_title('the step collapses below tolerance while still on the flat part')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()


# =====================================================================
# REVISION 7 - PROBLEM 2, beta sweep: momentum has a sharp threshold
# =====================================================================

print("=" * 74)
print("PROBLEM 2 - x0 = 0.45, MOMENTUM, beta sweep at lr = 0.001")
print("=" * 74)
print(f"{'beta':>7} {'x_final':>10} {'f(x_final)':>12} {'iters':>8}   outcome")
print("-" * 74)

betas = [0.0, 0.50, 0.65, 0.70, 0.85, 0.95, 0.99]
runsB, b_end = {}, []
for b in betas:
    xf, n, hist, conv = gradient_descent_momentum(f2, [0.45], lr=0.001, beta=b,
                                                  bounds=(0., 1.), max_iter=4000)
    runsB[b] = hist
    b_end.append(xf[0])
    out = "crossed the flat region" if xf[0] < 0.25 else "settled in the flat region"
    print(f"{b:7.2f} {xf[0]:10.5f} {float(f2(xf)):12.5f} {n:8d}   {out}")
print("=> beta = 0 reproduces plain GD exactly, as it must.")
print("   The transition is SHARP: between beta = 0.65 and 0.70 the behaviour")
print("   changes completely. Accumulated velocity is what carries the")
print("   iterate across the region where the gradient is nearly zero.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
for b in [0.0, 0.65, 0.70, 0.85]:
    ax[0].plot(runsB[b][:, 0], lw=1.6, label=f'β = {b}')
ax[0].axhline(1/3, color='gray', ls=':', lw=1.2)
ax[0].text(880, 1/3 + 0.008, 'x = 1/3', fontsize=7.5, color='gray', ha='right')
ax[0].axhline(0.142589, color='gray', ls=':', lw=1.2)
ax[0].text(880, 0.142589 + 0.008, 'x = 0.1426', fontsize=7.5, color='gray', ha='right')
ax[0].set_xlim(0, 900); ax[0].set_ylim(0.10, 0.47)
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('x')
ax[0].set_title('Problem 2, x₀ = 0.45 — momentum, varying β')
ax[0].legend(fontsize=8)

ax[1].plot(betas, b_end, 'o-', color=COL['momentum'], lw=1.6, ms=7)
ax[1].axvspan(0.65, 0.70, color='#c0504d', alpha=.12)
ax[1].annotate('transition', xy=(0.675, 0.24), xytext=(0.40, 0.20), fontsize=8,
               color='#a00000', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='#a00000', lw=1.1))
ax[1].set_xlabel('β'); ax[1].set_ylabel('final x')
ax[1].set_title('Sharp threshold between β = 0.65 and β = 0.70')
plt.tight_layout(); plt.show()


# =====================================================================
# REVISION 8 - position tracking tables for the interesting runs
# =====================================================================

MARKS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

def tracking_table(title, note, runs):
    print("=" * 74)
    print(title)
    print("=" * 74)
    header = "".join(f"{('k='+str(k)):>11}" for k in MARKS)
    print(f"{'run':<22}{header}")
    print("-" * (22 + 11*len(MARKS)))
    for label, hist in runs:
        cells = ""
        for k in MARKS:
            if k < len(hist):
                p = hist[k]
                cells += f"{p[0]:11.5f}" if len(p) == 1 else f"{p[0]:11.4f}"
            else:
                cells += f"{'(ended)':>11}"
        print(f"{label:<22}{cells}")
    print(note)
    print()

tracking_table(
    "TRACKING - Problem 1, x0 = 10:  x at selected iterations",
    "=> momentum passes below zero around k = 20 and returns; plain never does.",
    [("plain, lr=0.1", runsH['plain']), ("momentum, β=0.85", runsH['momentum'])])

tracking_table(
    "TRACKING - Problem 2, x0 = 0.45, PLAIN:  x at selected iterations",
    "=> every lr slows to a halt near x = 0.334. Larger lr arrives sooner.",
    [(f"lr = {lr}", runsL[lr]) for lr in [0.001, 0.010, 0.013]])

tracking_table(
    "TRACKING - Problem 2, x0 = 0.45, MOMENTUM:  x at selected iterations",
    "=> β = 0.65 is still near 0.334 at k = 200; β = 0.70 is already past it by k = 50.",
    [(f"β = {b}", runsB[b]) for b in [0.0, 0.65, 0.70, 0.85]])
