import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)     # float64, same precision as numpy

# =====================================================================
# GENERALIZED SOLVERS
#   - x is ALWAYS a vector. A 1-variable problem uses n = 1, i.e. x0 = [2.0]
#   - gradient comes from jax.grad(f), never from finite difference
#   - bounds=None for unconstrained, bounds=(lo, hi) for a box constraint
# =====================================================================

def project(x, bounds):
    if bounds is None:
        return x
    lo, hi = bounds
    return jnp.clip(x, lo, hi)


def gradient_descent(f, x0, lr=0.01, bounds=None, tol=1e-6, max_iter=5000):
    grad_f = jax.grad(f)
    x = jnp.array(x0, dtype=float)
    history = [np.array(x)]
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        if jnp.linalg.norm(g) < tol:
            return np.array(x), i, np.array(history), True
        x = x - lr * g
        x = project(x, bounds)
        history.append(np.array(x))
    return np.array(x), max_iter, np.array(history), False


def gradient_descent_momentum(f, x0, lr=0.01, beta=0.85, bounds=None, tol=1e-6, max_iter=5000):
    grad_f = jax.grad(f)
    x = jnp.array(x0, dtype=float)
    v = jnp.zeros_like(x)
    history = [np.array(x)]
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        if jnp.linalg.norm(g) < tol:
            return np.array(x), i, np.array(history), True
        v = beta * v + g
        x = x - lr * v
        x = project(x, bounds)
        history.append(np.array(x))
    return np.array(x), max_iter, np.array(history), False


def gradient_descent_adam(f, x0, lr=0.05, b1=0.9, b2=0.999, eps=1e-8,
                          bounds=None, tol=1e-6, max_iter=5000):
    grad_f = jax.grad(f)
    x = jnp.array(x0, dtype=float)
    m = jnp.zeros_like(x)
    v = jnp.zeros_like(x)
    history = [np.array(x)]
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        if jnp.linalg.norm(g) < tol:
            return np.array(x), i, np.array(history), True
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g**2
        m_hat = m / (1 - b1**i)
        v_hat = v / (1 - b2**i)
        x = x - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        x = project(x, bounds)
        history.append(np.array(x))
    return np.array(x), max_iter, np.array(history), False


# gradient norm along a stored path - used for the convergence plots
def gnorm_history(f, history):
    grad_f = jax.grad(f)
    return np.array([float(jnp.linalg.norm(grad_f(jnp.array(p)))) for p in history])


# =====================================================================
# THE THREE PROBLEMS - each written for a VECTOR input x
# =====================================================================

def f1(x):
    return x[0]**2

def f2(x):
    return (6*x[0] - 2)**2 * jnp.sin(12*x[0] - 4)

def f3(x):
    return x[0]**4 - 4*x[0]*x[1] + x[1]**4


plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 9})
COL = {'plain': '#1f4e79', 'momentum': '#c0504d', 'adam': '#4f8a10'}
TOL = 1e-6


# =====================================================================
# PROBLEM 1: f(x) = x^2   -   does an "improved" method help an easy problem?
# =====================================================================

print("=" * 66)
print("PROBLEM 1:  f(x) = x^2   (unconstrained, 1 variable, x0 = 2.0)")
print("=" * 66)
print(f"{'method':<10} {'x_min':>12} {'f(x_min)':>12} {'iters':>7}  converged")
print("-" * 66)

runs1 = {}
for name, solver, kw in [("plain",    gradient_descent,          dict(lr=0.1)),
                         ("momentum", gradient_descent_momentum, dict(lr=0.1, beta=0.85)),
                         ("adam",     gradient_descent_adam,     dict(lr=0.05))]:
    x_min, n, hist, conv = solver(f1, [2.0], **kw)
    runs1[name] = hist
    print(f"{name:<10} {x_min[0]:12.6f} {float(f1(x_min)):12.6f} {n:7d}  {conv}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
xs1 = np.linspace(-2.3, 2.3, 300)
ax[0].plot(xs1, xs1**2, 'k-', lw=1.2, label='f(x) = x²')
for name, hist in runs1.items():
    h = hist[:, 0]
    ax[0].plot(h, h**2, 'o-', ms=3, lw=1.2, color=COL[name],
               markevery=max(1, len(h)//25), label=name)
ax[0].set_xlabel('x'); ax[0].set_ylabel('f(x)')
ax[0].set_title('Problem 1 — all three methods, x₀ = 2.0')
ax[0].legend()

for name, hist in runs1.items():
    ax[1].semilogy(gnorm_history(f1, hist), lw=1.5, color=COL[name], label=name)
ax[1].axhline(TOL, color='gray', ls='--', lw=1, label='tolerance')
ax[1].set_xlim(0, 300); ax[1].set_ylim(1e-8, 1e1)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('‖∇f‖')
ax[1].set_title('Plain is fastest — no pathology to fix')
ax[1].legend()
plt.tight_layout(); plt.show()


# =====================================================================
# PROBLEM 2a: plain GD - three starting points, three DIFFERENT destinations
# =====================================================================

STAT2 = {0.142589: 'local min', 1/3: 'degenerate', 0.757249: 'GLOBAL min'}
starts2 = [0.05, 0.45, 0.90]          # chosen: one per destination

print()
print("=" * 66)
print("PROBLEM 2 - PLAIN GD   (constrained [0,1]) - x0 decides the outcome")
print("=" * 66)
print(f"{'x0':>5} {'x_min':>10} {'f(x_min)':>11} {'iters':>7} {'converged':>11}   destination")
print("-" * 66)

runs2a = {}
for x0 in starts2:
    x_min, n, hist, conv = gradient_descent(f2, [x0], lr=0.001, bounds=(0.0, 1.0))
    runs2a[x0] = hist
    dest = min(STAT2, key=lambda s: abs(s - x_min[0]))
    print(f"{x0:5.2f} {x_min[0]:10.5f} {float(f2(x_min)):11.5f} {n:7d} {str(conv):>11}   {STAT2[dest]}")

xs2 = np.linspace(0, 1, 400)
fx2 = np.array([float(f2(jnp.array([v]))) for v in xs2])

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(xs2, fx2, 'k-', lw=1.2, label='f(x)')
for s, lbl in STAT2.items():
    ax[0].axvline(s, color='gray', ls=':', lw=1)
    ax[0].text(s, 15.5, lbl, rotation=90, fontsize=7, va='top', ha='right', color='gray')
for x0, hist in runs2a.items():
    h = hist[:, 0]
    fh = np.array([float(f2(jnp.array([v]))) for v in h])
    ax[0].plot(h, fh, 'o-', ms=3.5, lw=1.2, markevery=max(1, len(h)//25), label=f'x₀ = {x0}')
ax[0].set_xlabel('x'); ax[0].set_ylabel('f(x)')
ax[0].set_title('Problem 2 — plain GD, three destinations')
ax[0].legend(loc='center left')

for x0, hist in runs2a.items():
    ax[1].semilogy(gnorm_history(f2, hist), lw=1.5, label=f'x₀ = {x0}')
ax[1].axhline(TOL, color='gray', ls='--', lw=1, label='tolerance')
ax[1].set_xlim(0, 120); ax[1].set_ylim(1e-8, 1e3)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('‖∇f‖')
ax[1].set_title('x₀=0.45 flattens out but never reaches tolerance')
ax[1].legend()
plt.tight_layout(); plt.show()


# =====================================================================
# PROBLEM 2b: the one run that fails - does an improved method rescue it?
# =====================================================================

print()
print("=" * 66)
print("PROBLEM 2 - x0 = 0.45 (the plateau run) - all three methods")
print("=" * 66)
print(f"{'method':<10} {'x_min':>10} {'f(x_min)':>11} {'iters':>7}  converged")
print("-" * 66)

runs2b = {}
for name, solver, kw in [("plain",    gradient_descent,          dict(lr=0.001)),
                         ("momentum", gradient_descent_momentum, dict(lr=0.001, beta=0.85)),
                         ("adam",     gradient_descent_adam,     dict(lr=0.05))]:
    x_min, n, hist, conv = solver(f2, [0.45], bounds=(0.0, 1.0), **kw)
    runs2b[name] = hist
    print(f"{name:<10} {x_min[0]:10.5f} {float(f2(x_min)):11.5f} {n:7d}  {conv}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].axhline(1/3, color='gray', ls=':', lw=1.2)
ax[0].text(255, 1/3 + 0.006, 'degenerate  x = 1/3  (f′=f″=0)',
           fontsize=7, ha='right', va='bottom', color='gray')
ax[0].axhline(0.142589, color='gray', ls=':', lw=1.2)
ax[0].text(255, 0.142589 + 0.006, 'local minimum  x = 0.1426',
           fontsize=7, ha='right', va='bottom', color='gray')
for name, hist in runs2b.items():
    h = hist[:, 0]
    ax[0].plot(h, lw=1.6, color=COL[name], label=name)
ax[0].set_xlim(0, 260); ax[0].set_ylim(0.04, 0.47)
ax[0].set_xlabel('iteration'); ax[0].set_ylabel('x')
ax[0].set_title('Problem 2, x₀ = 0.45 — plain is trapped at the plateau')
ax[0].legend(loc='upper right')

for name, hist in runs2b.items():
    ax[1].semilogy(gnorm_history(f2, hist), lw=1.5, color=COL[name], label=name)
ax[1].axhline(TOL, color='gray', ls='--', lw=1, label='tolerance')
ax[1].set_xlim(0, 260); ax[1].set_ylim(1e-8, 1e2)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('‖∇f‖')
ax[1].set_title('plain decays too slowly to ever reach tolerance')
ax[1].legend()
plt.tight_layout(); plt.show()


# =====================================================================
# PROBLEM 3: plain GD - three starts, two minima and the SADDLE
# =====================================================================

starts3 = [[1.8, 1.2], [-1.6, -0.4], [0.05, -0.05]]      # one per destination
LABEL3 = {(1., 1.): 'minimum (1,1)', (-1., -1.): 'minimum (-1,-1)', (0., 0.): 'SADDLE (0,0)'}

print()
print("=" * 66)
print("PROBLEM 3 - PLAIN GD   (unconstrained, 2 variables)")
print("=" * 66)
print(f"{'x0':>16} {'x_min':>23} {'f':>9} {'iters':>6} {'conv':>6}   type")
print("-" * 66)

runs3 = {}
for x0 in starts3:
    x_min, n, hist, conv = gradient_descent(f3, x0, lr=0.02)
    runs3[tuple(x0)] = hist
    key = min(LABEL3, key=lambda s: (s[0]-x_min[0])**2 + (s[1]-x_min[1])**2)
    print(f"({x0[0]:6.2f},{x0[1]:6.2f})   ({x_min[0]:8.5f},{x_min[1]:9.5f}) "
          f"{float(f3(x_min)):9.5f} {n:6d} {str(conv):>6}   {LABEL3[key]}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
gg = np.linspace(-2, 2, 250)
X1, X2 = np.meshgrid(gg, gg)
Z = X1**4 - 4*X1*X2 + X2**4
ax[0].contourf(X1, X2, Z, levels=35, cmap='Blues_r')
ax[0].contour(X1, X2, Z, levels=18, colors='k', linewidths=0.25)
ax[0].plot([-2, 2], [2, -2], '--', color='#FFC000', lw=1.3, label='separatrix x₂ = −x₁')
ax[0].plot([1, -1], [1, -1], 'w*', ms=13, mec='k', mew=.6, ls='none', label='minima')
ax[0].plot([0], [0], 'wX', ms=10, mec='k', mew=.6, ls='none', label='saddle')
for x0, hist in runs3.items():
    ax[0].plot(hist[:, 0], hist[:, 1], 'o-', ms=3, lw=1.3,
               markevery=max(1, len(hist)//25), label=f'x₀ = {x0}')
ax[0].set_xlabel('x₁'); ax[0].set_ylabel('x₂'); ax[0].grid(False)
ax[0].annotate('converged here:\n‖∇f‖ < tol, but SADDLE', xy=(0.06, -0.06),
               xytext=(0.30, -1.55), fontsize=7.5, color='w', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='w', lw=1.2))
ax[0].set_title('Problem 3 — plain GD, two minima and the saddle')
ax[0].legend(fontsize=7, loc='upper left')

for x0, hist in runs3.items():
    ax[1].semilogy(gnorm_history(f3, hist), lw=1.5, label=f'x₀ = {x0}')
ax[1].axhline(TOL, color='gray', ls='--', lw=1, label='tolerance')
ax[1].set_xlim(0, 170); ax[1].set_ylim(1e-8, 1e2)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('‖∇f‖')
ax[1].set_title('all three pass the SAME test — one is not a minimum')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()


# =====================================================================
# SUMMARY: iteration count by method, across problems
# =====================================================================

cases = [("P1\nx₀=2.0",       f1, [2.0],       None,     dict(lr=0.1),   dict(lr=0.1, beta=0.85),   dict(lr=0.05)),
         ("P2\nx₀=0.45",      f2, [0.45],      (0., 1.), dict(lr=0.001), dict(lr=0.001, beta=0.85), dict(lr=0.05)),
         ("P2\nx₀=0.90",      f2, [0.90],      (0., 1.), dict(lr=0.001), dict(lr=0.001, beta=0.85), dict(lr=0.05)),
         ("P3\nx₀=(1.8,1.2)", f3, [1.8, 1.2],  None,     dict(lr=0.02),  dict(lr=0.02, beta=0.85),  dict(lr=0.05))]

names, it_p, it_m, it_a, failed = [], [], [], [], []
print()
print("=" * 66)
print("SUMMARY - iterations to converge")
print("=" * 66)
print(f"{'case':<16} {'plain':>8} {'momentum':>10} {'adam':>8}")
print("-" * 66)
for lbl, f, x0, bnd, kp, km, ka in cases:
    _, n_p, _, c_p = gradient_descent(f, x0, bounds=bnd, **kp)
    _, n_m, _, c_m = gradient_descent_momentum(f, x0, bounds=bnd, **km)
    _, n_a, _, c_a = gradient_descent_adam(f, x0, bounds=bnd, **ka)
    names.append(lbl); it_p.append(n_p); it_m.append(n_m); it_a.append(n_a)
    failed.append(not c_p)
    tag = "   <- plain did NOT converge" if not c_p else ""
    print(f"{lbl.replace(chr(10), ' '):<16} {n_p:>8} {n_m:>10} {n_a:>8}{tag}")

fig, ax = plt.subplots(figsize=(8, 3.8))
w, idx = 0.26, np.arange(len(names))
ax.bar(idx - w, it_p, w, color=COL['plain'],    label='plain')
ax.bar(idx,     it_m, w, color=COL['momentum'], label='momentum')
ax.bar(idx + w, it_a, w, color=COL['adam'],     label='adam')
for k, isfail in enumerate(failed):
    if isfail:
        ax.text(idx[k] - w, it_p[k] * 1.45, 'did not\nconverge', ha='center',
                fontsize=7, color=COL['plain'], fontweight='bold')
ax.set_yscale('log'); ax.set_ylim(5, 6e4)
ax.set_xticks(idx); ax.set_xticklabels(names)
ax.set_ylabel('iterations  (log scale)')
ax.set_title('Improvements help only where the pathology exists')
ax.legend(); ax.grid(axis='y', alpha=.3)
plt.tight_layout(); plt.show()
