import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# =====================================================================
# SAME GENERALIZED SOLVERS
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


def gnorm_history(f, history):
    grad_f = jax.grad(f)
    return np.array([float(jnp.linalg.norm(grad_f(jnp.array(p)))) for p in history])


def classify(f, x, zero=1e-6):
    H = np.array(jax.hessian(f)(jnp.array(x, dtype=float)))
    ev = np.linalg.eigvalsh(np.atleast_2d(H))
    if np.all(ev > zero):
        return "MINIMUM", ev
    if np.all(ev < -zero):
        return "maximum", ev
    if np.any(ev > zero) and np.any(ev < -zero):
        return "SADDLE", ev
    return "DEGENERATE", ev


def f3(x):
    return x[0]**4 - 4*x[0]*x[1] + x[1]**4


plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 9})
COL = {'plain': '#1f4e79', 'momentum': '#c0504d', 'adam': '#4f8a10'}
TOL = 1e-6

METHODS = [("plain",    gradient_descent,          dict(lr=0.02)),
           ("momentum", gradient_descent_momentum, dict(lr=0.02, beta=0.85)),
           ("adam",     gradient_descent_adam,     dict(lr=0.05))]

gg = np.linspace(-2, 2, 250)
X1, X2 = np.meshgrid(gg, gg)
Z = X1**4 - 4*X1*X2 + X2**4


def run_all(x0):
    out = {}
    print("=" * 76)
    print(f"PROBLEM 3 - x0 = ({x0[0]}, {x0[1]})   all three methods")
    print("=" * 76)
    print(f"{'method':<10} {'x_min':>22} {'f(x_min)':>10} {'iters':>7} {'conv':>7}   classification")
    print("-" * 76)
    for name, solver, kw in METHODS:
        x_min, n, hist, conv = solver(f3, x0, **kw)
        out[name] = (hist, n)
        kind, ev = classify(f3, x_min)
        print(f"{name:<10} ({x_min[0]:9.6f},{x_min[1]:10.6f}) {float(f3(x_min)):10.5f} "
              f"{n:7d} {str(conv):>7}   {kind}  (eig {ev[0]:+.1f}, {ev[1]:+.1f})")
    return out


# =====================================================================
# COMPARISON A:  x0 = (0.05, -0.05)   the saddle start
# =====================================================================

runsA = run_all([0.05, -0.05])
print("=> all three report success at a point that is NOT a minimum.")
print("   Adam is the fastest method here - to the wrong answer.\n")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))

zg = np.linspace(-0.09, 0.09, 200)                 # zoom: the action is tiny
ZX1, ZX2 = np.meshgrid(zg, zg)
ZZ = ZX1**4 - 4*ZX1*ZX2 + ZX2**4
ax[0].contourf(ZX1, ZX2, ZZ, levels=30, cmap='Blues_r')
ax[0].contour(ZX1, ZX2, ZZ, levels=15, colors='k', linewidths=0.25)
ax[0].plot([-0.09, 0.09], [0.09, -0.09], '--', color='#FFC000', lw=1.3,
           label='separatrix x₂ = −x₁')
ax[0].plot([0], [0], 'wX', ms=11, mec='k', mew=.7, ls='none', label='saddle (0,0)')
for name, (hist, n) in runsA.items():
    ax[0].plot(hist[:, 0], hist[:, 1], 'o-', ms=4, lw=1.4, color=COL[name],
               markevery=max(1, len(hist)//20), label=f'{name} ({n} it)')
ax[0].plot(0.05, -0.05, 'ks', ms=7, label='x₀')
ax[0].set_xlabel('x₁'); ax[0].set_ylabel('x₂'); ax[0].grid(False)
ax[0].set_title('Problem 3, x₀ = (0.05, −0.05) — zoomed on the saddle')
ax[0].legend(fontsize=7, loc='upper right')

for name, (hist, n) in runsA.items():
    ax[1].semilogy(gnorm_history(f3, hist), 'o-', ms=3, lw=1.5, color=COL[name],
                   markevery=max(1, len(hist)//25), label=f'{name} ({n} it)')
ax[1].axhline(TOL, color='gray', ls='--', lw=1, label='tolerance')
ax[1].set_xlim(-4, 165); ax[1].set_ylim(1e-9, 1e1)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('‖∇f‖')
ax[1].set_title('Adam converges 76× faster — to a NON-minimum')
ax[1].legend()
plt.tight_layout(); plt.show()


# =====================================================================
# COMPARISON B:  x0 = (0.90, -1.40)   near the separatrix
# =====================================================================

runsB = run_all([0.90, -1.40])
print("=> identical problem, identical x0, both answers f = -2 - but DIFFERENT minima.")
print("   Changing the optimiser changed the solution returned.\n")

fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
ax[0].contourf(X1, X2, Z, levels=35, cmap='Blues_r')
ax[0].contour(X1, X2, Z, levels=18, colors='k', linewidths=0.25)
ax[0].plot([-2, 2], [2, -2], '--', color='#FFC000', lw=1.3, label='separatrix x₂ = −x₁')
ax[0].plot([1, -1], [1, -1], 'w*', ms=13, mec='k', mew=.6, ls='none', label='minima')
ax[0].plot([0], [0], 'wX', ms=10, mec='k', mew=.6, ls='none', label='saddle')
for name, (hist, n) in runsB.items():
    ax[0].plot(hist[:, 0], hist[:, 1], 'o-', ms=3, lw=1.4, color=COL[name],
               markevery=max(1, len(hist)//25), label=f'{name} ({n} it)')
ax[0].plot(0.90, -1.40, 'ks', ms=7, label='x₀')
ax[0].annotate('momentum crosses the separatrix\nand ends in the OTHER basin',
               xy=(0.62, 0.55), xytext=(-1.92, 1.30), fontsize=7.5, color='w',
               fontweight='bold', arrowprops=dict(arrowstyle='->', color='w', lw=1.2))
ax[0].set_xlabel('x₁'); ax[0].set_ylabel('x₂'); ax[0].grid(False)
ax[0].set_title('Problem 3, x₀ = (0.90, −1.40) — same start, different answers')
ax[0].legend(fontsize=7, loc='upper right')

for name, (hist, n) in runsB.items():
    ax[1].semilogy(gnorm_history(f3, hist), lw=1.5, color=COL[name], label=f'{name} ({n} it)')
ax[1].axhline(TOL, color='gray', ls='--', lw=1, label='tolerance')
ax[1].set_xlim(0, 340); ax[1].set_ylim(1e-8, 1e2)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('‖∇f‖')
ax[1].set_title('all three converge cleanly — the disagreement is invisible here')
ax[1].legend()
plt.tight_layout(); plt.show()
