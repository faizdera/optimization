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


def f2(x):
    return (6*x[0] - 2)**2 * jnp.sin(12*x[0] - 4)

def f3(x):
    return x[0]**4 - 4*x[0]*x[1] + x[1]**4


plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 9})
TOL = 1e-6

xs2 = np.linspace(0, 1, 400)
fx2 = np.array([float(f2(jnp.array([v]))) for v in xs2])
STAT2 = {0.142589: 'local min', 1/3: 'degenerate', 0.757249: 'GLOBAL min'}
starts2 = [0.05, 0.45, 0.90]                              # one per destination

gg = np.linspace(-2, 2, 250)
X1, X2 = np.meshgrid(gg, gg)
Z = X1**4 - 4*X1*X2 + X2**4
starts3 = [[1.8, 1.2], [-1.6, -0.4], [0.05, -0.05]]       # one per destination
LABEL3 = {(1., 1.): 'minimum (1,1)', (-1., -1.): 'minimum (-1,-1)', (0., 0.): 'SADDLE (0,0)'}


# =====================================================================
# PROBLEM 2 - MOMENTUM and ADAM   (plain already done)
# =====================================================================

for title, solver, kw, xmax in [
        ("MOMENTUM", gradient_descent_momentum, dict(lr=0.001, beta=0.85), 260),
        ("ADAM",     gradient_descent_adam,     dict(lr=0.05),             360)]:

    print("=" * 66)
    print(f"PROBLEM 2 - {title}   (constrained [0,1], 1 variable)")
    print("=" * 66)
    print(f"{'x0':>5} {'x_min':>10} {'f(x_min)':>11} {'iters':>7} {'converged':>11}   destination")
    print("-" * 66)

    runs = {}
    for x0 in starts2:
        x_min, n, hist, conv = solver(f2, [x0], bounds=(0.0, 1.0), **kw)
        runs[x0] = hist
        dest = min(STAT2, key=lambda s: abs(s - x_min[0]))
        print(f"{x0:5.2f} {x_min[0]:10.5f} {float(f2(x_min)):11.5f} {n:7d} "
              f"{str(conv):>11}   {STAT2[dest]}")
    print()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(xs2, fx2, 'k-', lw=1.2, label='f(x)')
    for s, lbl in STAT2.items():
        ax[0].axvline(s, color='gray', ls=':', lw=1)
        ax[0].text(s, 15.5, lbl, rotation=90, fontsize=7, va='top', ha='right', color='gray')
    for x0, hist in runs.items():
        h = hist[:, 0]
        fh = np.array([float(f2(jnp.array([v]))) for v in h])
        ax[0].plot(h, fh, 'o-', ms=3.5, lw=1.2, markevery=max(1, len(h)//25), label=f'x₀ = {x0}')
    ax[0].set_xlabel('x'); ax[0].set_ylabel('f(x)')
    ax[0].set_title(f'Problem 2 — {title.lower()}, three starting points')
    ax[0].legend(loc='center left')

    for x0, hist in runs.items():
        ax[1].semilogy(gnorm_history(f2, hist), lw=1.5, label=f'x₀ = {x0}')
    ax[1].axhline(TOL, color='gray', ls='--', lw=1, label='tolerance')
    ax[1].set_xlim(0, xmax); ax[1].set_ylim(1e-8, 1e3)
    ax[1].set_xlabel('iteration'); ax[1].set_ylabel('‖∇f‖')
    ax[1].set_title('every run reaches tolerance — including x₀ = 0.45')
    ax[1].legend()
    plt.tight_layout(); plt.show()


# =====================================================================
# PROBLEM 3 - MOMENTUM and ADAM   (plain already done)
# =====================================================================

for title, solver, kw, xmax in [
        ("MOMENTUM", gradient_descent_momentum, dict(lr=0.02, beta=0.85), 210),
        ("ADAM",     gradient_descent_adam,     dict(lr=0.05),            400)]:

    print("=" * 66)
    print(f"PROBLEM 3 - {title}   (unconstrained, 2 variables)")
    print("=" * 66)
    print(f"{'x0':>16} {'x_min':>23} {'f':>9} {'iters':>6} {'conv':>6}   type")
    print("-" * 66)

    runs = {}
    for x0 in starts3:
        x_min, n, hist, conv = solver(f3, x0, **kw)
        runs[tuple(x0)] = hist
        key = min(LABEL3, key=lambda s: (s[0]-x_min[0])**2 + (s[1]-x_min[1])**2)
        print(f"({x0[0]:6.2f},{x0[1]:6.2f})   ({x_min[0]:8.5f},{x_min[1]:9.5f}) "
              f"{float(f3(x_min)):9.5f} {n:6d} {str(conv):>6}   {LABEL3[key]}")
    print()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    ax[0].contourf(X1, X2, Z, levels=35, cmap='Blues_r')
    ax[0].contour(X1, X2, Z, levels=18, colors='k', linewidths=0.25)
    ax[0].plot([-2, 2], [2, -2], '--', color='#FFC000', lw=1.3, label='separatrix x₂ = −x₁')
    ax[0].plot([1, -1], [1, -1], 'w*', ms=13, mec='k', mew=.6, ls='none', label='minima')
    ax[0].plot([0], [0], 'wX', ms=10, mec='k', mew=.6, ls='none', label='saddle')
    for x0, hist in runs.items():
        ax[0].plot(hist[:, 0], hist[:, 1], 'o-', ms=3, lw=1.3,
                   markevery=max(1, len(hist)//25), label=f'x₀ = {x0}')
    ax[0].annotate('still converges here:\n‖∇f‖ < tol, but SADDLE', xy=(0.06, -0.06),
                   xytext=(0.30, -1.55), fontsize=7.5, color='w', fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='w', lw=1.2))
    ax[0].set_xlabel('x₁'); ax[0].set_ylabel('x₂'); ax[0].grid(False)
    ax[0].set_title(f'Problem 3 — {title.lower()}, three starting points')
    ax[0].legend(fontsize=7, loc='upper left')

    for x0, hist in runs.items():
        ax[1].semilogy(gnorm_history(f3, hist), lw=1.5, label=f'x₀ = {x0}')
    ax[1].axhline(TOL, color='gray', ls='--', lw=1, label='tolerance')
    ax[1].set_xlim(0, xmax); ax[1].set_ylim(1e-8, 1e2)
    ax[1].set_xlabel('iteration'); ax[1].set_ylabel('‖∇f‖')
    ax[1].set_title('saddle is NOT avoided — no first-order method can')
    ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.show()
