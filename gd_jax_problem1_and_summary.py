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


# classify a converged point using the HESSIAN, also from JAX autograd.
# this is the second-order test the solvers themselves never perform.
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


def f1(x):
    return x[0]**2

def f2(x):
    return (6*x[0] - 2)**2 * jnp.sin(12*x[0] - 4)

def f3(x):
    return x[0]**4 - 4*x[0]*x[1] + x[1]**4


plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 9})
TOL = 1e-6


# =====================================================================
# PROBLEM 1 - one figure per method, three starting points each
#             (same format as Problems 2 and 3)
# =====================================================================

starts1 = [2.0, -1.5, 0.8]
xs1 = np.linspace(-2.4, 2.4, 300)

for title, solver, kw, xmax in [
        ("PLAIN",    gradient_descent,          dict(lr=0.1),            120),
        ("MOMENTUM", gradient_descent_momentum, dict(lr=0.1, beta=0.85), 220),
        ("ADAM",     gradient_descent_adam,     dict(lr=0.05),           320)]:

    print("=" * 70)
    print(f"PROBLEM 1 - {title}   f(x) = x^2   (unconstrained, 1 variable)")
    print("=" * 70)
    print(f"{'x0':>6} {'x_min':>12} {'f(x_min)':>12} {'iters':>7} {'conv':>7}   classification")
    print("-" * 70)

    runs = {}
    for x0 in starts1:
        x_min, n, hist, conv = solver(f1, [x0], **kw)
        runs[x0] = hist
        kind, ev = classify(f1, x_min)
        print(f"{x0:6.2f} {x_min[0]:12.6f} {float(f1(x_min)):12.6f} {n:7d} {str(conv):>7}   "
              f"{kind}  (eig {ev[0]:.1f})")
    print()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(xs1, xs1**2, 'k-', lw=1.2, label='f(x) = x²')
    ax[0].axvline(0, color='gray', ls=':', lw=1)
    ax[0].text(0, 5.6, 'global min  x = 0', rotation=90, fontsize=7,
               va='top', ha='right', color='gray')
    for x0, hist in runs.items():
        h = hist[:, 0]
        ax[0].plot(h, h**2, 'o-', ms=3.5, lw=1.2, markevery=max(1, len(h)//25),
                   label=f'x₀ = {x0}')
    ax[0].set_xlabel('x'); ax[0].set_ylabel('f(x)')
    ax[0].set_title(f'Problem 1 — {title.lower()}, three starting points')
    ax[0].legend(loc='upper center')

    for x0, hist in runs.items():
        ax[1].semilogy(gnorm_history(f1, hist), lw=1.5, label=f'x₀ = {x0}')
    ax[1].axhline(TOL, color='gray', ls='--', lw=1, label='tolerance')
    ax[1].set_xlim(0, xmax); ax[1].set_ylim(1e-8, 1e1)
    ax[1].set_xlabel('iteration'); ax[1].set_ylabel('‖∇f‖')
    ax[1].set_title('single minimum — every start reaches the same point')
    ax[1].legend()
    plt.tight_layout(); plt.show()


# =====================================================================
# MASTER COMPARISON TABLE - every problem x method x starting point
# =====================================================================

SOLVERS = [("plain",    gradient_descent,          {1: dict(lr=0.1),  2: dict(lr=0.001),             3: dict(lr=0.02)}),
           ("momentum", gradient_descent_momentum, {1: dict(lr=0.1, beta=0.85), 2: dict(lr=0.001, beta=0.85), 3: dict(lr=0.02, beta=0.85)}),
           ("adam",     gradient_descent_adam,     {1: dict(lr=0.05), 2: dict(lr=0.05),              3: dict(lr=0.05)})]

PROBLEMS = [(1, f1, [[2.0], [-1.5], [0.8]],                     None),
            (2, f2, [[0.05], [0.45], [0.90]],                   (0.0, 1.0)),
            (3, f3, [[1.8, 1.2], [-1.6, -0.4], [0.05, -0.05]],  None)]

print()
print("=" * 78)
print("MASTER COMPARISON - iterations, and CLASSIFICATION of the converged point")
print("=" * 78)
print(f"{'prob':>4} {'x0':>16} {'method':<9} {'iters':>7} {'conv':>7} {'f(x_min)':>11}   classification")
print("-" * 78)

table = {}
for pid, f, starts, bnd in PROBLEMS:
    for x0 in starts:
        for mname, solver, kws in SOLVERS:
            x_min, n, hist, conv = solver(f, x0, bounds=bnd, **kws[pid])
            kind = classify(f, x_min)[0] if conv else "- (did not converge)"
            table[(pid, tuple(x0), mname)] = (n, conv, kind)
            x0s = f"({x0[0]:.2f})" if len(x0) == 1 else f"({x0[0]:.2f},{x0[1]:.2f})"
            print(f"{pid:>4} {x0s:>16} {mname:<9} {n:>7} {str(conv):>7} "
                  f"{float(f(x_min)):11.5f}   {kind}")
        print("-" * 78)

n_runs = len(table)
n_fail = sum(1 for v in table.values() if not v[1])
n_bad = sum(1 for v in table.values() if v[1] and v[2] != "MINIMUM")
print(f"total runs: {n_runs}    did not converge: {n_fail}    "
      f"converged to a NON-minimum: {n_bad}")
print("=> the stopping test checks only  grad f = 0.  Classification needs the Hessian.")


# =====================================================================
# FIGURE: iteration count, every problem x method x start
# =====================================================================

labels, plain_it, mom_it, adam_it, fails, notmin = [], [], [], [], [], []
for pid, f, starts, bnd in PROBLEMS:
    for x0 in starts:
        x0s = f"{x0[0]:.2f}" if len(x0) == 1 else f"({x0[0]:.2f},{x0[1]:.2f})"
        labels.append(f"P{pid}\n{x0s}")
        plain_it.append(table[(pid, tuple(x0), "plain")][0])
        mom_it.append(table[(pid, tuple(x0), "momentum")][0])
        adam_it.append(table[(pid, tuple(x0), "adam")][0])
        fails.append(not table[(pid, tuple(x0), "plain")][1])
        notmin.append(any(table[(pid, tuple(x0), m)][2] == "SADDLE" for m in
                          ("plain", "momentum", "adam")))

fig, ax = plt.subplots(figsize=(12, 4.2))
w, idx = 0.26, np.arange(len(labels))
ax.bar(idx - w, plain_it, w, color='#1f4e79', label='plain')
ax.bar(idx,     mom_it,   w, color='#c0504d', label='momentum')
ax.bar(idx + w, adam_it,  w, color='#4f8a10', label='adam')
for k, bad in enumerate(fails):
    if bad:
        ax.text(idx[k] - w, plain_it[k]*1.5, 'did not\nconverge', ha='center',
                fontsize=7, color='#1f4e79', fontweight='bold')
for k, bad in enumerate(notmin):
    if bad:
        ax.text(idx[k], 3.0, 'all 3 methods\nreached the SADDLE', ha='center',
                va='bottom', fontsize=7, color='#a00000', fontweight='bold')
        ax.axvspan(idx[k]-0.45, idx[k]+0.45, color='#a00000', alpha=.07)
for xline in (2.5, 5.5):
    ax.axvline(xline, color='gray', lw=.8, ls='--')
ax.set_yscale('log'); ax.set_ylim(1.3, 6e4)
ax.set_xticks(idx); ax.set_xticklabels(labels)
ax.set_ylabel('iterations  (log scale)')
ax.set_title('Plain GD wins everywhere except the one run it cannot finish')
ax.legend(loc='upper left'); ax.grid(axis='y', alpha=.3)
plt.tight_layout(); plt.show()


# =====================================================================
# SUPPORTING EVIDENCE - momentum DOES win when its pathology is present
#   f(x) = 0.5 (x1^2 + k x2^2),  condition number k
# =====================================================================

def make_quad(k):
    def fq(x):
        return 0.5 * (x[0]**2 + k * x[1]**2)
    return fq

print()
print("=" * 70)
print("SUPPORTING TEST - ill-conditioned quadratic  f = 0.5(x1^2 + k x2^2)")
print("=" * 70)
print(f"{'kappa':>6} {'plain':>8} {'momentum':>10}   speed-up")
print("-" * 70)

kappas = [1, 2, 5, 10, 25, 50, 100]
it_plain, it_mom = [], []
for k in kappas:
    fq = make_quad(k)
    lr_p = 1.0 / k                                            # stable: lr < 2/lambda_max
    lr_m = 4.0 / (1 + np.sqrt(k))**2                          # optimal heavy-ball
    bt_m = ((np.sqrt(k) - 1) / (np.sqrt(k) + 1))**2
    _, np_, _, _ = gradient_descent(fq, [1.0, 1.0], lr=lr_p, tol=1e-6, max_iter=20000)
    _, nm_, _, _ = gradient_descent_momentum(fq, [1.0, 1.0], lr=lr_m, beta=bt_m,
                                             tol=1e-6, max_iter=20000)
    it_plain.append(np_); it_mom.append(nm_)
    print(f"{k:6d} {np_:8d} {nm_:10d}   {np_/nm_:6.2f}x")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].loglog(kappas, it_plain, 'o-', color='#1f4e79', lw=1.6, label='plain')
ax[0].loglog(kappas, it_mom,   'o-', color='#c0504d', lw=1.6, label='momentum (optimal β)')
ax[0].set_xlabel('condition number κ'); ax[0].set_ylabel('iterations')
ax[0].set_title('Momentum scales with √κ, plain GD with κ')
ax[0].legend()

ax[1].semilogx(kappas, np.array(it_plain)/np.array(it_mom), 'o-', color='k', lw=1.6)
ax[1].axhline(1, color='gray', ls='--', lw=1)
ax[1].annotate('Problem 3 sits here (κ = 2): gain is small even with\n'
               'optimal β, and with untuned β = 0.85 it is a net LOSS',
               xy=(2, np.array(it_plain)[1]/np.array(it_mom)[1]), xytext=(3.2, 6.5),
               fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=1))
ax[1].set_xlabel('condition number κ'); ax[1].set_ylabel('speed-up  (plain / momentum)')
ax[1].set_title('The improvement only pays off once κ is large')
plt.tight_layout(); plt.show()


# =====================================================================
# SUPPORTING EVIDENCE - step-size stability bound  lr < 2 / lambda_max
# =====================================================================

print()
print("=" * 70)
print("SUPPORTING TEST - step-size bound on Problem 1 (f = x^2, f'' = 2)")
print("=" * 70)
print(f"{'lr':>7} {'|1-2lr|':>9} {'iters':>8} {'converged':>11}")
print("-" * 70)

lrs = [0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.99, 1.05]
it_lr, ok_lr = [], []
for lr in lrs:
    x_min, n, hist, conv = gradient_descent(f1, [2.0], lr=lr, max_iter=3000)
    it_lr.append(n if conv else np.nan); ok_lr.append(conv)
    print(f"{lr:7.2f} {abs(1-2*lr):9.2f} {n:8d} {str(conv):>11}")
print("theory: converges iff |1 - 2*lr| < 1, i.e. 0 < lr < 1.  Exact in 1 step at lr = 0.5.")

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(lrs, it_lr, 'o-', color='#1f4e79', lw=1.6)
ax.axvline(1.0, color='#c0504d', ls='--', lw=1.4)
ax.text(1.0, 200, ' stability limit\n lr = 2/f″ = 1', color='#c0504d', fontsize=8, va='top')
ax.axvline(0.5, color='gray', ls=':', lw=1.2)
ax.text(0.5, 200, ' optimal\n lr = 1/f″', color='gray', fontsize=8, va='top', ha='right')
ax.set_yscale('log'); ax.set_xlabel('step size lr'); ax.set_ylabel('iterations (log scale)')
ax.set_title('Problem 1 — measured step-size bound matches theory exactly')
plt.tight_layout(); plt.show()
