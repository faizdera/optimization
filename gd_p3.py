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

def gradient_descent(f, x0, lr=0.02, tol=1e-8, max_iter=40000, grad_f=None):
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


def gradient_descent_momentum(f, x0, lr=0.02, beta=0.85, ema=False,
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
# PROBLEM 3
# =====================================================================

def f3(x):
    return x[0]**4 - 4*x[0]*x[1] + x[1]**4

def f3_grad(x):
    return jnp.array([4*x[0]**3 - 4*x[1],
                      4*x[1]**3 - 4*x[0]])

def gnorm(p):
    return float(jnp.linalg.norm(f3_grad(jnp.array(p))))

def fval(p):
    return float(f3(jnp.array(p)))


plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'font.size': 9})
C = {'plain': '#1f4e79', 'momentum': '#c0504d', 'adam': '#4f8a10'}
TOL = 1e-8
STAT = [(0.0, 0.0), (1.0, 1.0), (-1.0, -1.0)]

gg = np.linspace(-2.4, 2.4, 300)
X1, X2 = np.meshgrid(gg, gg)
Z = X1**4 - 4*X1*X2 + X2**4


def draw_field(ax, lim=2.4):
    m = np.abs(gg) <= lim
    ax.contourf(X1, X2, Z, levels=40, cmap='Blues_r')
    ax.contour(X1, X2, Z, levels=20, colors='k', linewidths=0.25)
    for s in STAT:
        ax.plot(s[0], s[1], 'w+', ms=11, mew=1.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('x₁'); ax.set_ylabel('x₂'); ax.grid(False)


print("=" * 84)
print("PROBLEM 3     f(x1, x2) = x1^4 - 4 x1 x2 + x2^4     unconstrained")
print("=" * 84)
print("df/dx1 = 4 x1^3 - 4 x2 = 0   ->   x2 = x1^3")
print("df/dx2 = 4 x2^3 - 4 x1 = 0   ->   x1 = x2^3")
print("Substituting gives x1^9 = x1, that is x1 (x1 - 1)(x1 + 1)(x1^2 + 1)(x1^4 + 1) = 0.")
print("The last two factors have no real root, so x1 is 0, 1 or -1 and there are")
print("exactly three stationary points:")
for s in STAT:
    print(f"   ({s[0]:5.2f}, {s[1]:5.2f})    f = {fval(s):9.5f}")
print()
print("Gradient check, analytic against jax.grad:")
for p in ([1.8, -1.2], [2.0, 2.0], [-2.0, -2.0]):
    a = np.array(f3_grad(jnp.array(p))); j = np.array(jax.grad(f3)(jnp.array(p)))
    print(f"   ({p[0]:5.2f},{p[1]:6.2f})   analytic [{a[0]:9.4f},{a[1]:9.4f}]"
          f"   jax [{j[0]:9.4f},{j[1]:9.4f}]   difference {np.max(np.abs(a-j)):.1e}")
print()


# ---------------------------------------------------------------------
# 1. PLAIN GRADIENT DESCENT, INCLUDING THE TWO CORNERS
# ---------------------------------------------------------------------

starts = [[2.0, 2.0], [-2.0, -2.0], [1.8, 1.2], [-1.6, -0.4],
          [-1.5, 1.6], [0.9, -1.4], [0.05, -0.05]]

print("=" * 84)
print("1. PLAIN GRADIENT DESCENT,  lr = 0.02")
print("=" * 84)
print(f"{'x0':>16} {'x_final':>24} {'f':>10} {'|grad| there':>14} {'iters':>8} {'conv':>7}")
print("-" * 84)
base = {}
for x0 in starts:
    xm, n, h, c = gradient_descent(f3, x0, lr=0.02, grad_f=f3_grad)
    base[tuple(x0)] = h
    print(f"({x0[0]:6.2f},{x0[1]:6.2f})   ({xm[0]:10.6f},{xm[1]:10.6f}) "
          f"{fval(xm):10.5f} {gnorm(xm):14.3e} {n:8d} {str(c):>7}")
print()
print("The two corner starts behave the same way as the interior ones: (2, 2) and")
print("(-2, -2) lie on the line x2 = x1, which the gradient keeps them on, and each")
print("descends to the nearest stationary point. Starting further out costs very few")
print("extra iterations because the first steps are large where the quartic is steep.")
print("The run from (0.05, -0.05) begins almost exactly on the line x2 = -x1 and")
print("finishes at the origin.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
draw_field(ax[0])
ax[0].plot([-2.4, 2.4], [2.4, -2.4], '--', color='#FFC000', lw=1.2, label='x₂ = −x₁')
for x0 in starts:
    h = base[tuple(x0)]
    ax[0].plot(h[:, 0], h[:, 1], 'o-', ms=3, lw=1.3,
               markevery=max(1, len(h)//20), label=f'({x0[0]:g}, {x0[1]:g})')
ax[0].set_title('Plain gradient descent, lr = 0.02')
ax[0].legend(fontsize=6.5, loc='upper left', ncol=2)

for x0 in starts:
    ax[1].semilogx(steps(base[tuple(x0)]), lw=1.3, label=f'({x0[0]:g}, {x0[1]:g})')
ax[1].set_yscale('log')
ax[1].axhline(TOL, color='gray', ls='--', lw=1)
ax[1].set_xlim(1, 1e3); ax[1].set_ylim(1e-10, 1e1)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('|x$_{k+1}$ − x$_k$|')
ax[1].set_title('Step size against iteration')
ax[1].legend(fontsize=6.5, ncol=2)
plt.tight_layout(); plt.show()

cps = (0, 1, 2, 5, 10, 25, 50, 100, 200)
print("=" * 104)
print("POSITION AT SELECTED ITERATIONS,  plain gradient descent, lr = 0.02")
print("=" * 104)
print(f"{'x0':<16}" + "".join(f"{('k='+str(c)):>19}" for c in cps[:5]) + f"{'final k':>10}")
print("-" * 104)
for x0 in starts:
    h = base[tuple(x0)]
    cells = "".join((f"({h[k,0]:7.3f},{h[k,1]:7.3f})" if k < len(h) else f"{'-':>19}")
                    for k in cps[:5])
    print(f"({x0[0]:6.2f},{x0[1]:6.2f}) {cells}{len(h)-1:>10}")
print()


# ---------------------------------------------------------------------
# 2. PLAIN GRADIENT DESCENT, VARYING lr
# ---------------------------------------------------------------------

print("=" * 84)
print("2. PLAIN GRADIENT DESCENT FROM (2, 2) AND (0.9, -1.4), VARYING lr")
print("=" * 84)
print("At (1, 1) the Hessian has eigenvalues 8 and 16, so near a stationary point the")
print("stability limit is lr < 2/16 = 0.125. Further out the curvature is much larger:")
print("at (2, 2) the eigenvalues are 44 and 52, giving lr < 0.038 there.")
print()
print(f"{'x0':>16} {'lr':>8} {'x_final':>24} {'iters':>8} {'conv':>7}")
print("-" * 84)
lrscan = {}
for x0 in ([2.0, 2.0], [0.9, -1.4]):
    for lr in (0.005, 0.01, 0.02, 0.035, 0.04, 0.10):
        xm, n, h, c = gradient_descent(f3, x0, lr=lr, grad_f=f3_grad)
        lrscan[(tuple(x0), lr)] = h
        pos = f"({xm[0]:10.6f},{xm[1]:10.6f})" if c else f"{'diverged':>24}"
        print(f"({x0[0]:6.2f},{x0[1]:6.2f}) {lr:8.3f} {pos} {n:8d} {str(c):>7}")
    print("-" * 84)
print("No step size in this range diverges: the quartic grows fast enough that an")
print("overlarge step is pulled straight back. What changes instead is the endpoint.")
print("From (2, 2) the count falls steadily from 357 at lr = 0.005 to 40 at lr = 0.040,")
print("all reaching (1, 1). At lr = 0.100 the first step is large enough to cross the")
print("line x2 = -x1 and the run finishes at (-1, -1) in 17 iterations. The same thing")
print("happens in the other direction from (0.9, -1.4), which ends at (-1, -1) for")
print("every lr up to 0.040 and at (1, 1) for lr = 0.100.")
print("The step size therefore selects which stationary point is reached, and the")
print("fastest run in each column is the one that changed the answer.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
draw_field(ax[0])
for lr in (0.005, 0.02, 0.035):
    h = lrscan[((0.9, -1.4), lr)]
    ax[0].plot(h[:, 0], h[:, 1], 'o-', ms=3, lw=1.3,
               markevery=max(1, len(h)//20), label=f'lr = {lr}')
ax[0].plot(0.9, -1.4, 'ks', ms=7, label='x₀')
ax[0].set_title('Paths from (0.9, −1.4), varying lr')
ax[0].legend(fontsize=8, loc='upper left')

for lr in (0.005, 0.02, 0.035):
    h = lrscan[((0.9, -1.4), lr)]
    ax[1].plot(h[:, 0], h[:, 1], lw=1.3, label=f'lr = {lr}')
for s in STAT:
    ax[1].plot(s[0], s[1], 'k+', ms=11, mew=1.6)
ax[1].set_xlabel('x₁'); ax[1].set_ylabel('x₂')
ax[1].set_xlim(-1.6, 1.6); ax[1].set_ylim(-1.6, 1.6)
ax[1].set_title('Same paths without the contour field')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------
# 3. MOMENTUM
# ---------------------------------------------------------------------

print("=" * 84)
print("3. MOMENTUM FROM (2, 2) AND (0.9, -1.4)")
print("=" * 84)
print(f"{'x0':>16} {'lr':>7} {'beta':>6} {'form':>11} {'x_final':>24} {'iters':>8} {'conv':>7}")
print("-" * 84)
mom = {}
for x0 in ([2.0, 2.0], [0.9, -1.4]):
    for lr, b, ema in [(0.02, 0.30, False), (0.02, 0.50, False), (0.02, 0.70, False),
                       (0.02, 0.85, False), (0.02, 0.95, False),
                       (0.02, 0.85, True), (0.10, 0.85, True)]:
        xm, n, h, c = gradient_descent_momentum(f3, x0, lr=lr, beta=b, ema=ema,
                                                grad_f=f3_grad)
        mom[(tuple(x0), lr, b, ema)] = h
        fm = 'averaged' if ema else 'classical'
        pos = f"({xm[0]:10.6f},{xm[1]:10.6f})" if c else f"{'diverged':>24}"
        print(f"({x0[0]:6.2f},{x0[1]:6.2f}) {lr:7.3f} {b:6.2f} {fm:>11} {pos} "
              f"{n:8d} {str(c):>7}")
    print("-" * 84)
print("From (2, 2), beta = 0.30 and 0.50 converge in 48 and 54 iterations against 92")
print("for plain descent, so the reasoning behind trying momentum holds here as it did")
print("in Problem 1: lr = 0.02 is well inside the stability limit and the accumulated")
print("velocity supplies the step size that lr alone does not. Above beta = 0.70 the")
print("count rises again, and at beta = 0.85 and 0.95 the overshoot is large enough to")
print("carry the iterate across x2 = -x1, so it finishes at (-1, -1) rather than (1, 1).")
print()
print("From (0.9, -1.4) the same thing happens but not monotonically: beta = 0.30, 0.50,")
print("0.70 and 0.95 all end at (-1, -1) while beta = 0.85 ends at (1, 1). Which side of")
print("the line the iterate happens to be on when the velocity decays is what decides")
print("the outcome, and that is not a smooth function of beta.")
print()
print("Both observations are the same point: on this surface the accumulated velocity")
print("does not improve the answer, it changes which answer is returned.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
draw_field(ax[0])
ax[0].plot([-2.4, 2.4], [2.4, -2.4], '--', color='#FFC000', lw=1.2, label='x₂ = −x₁')
for b in (0.30, 0.70, 0.85, 0.95):
    h = mom[((0.9, -1.4), 0.02, b, False)]
    ax[0].plot(h[:, 0], h[:, 1], 'o-', ms=3, lw=1.3,
               markevery=max(1, len(h)//20), label=f'β = {b}')
ax[0].plot(0.9, -1.4, 'ks', ms=7, label='x₀')
ax[0].set_title('Momentum from (0.9, −1.4), classical form, lr = 0.02')
ax[0].legend(fontsize=7.5, loc='upper left')

for b in (0.30, 0.50, 0.70, 0.85, 0.95):
    h = mom[((2.0, 2.0), 0.02, b, False)]
    ax[1].semilogy(steps(h), lw=1.3, label=f'β = {b}')
ax[1].axhline(TOL, color='gray', ls='--', lw=1)
ax[1].set_xlim(0, 900); ax[1].set_ylim(1e-10, 1e1)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('|x$_{k+1}$ − x$_k$|')
ax[1].set_title('Step size from (2, 2), varying β')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------
# 4. ADAM
# ---------------------------------------------------------------------

print("=" * 84)
print("4. ADAM FROM (2, 2) AND (0.9, -1.4)")
print("=" * 84)
print(f"{'x0':>16} {'lr':>7} {'b1':>6} {'b2':>8} {'x_final':>24} {'iters':>8} {'conv':>7}")
print("-" * 84)
ad = {}
for x0 in ([2.0, 2.0], [0.9, -1.4]):
    for lr, b1, b2 in [(0.05, 0.9, 0.999), (0.02, 0.9, 0.999), (0.10, 0.9, 0.999),
                       (0.05, 0.5, 0.999), (0.05, 0.0, 0.999), (0.05, 0.9, 0.99)]:
        xm, n, h, c, mm, vv = gradient_descent_adam(f3, x0, lr=lr, b1=b1, b2=b2,
                                                    grad_f=f3_grad)
        ad[(tuple(x0), lr, b1, b2)] = (h, mm, vv)
        pos = f"({xm[0]:10.6f},{xm[1]:10.6f})" if c else f"{'diverged':>24}"
        print(f"({x0[0]:6.2f},{x0[1]:6.2f}) {lr:7.3f} {b1:6.2f} {b2:8.3f} {pos} "
              f"{n:8d} {str(c):>7}")
    print("-" * 84)
print("From (2, 2) the count falls as lr rises, which is the opposite of the pattern")
print("for plain descent, because adam's step is held near lr and a larger lr simply")
print("covers the distance faster. Reducing b1 again removes overshoot and lowers the")
print("count, matching the behaviour seen in Problems 1 and 2.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
draw_field(ax[0])
ax[0].plot([-2.4, 2.4], [2.4, -2.4], '--', color='#FFC000', lw=1.2, label='x₂ = −x₁')
for lr in (0.10, 0.05, 0.02):
    h = ad[((0.9, -1.4), lr, 0.9, 0.999)][0]
    ax[0].plot(h[:, 0], h[:, 1], 'o-', ms=3, lw=1.3,
               markevery=max(1, len(h)//20), label=f'lr = {lr}')
ax[0].plot(0.9, -1.4, 'ks', ms=7, label='x₀')
ax[0].set_title('Adam from (0.9, −1.4), varying lr')
ax[0].legend(fontsize=8, loc='upper left')

h5, m5, v5 = ad[((2.0, 2.0), 0.05, 0.9, 0.999)]
kk = np.arange(len(m5))
gnv = np.array([gnorm(p) for p in h5[:-1]])
ax[1].semilogy(kk, gnv, lw=1.4, color='k', label='‖g‖')
ax[1].semilogy(kk, np.linalg.norm(m5, axis=1), lw=1.4, color=C['momentum'], label='‖m̂‖')
ax[1].semilogy(kk, np.linalg.norm(np.sqrt(v5), axis=1), lw=1.4, color=C['adam'], label='‖√v̂‖')
ax[1].set_xlim(0, 400); ax[1].set_ylim(1e-8, 1e3)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('magnitude')
ax[1].set_title('Adam internal quantities from (2, 2), lr = 0.05')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()


# ---------------------------------------------------------------------
# 5. THE THREE METHODS SIDE BY SIDE
# ---------------------------------------------------------------------

print("=" * 84)
print("5. THE THREE METHODS FROM EACH STARTING POINT")
print("=" * 84)
print(f"{'x0':>16} {'plain':>22} {'momentum':>22} {'adam':>22}")
print("-" * 84)
for x0 in starts:
    row = ""
    for solver, kw in [(gradient_descent, dict(lr=0.02)),
                       (gradient_descent_momentum, dict(lr=0.02, beta=0.85)),
                       (gradient_descent_adam, dict(lr=0.05))]:
        out = solver(f3, x0, grad_f=f3_grad, **kw)
        xm, n, c = out[0], out[1], out[3]
        row += (f" ({xm[0]:7.4f},{xm[1]:7.4f}){n:5d}" if c else f"{'diverged':>22}")
    print(f"({x0[0]:6.2f},{x0[1]:6.2f}){row}")
print()
print("Plain descent uses the fewest iterations from every start. The run from")
print("(0.05, -0.05) finishes at the origin under all three methods.")
print()

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
draw_field(ax[0])
sel = [('plain', gradient_descent, dict(lr=0.02)),
       ('momentum', gradient_descent_momentum, dict(lr=0.02, beta=0.85)),
       ('adam', gradient_descent_adam, dict(lr=0.05))]
for lab, solver, kw in sel:
    out = solver(f3, [2.0, 2.0], grad_f=f3_grad, **kw)
    h = out[2]
    ax[0].plot(h[:, 0], h[:, 1], 'o-', ms=3, lw=1.3, color=C[lab],
               markevery=max(1, len(h)//20), label=f'{lab} ({out[1]})')
ax[0].plot(2.0, 2.0, 'ks', ms=7, label='x₀')
ax[0].set_title('Paths from the corner (2, 2)')
ax[0].legend(fontsize=8, loc='upper left')

for lab, solver, kw in sel:
    out = solver(f3, [2.0, 2.0], grad_f=f3_grad, **kw)
    ax[1].loglog(steps(out[2]), lw=1.4, color=C[lab], label=lab)
ax[1].axhline(TOL, color='gray', ls='--', lw=1)
ax[1].set_xlim(1, 2e3); ax[1].set_ylim(1e-10, 1e1)
ax[1].set_xlabel('iteration'); ax[1].set_ylabel('|x$_{k+1}$ − x$_k$|')
ax[1].set_title('Step size from (2, 2)')
ax[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
