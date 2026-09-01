import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

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

def gradient_descent_momentum(f, x0, lr=0.001, beta=0.85, tol=1e-8, max_iter=40000, grad_f=None):
    if grad_f is None:
        grad_f = jax.grad(f)
    x = jnp.array(x0, dtype=float)
    v = jnp.zeros_like(x)
    hist = [np.array(x)]
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        v = beta*v + g
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
    for i in range(1, max_iter + 1):
        g = grad_f(x)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g**2
        m_hat = m / (1 - b1**i); v_hat = v / (1 - b2**i)
        x_new = x - lr * m_hat / (jnp.sqrt(v_hat) + eps)
        if not np.all(np.isfinite(np.array(x_new))):
            return np.array(x), i, np.array(hist), False
        hist.append(np.array(x_new))
        if jnp.linalg.norm(x_new - x) < tol:
            return np.array(x_new), i, np.array(hist), True
        x = x_new
    return np.array(x), max_iter, np.array(hist), False


def f2(x):
    return (6*x[0] - 2)**2 * jnp.sin(12*x[0] - 4)

def f2_grad(x):
    u = 6*x[0] - 2
    return jnp.array([12*u*(jnp.sin(12*x[0]-4) + u*jnp.cos(12*x[0]-4))])

def gval(v):
    return float(f2(jnp.array([v])))

def gnorm(v):
    return float(abs(f2_grad(jnp.array([v]))[0]))


starts = [0.05, 0.20, 0.30, 0.45, 0.60, 0.90]

# best configuration found for each method from the earlier sweeps
CFG = {
    'plain':    dict(lr=0.001),
    'momentum': dict(lr=0.001, beta=0.85),
    'adam':     dict(lr=0.05, b1=0.7, b2=0.999),
}

print("=" * 90)
print("PROBLEM 2 - ALL STARTING POINTS, EACH METHOD AT ITS BEST CONFIGURATION")
print("=" * 90)
print(f"plain:    lr = {CFG['plain']['lr']}")
print(f"momentum: lr = {CFG['momentum']['lr']}, beta = {CFG['momentum']['beta']}")
print(f"adam:     lr = {CFG['adam']['lr']}, b1 = {CFG['adam']['b1']}, b2 = {CFG['adam']['b2']}")
print("-" * 90)
print(f"{'x0':>6} {'method':<10} {'x_final':>11} {'f(x_final)':>12} {'|f'+chr(39)+'| there':>12} {'iters':>8} {'conv':>7}")
print("-" * 90)

for x0 in starts:
    xm, n, h, c = gradient_descent(f2, [x0], grad_f=f2_grad, **CFG['plain'])
    print(f"{x0:6.2f} {'plain':<10} {xm[0]:11.6f} {gval(xm[0]):12.6f} {gnorm(xm[0]):12.3e} {n:8d} {str(c):>7}")

    xm, n, h, c = gradient_descent_momentum(f2, [x0], grad_f=f2_grad, **CFG['momentum'])
    print(f"{x0:6.2f} {'momentum':<10} {xm[0]:11.6f} {gval(xm[0]):12.6f} {gnorm(xm[0]):12.3e} {n:8d} {str(c):>7}")

    xm, n, h, c = gradient_descent_adam(f2, [x0], grad_f=f2_grad, **CFG['adam'])
    print(f"{x0:6.2f} {'adam':<10} {xm[0]:11.6f} {gval(xm[0]):12.6f} {gnorm(xm[0]):12.3e} {n:8d} {str(c):>7}")
    print("-" * 90)
