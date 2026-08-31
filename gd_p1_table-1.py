import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

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

def f1(x):
    return x[0]**2
def f1_grad(x):
    return jnp.array([2*x[0]])

starts = [2, 10, 50, 100, 200, 500, 1000]
lr = 0.1

print("=" * 66)
print(f"PROBLEM 1 - PLAIN GRADIENT DESCENT, CONVERGENCE TABLE   lr = {lr}")
print("=" * 66)
print(f"{'x0':>7} {'x_final':>12} {'|grad| final':>14} {'iters':>8} {'converged':>11}")
print("-" * 66)
for x0 in starts:
    xm, n, h, c = gradient_descent(f1, [float(x0)], lr=lr, grad_f=f1_grad)
    gfinal = float(abs(f1_grad(jnp.array(xm))[0]))
    print(f"{x0:>7} {xm[0]:12.3e} {gfinal:14.3e} {n:8d} {str(c):>11}")
