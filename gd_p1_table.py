import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

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

rows = []
for x0 in starts:
    xm, n, h, c = gradient_descent(f1, [float(x0)], lr=lr, grad_f=f1_grad)
    gfinal = float(abs(f1_grad(jnp.array(xm))[0]))
    rows.append([f"{x0:g}", f"{xm[0]:.3e}", f"{gfinal:.3e}", f"{n}", str(c)])

col_labels = ["x0", "x_final", "|grad| final", "iterations", "converged"]

fig, ax = plt.subplots(figsize=(7.5, 0.42*len(rows) + 0.6))
ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=col_labels, loc='upper center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#1f4e79')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
ax.set_title(f'Problem 1 — plain gradient descent, lr = {lr}\nconvergence summary across starting points', pad=6)
plt.tight_layout()
plt.savefig('/home/claude/qaplot/p1_table.png', dpi=140)
print("saved")
