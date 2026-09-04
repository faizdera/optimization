"""
==================================================================
1D BURGERS' EQUATION -- CONSOLIDATED PROGRAM
Combines all schemes developed during the project:
  1. Explicit FTBS (conservative, upwind) + CFL sweep
  2. Implicit FTBS (conservative flux form, Picard iteration)
  3. FTCS / FTFS instability demonstration (von Neumann check)
  4. Lax-Wendroff (2nd order, predictor-corrector) + multiple durations
Each section is the SAME algorithm already built and verified earlier
in the project -- only reorganized into one file.
==================================================================
"""
import numpy as np
import matplotlib.pyplot as plt

# ==================================================================
# COMMON SETUP
# ==================================================================
L = 10.0
nx = 400
x = np.linspace(0, L, nx)
dx = x[1] - x[0]

def flux(u):
    return 0.5*u**2

def initial_condition():
    return np.exp(-(x - 3.0)**2)

# analytic breaking time: Tc = -1/min(u0')
xb = 3.0 + 1/np.sqrt(2)
min_slope = -2*(xb-3.0)*np.exp(-(xb-3.0)**2)
t_break = -1.0/min_slope
print(f"Analytic breaking time Tc = {t_break:.4f}\n")

times = [0.0, 0.5, t_break, 2.0, 3.5, 5.0]


# ==================================================================
# SECTION 1 -- EXPLICIT FTBS (conservative, upwind)
# solve() is unchanged from the working version; cfl and u are
# reset before each call for the CFL sweep.
# ==================================================================
def solve_explicit_ftbs(cfl, u):
    t = 0.0
    saves = {}
    targets = sorted(times)
    ti = 0
    while ti < len(targets):
        if t >= targets[ti] - 1e-9:
            saves[targets[ti]] = u.copy()
            ti += 1
            continue
        dt = cfl * dx/max(np.max(np.abs(u)), 1e-9)
        dt = min(dt, targets[ti]-t)
        un = u.copy()
        F = flux(un)
        u[1:] = un[1:] - dt/dx*(F[1:] - F[:-1])
        u[0] = 0.0
        u[-1] = 0.0
        t += dt
    return saves

print("=== SECTION 1: Explicit FTBS -- CFL sweep ===")
cfl_values = [0.3, 0.5, 1.0, 1.5, 2.0]
all_saves_explicit = {}
for cfl_val in cfl_values:
    u0 = initial_condition()
    all_saves_explicit[cfl_val] = solve_explicit_ftbs(cfl_val, u0)

final_time = times[-1]
plt.figure(figsize=(10,6))
plt.plot(x, all_saves_explicit[0.3][final_time], label="CFL=0.3")
plt.plot(x, all_saves_explicit[0.5][final_time], label="CFL=0.5")
plt.plot(x, all_saves_explicit[1.0][final_time], label="CFL=1.0")
plt.plot(x, all_saves_explicit[1.5][final_time], label="CFL=1.5")
plt.plot(x, all_saves_explicit[2.0][final_time], label="CFL=2.0")
plt.xlabel('x (position)'); plt.ylabel('u (speed)')
plt.title('Section 1: Explicit FTBS -- CFL Sweep')
plt.legend()
plt.savefig("section1_explicit_ftbs_cfl.png", dpi=140)
plt.close()

for c in cfl_values:
    print(f"  CFL={c}: max u at t={final_time:.2f} = {all_saves_explicit[c][final_time].max():.4f}")
print()


# ==================================================================
# SECTION 2 -- IMPLICIT FTBS (conservative flux form)
# Picard iteration with damping, flux computed from the guess.
# ==================================================================
print("=== SECTION 2: Implicit FTBS (conservative flux form) ===")

nt_imp = 40
dt_imp = 0.05

u = initial_condition()
u_init_imp = u.copy()

for n in range(nt_imp):
    u_old = u.copy()
    u_guess = u_old.copy()
    for k in range(5):                                  # Picard iterations
        F_guess = flux(u_guess)
        u_new = u_old.copy()
        u_new[1:] = u_old[1:] - (dt_imp/dx)*(F_guess[1:] - F_guess[:-1])
        u_guess = 0.5*u_guess + 0.5*u_new                 # damping
    u = u_guess
    u[0] = 0.0
    u[-1] = 0.0

plt.figure(figsize=(9,5))
plt.plot(x, u_init_imp, label='initial')
plt.plot(x, u, label=f'implicit FTBS (conservative), t={dt_imp*nt_imp:.2f}')
plt.xlabel('x'); plt.ylabel('u')
plt.title('Section 2: Implicit FTBS -- Conservative Flux Form')
plt.legend()
plt.savefig("section2_implicit_ftbs_flux.png", dpi=140)
plt.close()

print(f"  max u = {u.max():.4f}, min u = {u.min():.4f}, has NaN = {np.isnan(u).any()}\n")


# ==================================================================
# SECTION 3 -- FTCS / FTFS INSTABILITY DEMONSTRATION
# Linear advection, von Neumann amplification factor check.
# ==================================================================
print("=== SECTION 3: FTCS / FTFS unconditional instability ===")

nx_s = 100
xs = np.linspace(0, 2*np.pi, nx_s, endpoint=False)
dxs = xs[1]-xs[0]
c = 1.0
dt_s = 0.05*dxs/c
k_mode = 5

# --- FTCS ---
u_ftcs = 0.001*np.sin(k_mode*xs)
sigma = c*dt_s/dxs
xi_ftcs = np.sqrt(1 + sigma**2*np.sin(k_mode*dxs)**2)
print(f"  FTCS predicted |xi| = {xi_ftcs:.6f}  (>1 => unconditionally unstable)")

for n in range(400):
    un = np.concatenate((u_ftcs[-1:], u_ftcs, u_ftcs[:1]))
    u_ftcs = u_ftcs - c*dt_s/(2*dxs)*(un[2:]-un[:-2])
print(f"  FTCS: after 400 steps, max|u| = {np.max(np.abs(u_ftcs)):.6f} (started at 0.001000)\n")

# --- FTFS ---
u_ftfs = 0.001*np.sin(k_mode*xs)
xi_ftfs = np.sqrt((1+sigma*(1-np.cos(k_mode*dxs)))**2 + (sigma*np.sin(k_mode*dxs))**2)
print(f"  FTFS predicted |xi| = {xi_ftfs:.6f}  (>1 => unconditionally unstable)")

for n in range(400):
    un = np.concatenate((u_ftfs, u_ftfs[:1]))
    u_ftfs = u_ftfs - c*dt_s/dxs*(un[1:]-un[:-1])
print(f"  FTFS: after 400 steps, max|u| = {np.max(np.abs(u_ftfs)):.6f} (started at 0.001000)\n")


# ==================================================================
# SECTION 4 -- LAX-WENDROFF (2nd order, predictor-corrector)
# Multiple total durations to show bounded (not growing) overshoot.
# ==================================================================
print("=== SECTION 4: Lax-Wendroff -- multiple durations ===")

dt_lw = 0.005

def run_lax_wendroff(t_final):
    u = initial_condition()
    t = 0.0
    while t < t_final - 1e-9:
        step = min(dt_lw, t_final-t)
        un = u.copy()
        uL, uR = un[:-1], un[1:]
        u_half = 0.5*(uL+uR) - 0.5*(step/dx)*(flux(uR)-flux(uL))   # predictor
        F_half = flux(u_half)
        u[1:-1] = un[1:-1] - step/dx*(F_half[1:]-F_half[:-1])       # corrector
        u[0] = 0.0
        u[-1] = 0.0
        t += step
    return u

durations = [0.8, t_break, 2.0, 5.0, 10.0, 20.0]

plt.figure(figsize=(10,6))
for T in durations:
    u_lw = run_lax_wendroff(T)
    note = "pre-shock" if T < t_break else "post-shock"
    print(f"  t={T:6.2f}  max u={u_lw.max():.4f}  ({note})")
    plt.plot(x, u_lw, label=f't={T} (max u={u_lw.max():.2f})')

plt.axhline(1.0, color='k', ls=':', lw=1, label='physical max (u=1)')
plt.xlim(1,9); plt.ylim(-0.2,1.6)
plt.xlabel('x'); plt.ylabel('u')
plt.title('Section 4: Lax-Wendroff -- Bounded Overshoot, Not Instability')
plt.legend(fontsize=8)
plt.savefig("section4_lax_wendroff_durations.png", dpi=140)
plt.close()

print("\nAll sections complete. Plots saved:")
print("  section1_explicit_ftbs_cfl.png")
print("  section2_implicit_ftbs_flux.png")
print("  section4_lax_wendroff_durations.png")
