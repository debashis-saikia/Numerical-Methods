import numpy as np

m = 1.0                       # m = me (natural units)
V0 = 10.0                     # eV
a = 3.0                       # Angstrom
hbar2 = 7.6199682             # hbar^2 in me*(eV)*Å^2 as given

# numerical parameters
E_min = 1e-9
E_max = V0 - 1e-9
scan_points = 20000
bisect_tol = 1e-10
max_bisect_iter = 200

# helper: alpha, beta
def alpha(E):
    return np.sqrt(2.0 * m * E / hbar2)

def beta(E):
    return np.sqrt(2.0 * m * (V0 - E) / hbar2)

# Even: alpha * tan(alpha a) - beta = 0
def f_even(E):
    aE = alpha(E)
    # avoid passing through singularities; tan will blow -> numpy handles but may be huge
    return aE * np.tan(aE * a) - beta(E)

# Odd: alpha * cot(alpha a) + beta = 0  (since alpha cot = -beta)
def f_odd(E):
    aE = alpha(E)
    # cot(x) = cos(x)/sin(x)
    sinx = np.sin(aE * a)
    cosx = np.cos(aE * a)
    # handle regions where sinx is ~0 (cot singular)
    return aE * (cosx / sinx) + beta(E)

# robust check whether function is "finite and reasonable"
def is_finite_value(v):
    return np.isfinite(v) and (abs(v) < 1e8)

# Bisection method (assumes f(a) and f(b) have opposite sign and are finite)
def bisection(func, aE, bE, tol=bisect_tol, max_iter=max_bisect_iter):
    fa = func(aE); fb = func(bE)
    if not (is_finite_value(fa) and is_finite_value(fb)):
        return None
    if fa * fb > 0:
        return None
    lo, hi = aE, bE
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = func(mid)
        if not is_finite_value(fm):
            # if mid falls near singularity, shift slightly
            mid = mid + (hi - lo) * 1e-6
            fm = func(mid)
            if not is_finite_value(fm):
                return None
        if abs(fm) < tol:
            return mid
        if fa * fm < 0:
            hi = mid
            fb = fm
        else:
            lo = mid
            fa = fm
    return 0.5 * (lo + hi)

def find_roots_for_parity(func):
    Es = np.linspace(E_min, E_max, scan_points)
    vals = np.empty_like(Es)
    for i, E in enumerate(Es):
        try:
            vals[i] = func(E)
        except Exception:
            vals[i] = np.nan

    roots = []
    for i in range(len(Es) - 1):
        f1 = vals[i]; f2 = vals[i+1]
        if not (is_finite_value(f1) and is_finite_value(f2)):
            continue
        if f1 * f2 < 0:
            root = bisection(func, Es[i], Es[i+1])
            if root is not None:
                if all(abs(root - r) > 1e-6 for r in roots):
                    roots.append(root)
    return roots

# find even and odd roots
even_roots = find_roots_for_parity(f_even)
odd_roots  = find_roots_for_parity(f_odd)

levels = []
for E in even_roots:
    levels.append((E, "Even"))
for E in odd_roots:
    levels.append((E, "Odd"))

levels.sort(key=lambda x: x[0])

print("#---------------------------------------------------")
print("# Particle in a finite Quantum Well")
print("#---------------------------------------------------")
print("# Level Energy Wavefunction parity")
for i, (E, parity) in enumerate(levels, start=1):
    print(f"{i} {E:.6f} ({parity})")
if not levels:
    print("# No bound states found (check parameters)")

