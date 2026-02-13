import os
import numpy as np
import matplotlib.pyplot as plt
import statistics


base_dir = "/path/to/Semi_infinite/NaMMT"   # contains Na_-4/, Na_-3/, ..., each has NaMMT.out
out_file = "NaMMT.out"

skip_header = 345  # your LAMMPS output header length

# column indices in NaMMT.out after the header (0-based)
COL_TIME = 0
COL_FORCEZ_DOWN = 19
COL_FORCEZ_UP = 22
COL_Z_TOP = 13
COL_Z_BOTTOM = 16

# range of folders you have
i_min = -4
i_max = 101


def Trap_yalda(X, Y, n):
    """Cumulative trapezoid integration. Returns (Sum, X_array, Y_array)."""
    Sum = 0.0
    Y_array = [0.0]
    X_array = [X[0]]

    for i in range(0, n - 1):
        dx = (X[i + 1] - X[i]) / 2.0
        Sum += dx * (Y[i] + Y[i + 1])
        Y_array.append(-1.0 * Sum)    # minus sign: PMF = -∫F dd
        X_array.append(X[i + 1])

    return Sum, np.array(X_array), np.array(Y_array)


# ----------------------------
# MAIN
# ----------------------------
z = np.zeros((i_max - i_min + 1, 5))  # [d_avg, f_down_avg, f_up_avg, stdev_down, se_down]
row = 0

for i in range(i_min, i_max + 1):
    path = os.path.join(base_dir, f"Na_{i}", out_file)

    if not os.path.exists(path):
        print("Missing:", path)
        continue

    f = np.genfromtxt(path, dtype=float, skip_header=skip_header)
    if f.ndim == 1:
        f = f[None, :]  # single-row safeguard

    # time = f[:, COL_TIME] / 1000.0  # optional conversion if you want ps

    forcezDown = f[:, COL_FORCEZ_DOWN]
    forcezUp = f[:, COL_FORCEZ_UP]

    # separation (d-spacing-like): z_top - z_bottom
    d = f[:, COL_Z_TOP] - f[:, COL_Z_BOTTOM]

    d_avg = np.sum(d) / len(d)
    fdown_avg = np.sum(forcezDown) / len(forcezDown)
    fup_avg = np.sum(forcezUp) / len(forcezUp)

    SSDfDown = statistics.stdev(forcezDown) if len(forcezDown) > 1 else 0.0
    SE = SSDfDown / np.sqrt(len(forcezDown)) if len(forcezDown) > 1 else 0.0

    z[row, 0] = d_avg
    z[row, 1] = fdown_avg
    z[row, 2] = fup_avg
    z[row, 3] = SSDfDown
    z[row, 4] = SE
    row += 1

# trim unused rows if some files were missing
z = z[:row, :]

# sort by d-spacing
z = z[z[:, 0].argsort()]

# build PMF from force profile
x_arr = z[:, 0]     # distances
y_arr = z[:, 1]     # forcezDown avg (or choose z[:,2] if you want forcezUp)

summation, x_int, pmf = Trap_yalda(x_arr, y_arr, len(x_arr))
pmf = pmf - np.min(pmf)   # shift minimum to 0

# save
os.makedirs("data/sample", exist_ok=True)
out_path = os.path.join("data/sample", "pmf_semiperiodic_na_example.out")
np.savetxt(out_path, np.column_stack((x_int, pmf)))
print("Saved:", out_path)

# plot
plt.figure(figsize=(8, 5))
plt.plot(x_int, pmf, "o-")
plt.xlabel("d-spacing")
plt.ylabel("PMF (relative)")
plt.tight_layout()
plt.show()
