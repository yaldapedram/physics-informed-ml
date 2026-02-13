import sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) != 2:
        print("Usage: python -m scripts.plot_pmf_demo data/sample/<file>.out")
        raise SystemExit(1)

    path = sys.argv[1]
    arr = np.genfromtxt(path)
    d = arr[:, 0]
    e = arr[:, 1]

    plt.plot(d, e, "o-")
    plt.xlabel("d-spacing")
    plt.ylabel("free energy (relative)")
    plt.title("Sample PMF-style curve")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
