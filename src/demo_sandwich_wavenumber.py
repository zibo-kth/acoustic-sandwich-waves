"""Demo: compute k(omega) for a sandwich panel and plot Re/Im parts.

Run:
  python3 -m pip install numpy scipy matplotlib
  python3 src/demo_sandwich_wavenumber.py
"""

from __future__ import annotations

import numpy as np

from sandwich_wavenumber import Core, Params, Skin, SolverOpts, sandwich_wavenumber


def main():
    # Frequencies (Hz)
    freq = np.linspace(20, 2000, 200)

    # Bottom skin (y=0)
    skin1 = Skin(E=70e9, nu=0.33, rho=2700.0, t=1.0e-3, eta=0.01)

    # Top skin (y=h)
    skin3 = Skin(E=skin1.E, nu=skin1.nu, rho=skin1.rho, t=skin1.t, eta=skin1.eta)

    # Core
    core = Core(E=200e6, nu=0.25, rho=120.0, h=20e-3, eta=0.05)

    p = Params(skin1=skin1, skin3=skin3, core=core)
    opts = SolverOpts(use_root=True, do_newton_polish=True, k0_imag=0.0)

    k, info = sandwich_wavenumber(freq, p, opts)

    print(info)

    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(freq, np.real(k), linewidth=1.5)
        plt.grid(True)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Re{k} (rad/m)")
        plt.title("Sandwich wavenumber: real part")

        plt.figure()
        plt.plot(freq, np.imag(k), linewidth=1.5)
        plt.grid(True)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Im{k} (1/m)")
        plt.title("Sandwich wavenumber: imaginary part")

        plt.show()
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping plots.")


if __name__ == "__main__":
    main()
