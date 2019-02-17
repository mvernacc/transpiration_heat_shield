from matplotlib import pyplot as plt
import numpy as np

from proptools.convection import rannie_transpiration_cooling

def main():
    cool_flux_fraction = np.linspace(0, 2e-2)
    T_aw = 2000
    T_cool = 150
    r = np.array([rannie_transpiration_cooling(x, Pr_film=1.0, Re_bulk=1e5)
                  for x in cool_flux_fraction])
    T_wg = (T_aw - T_cool) / r + T_cool
    plt.plot(cool_flux_fraction, T_wg)
    plt.xlabel('Coolant mass flux fraction [-]')
    plt.ylabel('Wall temperature [K]')
    plt.show()

if __name__ == '__main__':
    main()
