"""Back-of-the-envelop math on a transpiration-cooled metallic heatshield."""
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import trapz
import skaero.atmosphere.coesa as atmo

from proptools.convection import rannie_transpiration_cooling

# Air gas properties
R = 287.
gamma = 1.4

def get_coolant_flux(M_inf, p_inf, T_inf, T_wg_max, T_cool):
    """
    Warning: it is highly dubious to apply the Rannie equation
    to a high-Mach external flow.
    """

    # Compute the main flow mass flux across the ehat shield
    # TODO correct for being behind a shock
    rho = p_inf / (R * T_inf)
    v = M_inf * (gamma * R * T_inf)**0.5
    mass_flux_external = rho * v

    L = 10    # Heat shield length scale [units: meter].
    mu = 1.7e-5    # Viscosity of air [units: pascal second].
    Re_bulk = rho * v * L / mu
    Pr_film = 1.

    # Adiabatic wall temperature for external flow
    r = 0.85    # recovery factory
    T_aw = (1 + r * (gamma - 1)/2 * M_inf**2) * T_inf

    def solve_fun(cool_flux_fraction):
        if cool_flux_fraction < 0 or cool_flux_fraction > 1:
            return np.inf
        temp_ratio = rannie_transpiration_cooling(
            cool_flux_fraction, Pr_film, Re_bulk)
        T_wg = (T_aw - T_cool) / temp_ratio + T_cool
        return T_wg - T_wg_max

    cool_flux_fraction = fsolve(solve_fun, 1e-3)[0]
    return cool_flux_fraction * mass_flux_external

def atmo_temperature_and_pressure(alt):
    """
    See https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
    Arguments:
        altitude [units: meter].
    Returns:
        tuple: temp [units: kelvin], pressure [units: pascal]
    """
    if alt > 25e3:
        temp = - 131.21 + 0.00299 * alt
        pressure = 2.488 * ((temp + 273.15)/216.6)**(-11.388)
    elif alt > 11e3:
        temp = - 56.46
        pressure = 22.65 * np.exp(1.73 - 0.000157 * alt)
    else:
        temp = 15.04 - 0.00649 * alt
        pressure = 101.29 * ((temp + 273.15)/288.08)**(5.256)
    temp = temp + 273.15
    pressure = pressure * 1e3
    return (temp, pressure)


def shuttle_traj():
    """Estiamte the coolant required by a transpiration heat shield on a shuttle-like
    trajectory."""
    T_cool = 150    # Coolant temperature [units: kelvin].
    T_wg_max = 1170    # Wall max temp. 304 SS solidus temp - 500 K.
    
    # Shuttle reentry trajectory
    # linear fit to https://i.stack.imgur.com/ki1hs.gif
    # from https://physics.stackexchange.com/questions/377212/why-do-spaceships-heat-up-when-entering-earth-but-not-when-exiting
    time = np.linspace(0, 23*60)
    alt = 80 * (time[-1] - time)/time[-1] + 20
    velocity = 7500 * (time[-1] - time)/time[-1]

    p_inf = np.zeros(len(time))
    T_inf = np.zeros(len(time))
    T_aw = np.zeros(len(time))
    M_inf = np.zeros(len(time))
    coolant_flux = np.zeros(len(time))

    # Step through the trajectory, computing the pressure, temperature
    # and required cooland flux at each step.
    for i in range(len(time)):
        T_inf[i], p_inf[i] = atmo_temperature_and_pressure(alt[i] * 1e3)
        M_inf[i] = velocity[i] / (gamma * R * T_inf[i])**0.5
        r = 0.85    # recovery factory
        T_aw[i] = (1 + r * (gamma - 1)/2 * M_inf[i]**2) * T_inf[i]
        # print('{:.4f}, {:.4f}'.format(M_inf[i], T_inf[i]))
        coolant_flux[i] = get_coolant_flux(
            M_inf[i], p_inf[i], T_inf[i], T_wg_max, T_cool)

    # Integrate the coolant used per area over the trajectory
    # [units: kilogram meter**2].
    coolant_used_per_area = trapz(coolant_flux, time)
    print('Coolant used = {:.1f} kg m^-2'.format(coolant_used_per_area))

    # Steel sheet mass/area
    steel_density = 8000    # [units: kilogram meter**-3].
    steel_thickness = 1e-3    # [units: meter].
    steel_mass_per_area = steel_density * steel_thickness
    print('+ steel_sheet = {:.1f} kg m^-2'.format(steel_mass_per_area))
    print('transp. heat shield total = {:.3} kg^-2'.format(
        steel_mass_per_area + coolant_used_per_area))


    # Shuttle insulation mass/area for comparison
    # Source: https://en.wikipedia.org/wiki/Space_Shuttle_thermal_protection_system#High-temperature_reusable_surface_insulation_(HRSI)
    shuttle_hrsi_density = 140    # [units: kilogram meter**-3].
    shuttle_hrsi_thickness = 10e-2    # [units: meter].
    shuttle_hsri_mass_per_area = shuttle_hrsi_density * shuttle_hrsi_thickness
    print('versus')
    print('Shuttle HRSI = {:.1f} kg m^-2'.format(shuttle_hsri_mass_per_area))

    # Plot results
    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(time, p_inf * 1e-3)
    plt.ylabel('Static pressure [kPa]')
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(time, T_inf, color='blue', label='static')
    plt.plot(time, T_aw, color='red', label='adib. wall')
    plt.grid()
    plt.legend()
    plt.ylabel('Temperature [K]')

    plt.subplot(4, 1, 3)
    plt.plot(time, M_inf)
    plt.grid()
    plt.ylabel('Mach number [-]')

    plt.subplot(4, 1, 4)
    plt.plot(time, coolant_flux)
    plt.grid()
    plt.ylabel('Coolant flux [kg m^-2 s^-1]')


def simple_demo():
    M_inf = 25
    p_inf = 5    # pressure at 70 km alt [units: pascal].
    T_inf = 220    # temperature at 70 km alt [units: kelvin].
    T_cool = 150    # Coolant temperature [units: kelvin].
    T_wg_max = 1170    # Wall max temp. 304 SS solidus temp - 500 K.
    coolant_flux = get_coolant_flux(M_inf, p_inf, T_inf, T_wg_max, T_cool)
    print('coolant flux = {:.4f} kg m^-2 s^-1'.format(coolant_flux))

if __name__ == '__main__':
    simple_demo()
    shuttle_traj()
    plt.show()
