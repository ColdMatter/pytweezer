import numpy as np
from scipy.constants import physical_constants
from scipy.special import sinc

m_Li = 6 * physical_constants['atomic mass constant'][0]
m_Ba = 138 * physical_constants['atomic mass constant'][0]
kB = physical_constants['Boltzmann constant'][0]


def pol1(p,x):
    return p[0]+p[1]*x


def pol1_double_offs(p, x):
    return p[0] + p[1]*(x-p[2])


def pol2(p,x):
    return p[0]+p[1]*x+p[2]*x**2


def pol10(p,x):
    result = 0
    for n, pn in enumerate(p):
        result += pn * x**n
    return result


def exponential(p, x):
    '''Exponential
    p[0]: lambda
    p[1]: amplitude
    p[2]: y-offset
    p[3]: x-offset
    '''
    _lambda = p[0]
    _A = p[1]
    _y0 = p[2]
    _x0 = p[3]
    return _A * np.exp(-_lambda * (x - _x0)) + _y0


def exponential_decay(p, x):
    '''Exponential decay e.g. of atoms in a trap.
    p[0]: lifetime
    p[1]: Scaling factor of exponential.
    p[2]: offset
    '''
    tau = p[0]
    A = p[1]
    c = p[2]
    return A*np.exp(-x/tau) + c


def tof_temperature(p, x):
    '''
    Model for the expansion of a thermal cloud.
    '''
    sigma_0 = p[0]
    T = p[1]
    t0 = p[2]
    return np.sqrt(abs(sigma_0**2 + kB*T/m_Li * (x-t0)**2))

#def spatial_thermometry(p, x):
#    '''
#    Model for the thermal spread of the ion inside the trap. Taken from Knuenz 2012 PRA.
#    '''
#    T = p[0]
#    z_point_spread_squared = p[1]**2
#    omega_scaling = p[2]
#    omega = x * omega_scaling
#    z_th_squared = kB * T / (omega**2 * m_Ba)
#    z_cam = np.real(np.sqrt(z_th_squared + z_point_spread_squared))
#    return z_cam

"""old version using a polynomial fit"""
# try:
#     p_poly_x = np.loadtxt('/home/bali/scripts/dataevaluation/2020_09_tickle_measurements/x_coeffs.txt')
#     p_poly_y = np.loadtxt('/home/bali/scripts/dataevaluation/2020_09_tickle_measurements/y_coeffs.txt')
#
#     def spatial_thermometry_scaled(p, x):
#         '''
#         Model for the thermal spread of the ion inside the trap. Taken from Knuenz 2012 PRA.
#         '''
#         T = p[0]
#         z_point_spread_squared = p[1]**2
#         #omega_scaling = p[2]
#         omega =  2 * np.pi * (np.polyval(p_poly_y,x)+np.polyval(p_poly_x,x))/2
#         z_th_squared = kB * T / (omega**2 * m_Ba)
#         z_cam = np.real(np.sqrt(z_th_squared + z_point_spread_squared))
#         return z_cam
# except:
#     spatial_thermometry_scaled = None

"""newer version wih the analytic function"""
try:
    p_x = np.loadtxt('/home/bali/scripts/dataevaluation/2021_05_tickle_measurements/2021_05_05_hf_popt.txt')
    p_y = np.loadtxt('/home/bali/scripts/dataevaluation/2021_05_tickle_measurements/2021_05_05_mf_popt.txt')

    def nu_rf(V, p):
        scaling = p[0]
        nu_dc = p[1]
        nu_rf  = scaling * V
        nu_eff = 1e3*np.sqrt(nu_rf**2 + nu_dc**2)
        return nu_eff

    def spatial_thermometry_scaled(p, x):
        '''
        Model for the thermal spread of the ion inside the trap. Taken from Knuenz 2012 PRA.
        '''
        T = p[0]
        z_point_spread_squared = p[1]**2
        #omega_scaling = p[2]
        omega_x = nu_rf(x, p_x)
        omega_y = nu_rf(x, p_y)
        omega =  2 * np.pi * np.sqrt(omega_x**2+omega_y**2)
        z_th_squared = kB * T / (omega**2 * m_Ba)
        z_cam = np.real(np.sqrt(z_th_squared + z_point_spread_squared))
        return z_cam
except:
    spatial_thermometry_scaled = None


def gaussian(p, x):
    '''
    A gaussian with constant offset
    '''
    mu = p[0]
    sigma = p[1]
    A0 = p[2]
    c = p[3]
    return A0/(np.sqrt(2*np.pi)*sigma) * np.exp(-(x-mu)**2/(2*sigma**2)) + c


def sin(p, x):
    A = p[0]
    T = p[1]
    x0 = p[2]
    y0 = p[3]
    return A * np.sin((x - x0) * 2 * np.pi / T) + y0


def gaussian_beam_intensity(p, x):
    '''
    A gaussian with constant offset
    '''
    mu = p[0]
    waist = p[1]
    A0 = p[2]
    c = p[3]
    return 2*A0/(np.sqrt(2*np.pi)*waist) * np.exp(-2*(x-mu)**2/waist**2) + c


def gaussian_beam_intensity_L(p, x):
    '''
    A gaussian with constant offset and linear background
    '''
    mu = p[0]
    waist = p[1]
    A0 = p[2]
    c = p[3]
    b = p[4]
    return 2*A0/(np.sqrt(2*np.pi)*waist) * np.exp(-2*(x-mu)**2/waist**2) + c + b*x


def lorentzian(p, x):
    mu = p[0]
    fwhm = p[1]
    A0 = p[2]
    c = p[3]
    y = (mu - x) / (fwhm/2)
    return A0 / (1 + y**2) + c


def sinc_profile(p, x):
    '''
    The squared sinc function to fit the resonance profile of squared pulse excitation.
    '''
    x0 = p[0]
    A0 = p[1]
    b = p[2]
    c = p[3]
    x_prime = b*(x-x0)
    return A0 * sinc(x_prime)**2 + c


def rabi_oscillation(p, x):
    '''
    Function describing a decohering Rabi oscillation.
    '''
    Tpi = p[0]
    omega = 2*np.pi/Tpi / 2
    tau = p[1]
    gamma_dec = 1/tau
    A0 = p[2]
    c = p[3]
    return A0/2 * (1 + np.cos(omega*x)*np.exp(-gamma_dec*x)) + c


def ac_stark_detuned_scattering(p ,x):
    '''
    Function describing the scattering with respect to detuning.
    Here the detuning is expressed with respect to the power and
    an unknown scaling factor, which we fit.
    '''
    A0 = p[0]
    s0 = p[1]
    beta = p[2]
    c = p[3]
    d = p[4]
    return A0*s0/(1+s0+(2*(0.01622070856*beta*(x-d)*1e9)/(2*np.pi*15.2e6))**2)+c


def saturation_parameter(p,x):
    '''
    Fits the saturation parameter of a given Lorentzian Profile

    '''
    A0 = p[0]
    s0 = p[1]
    c = p[2]
    return A0*s0/(1+s0+(2*2*(x-c)/(15.2))**2)


def lorentzian_cutoff(p, x):
    mu = p[0]
    fwhm = p[1]
    A0 = p[2]
    c = p[3]
    b = p[4]
    y = (mu - x) / (fwhm/2)
    return (x < b) * A0 / (1 + y**2) + c


fitfunctions = {
    '--None--': {'function_text': 'no Fitfunction', 'par_names': [], 'startvalues': [], 'function': None},
    'linear': {'name': 'linear fit', 'function_text': 'a + b⋅x', 'par_names': ['a', 'b'],
               'startvalues': [1, 0], 'function': pol1},
    'linear2': {'name': 'linear fit 2', 'function_text': 'a + b⋅(x - x0)', 'par_names': ['a', 'b', 'x0'],
                'startvalues': [0, 0, 0], 'function': pol1_double_offs},
    'quadratic': {'name': 'quadratic fit', 'function_text': 'a + b⋅x + c⋅x²', 'par_names': ['a', 'b', 'c'],
                  'startvalues': [1, 0, 0], 'function': pol2},
    'sin': {'name': 'sin fit', 'function_text': 'A⋅sin((x - x0)⋅2π / T) + y0',
            'par_names': ['A', 'T', 'x0', 'y0'], 'startvalues': [1, 6.28, 0, 0], 'function': sin},
    'pol10': {'name': 'pol10', 'function_text': '∑(a_k⋅x^k)',
              'par_names': ['a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'],
              'startvalues': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'function': pol10},
    'exponential': {'name': 'exponential', 'function_text': 'A⋅exp(-λ⋅(x - x0)) + y0',
                    'par_names': ['λ', 'A', 'y0', 'x0'], 'startvalues': [1, 1, 0, 0], 'function': exponential},
    'tof_temperature': {'name': 'TOF_temperature', 'function_text': 'sqrt(σ_0² + k_B⋅T / m⋅(t - t0)²)',
                        'par_names': ['σ_0', 'T', 't0'], 'startvalues': [1e-5, 1e-6, 18e-6],
                        'function': tof_temperature},
    'spatial_thermometry_scaled': {'name': 'Spatial thermometry', 'function_text': 'sqrt(σ_th² + σ_psf²)',
                                   'par_names': ['T', 'σ_psf'], 'startvalues': [360e-6, 10e-6],
                                   'function': spatial_thermometry_scaled},
    'exponential_decay': {'name': 'exponential_decay', 'function_text': 'A⋅exp(-x / 𝜏) + c',
                          'par_names': ['𝜏', 'A', 'c'], 'startvalues': [0.0003, 300e-6, 0.0],
                          'function': exponential_decay},
    'gaussian': {'name': 'gaussian', 'function_text': 'A0 / (sqrt(2π)⋅σ)⋅exp(-(x - µ)² / (2σ²)) + c',
                 'par_names': ['µ', 'σ', 'A0', 'c'], 'startvalues': [0, 1, 1, 0], 'function': gaussian},
    'gaussian_beam_intensity': {'name': 'gaussian_beam_intensity',
                                'function_text': '2 A0 / (sqrt(2π)⋅w_r)⋅exp(-2(x - µ)² / w_r²) + c',
                                'par_names': ['µ', 'w_r', 'A0', 'c'], 'startvalues': [50, 10, -1, 0],
                                'function': gaussian_beam_intensity},
    'gaussian_beam_intensity_L': {'name': 'gaussian_beam_intensity_L',
                                  'function_text': '2 A0 / (sqrt(2π)⋅w_r)⋅exp(-2(x - µ)² / w_r²) + b⋅x + c',
                                  'par_names': ['µ', 'w_r', 'A0', 'c', 'b'], 'startvalues': [50, 10, -1, 0, 0],
                                  'function': gaussian_beam_intensity_L},
    'ac_stark_detuned_scattering': {'name': 'ac_stark_detuned_scattering',
                                    'function_text': 'A0⋅s0 / (1 + s0 + (2⋅(0.01622070856⋅β⋅(x - d)⋅1e9)/'
                                                     '(2π⋅15.2e6))²) + c',
                                    'par_names': ['A0', 's0', 'β', 'c', 'd'], 'startvalues': [1, 0.1, 30, 0, 0],
                                    'function': ac_stark_detuned_scattering},
    'rabi_oscillation': {'name': 'rabi_oscillation', 'function_text': 'A0 / 2⋅(1 + cos(ω⋅t)⋅exp(-γ_dec⋅t)) + c',
                         'par_names': ['T_π', '𝜏', 'A0', 'c'], 'startvalues': [400e-6, 10e-3, 2e4, 0],
                         'function': rabi_oscillation},
    'saturation_parameter': {'name': 'saturation_parameter',
                             'function_text': ' A0⋅s0 / (1 + s0 + (4⋅(x - c) / (15.2⋅sqrt(1 + s0))²)',
                             'par_names': ['A0', 's0', 'c'], 'startvalues': [1, 5, 107.3],
                             'function': saturation_parameter},
    'lorentzian': {'name': 'lorentzian', 'function_text': 'A0 / (1 + (µ - x / (fwhm / 2))²) + c',
                   'par_names': ['µ', 'fwhm', 'A0', 'c'], 'startvalues': [318, 6, 18e3, 0], 'function': lorentzian},
    'lorentzian_cutoff': {'name': 'lorentzian cutoff',
                          'function_text': '(x < b) ⋅ A0 / (1 + (µ - x / (fwhm / 2))²) + c',
                          'par_names': ['µ', 'fwhm', 'A0', 'c', 'b'], 'startvalues': [318, 6, 18e3, 0, 318],
                          'function': lorentzian_cutoff},
    'sinc_profile': {'name': 'sinc_profile', 'function_text': 'A0⋅sinc(b⋅(x - x0))² + c',
                     'par_names': ['x0', 'A0', 'b', 'c'], 'startvalues': [73.739e6, -2.3e4, 7e-4, 23e3],
                     'function': sinc_profile}
}
