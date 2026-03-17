import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
import time

start_time = time.time()
np.random.seed()

# ------------------------------kmgt
# Fabry–Pérot cavity parameters
# ------------------------------
L = 0.1                      # FP cavity length (m)
c = 3e8
n0 = 1.46

# Mirror transmissions
T1 = 0.01                    # Input mirror transmission
T2 = 0.01                    # Output mirror transmission
alpha = (T1 + T2) / 2    # Effective cavity loss
theta = T1          # Input coupling

# Effective round-trip length (forth and back)
L_eff = L

# Free spectral range for FP cavity
FSR = c / (2 * n0 * L)
t_R = 1 / FSR

# ------------------------------
# Nonlinear/dispersion parameters
# ------------------------------
Pin = 1
t_pulse = 1000e-12
gamma = 0.001

beta2 = -2.2e-26
beta3 = 0
beta4 = 0

lambda_pump = 1550e-9
nt = 2**13                   # Number of temporal (spectral) points

Tmax = L / 80 * n0 / c
dT = Tmax / nt
T = np.linspace(-nt/2, nt/2-1, nt) * dT  # Temporal window
f = (1 / Tmax) * np.concatenate([np.arange(0, nt/2), np.arange(-nt/2, 0)])  # Frequency window
df = f[1] - f[0]
omega = 2 * np.pi * f

# Pump field (temporal)
At_pump = np.sqrt(Pin) * (1 / np.cosh(T / t_pulse))

# ------------------------------
# Time stepping
# ------------------------------
nloops = int(4e4)
nplot = int(3e3)

n_start = 0
n_step = 1 / 20
n_stop = nloops
num_steps = int((n_stop - n_start) / n_step)

t_slow_start = n_start * t_R
t_slow_step = n_step * t_R
t_slow_stop = n_stop * t_R

delta0_start = -1 * alpha
delta0_stop = 5 * alpha

# Dispersive operator (with L_eff instead of L)
dispersive_op = np.exp((1j/2 * beta2 * (2*np.pi*f)**2 +
                        1j/6 * beta3 * (2*np.pi*f)**3 +
                        1j/24 * beta4 * (2*np.pi*f)**4) *
                       t_slow_step / t_R * L_eff)

# Initial intracavity field
Anl = At_pump * np.zeros_like(T, dtype=complex)

co = 0
co_plot = 0

Aout_t = []
Aout_v = []
t_slow_plot = []
delta0_plot = []

# ------------------------------
# Main loop
# ------------------------------
for tslow in np.arange(t_slow_start, t_slow_stop, t_slow_step):
    co += 1
    delta0 = delta0_start + (delta0_stop - delta0_start) * tslow / t_slow_stop

    # Pump term (same structure, but theta = T1 now)
    S = np.sqrt(theta) / t_R * At_pump * (1 + 1e-6 * np.random.rand(*T.shape))

    # Dispersive step
    Ad = fft(dispersive_op * ifft(Anl))

    # Nonlinear step (with L_eff instead of L)
    K = 1 / t_R * (-alpha + 1j * gamma * L_eff * np.abs(Ad)**2 - 1j * delta0)
    Anl = (Ad + S / K) * np.exp(K * t_slow_step) - S / K

    if (co / num_steps * nplot) > co_plot + 1:
        co_plot += 1

        Aout = Anl
        Aout_t.append(Aout)
        Aout_v.append(fftshift(ifft(Aout)))
        t_slow_plot.append(tslow)
        delta0_plot.append(delta0)

        print(f"Step {co_plot}/{nplot}, Delta0: {delta0:.4f}")

        plt.figure(99)
        plt.subplot(2, 1, 1)
        plt.plot(fftshift(f) / 1e12, 10 * np.log10(np.abs(fftshift(ifft(Aout)))**2))
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Power (dB)")

        plt.subplot(2, 1, 2)
        plt.semilogy(T * 1e12, np.abs(Aout)**2)
        plt.xlabel("Time (ps)")
        plt.ylabel("Power")
        plt.pause(0.1)
        plt.figure(99).clear()

# Convert results
Aout_t = np.array(Aout_t)
Aout_v = np.array(Aout_v)
t_slow_plot = np.array(t_slow_plot)
delta0_plot = np.array(delta0_plot)
fplot = fftshift(f)

# ------------------------------
# Post-processing plots
# ------------------------------
plt.figure(11)
vmax = 20e12
vmax_ind = np.argmin(np.abs(fplot - vmax))
vmin_ind = np.argmin(np.abs(fplot + vmax))
sigplot = 10 * np.log10(np.abs(Aout_v[:, vmin_ind:vmax_ind])**2)
plt.pcolormesh(fplot[vmin_ind:vmax_ind] / 1e12, delta0_plot / alpha, sigplot, shading='auto')
plt.title("Spectral")
plt.xlabel("Frequency (THz)")
plt.ylabel(r"$\Delta_0$")
plt.colorbar()
plt.show()

plt.figure(12)
vmax = 6e12
vmax_ind = np.argmin(np.abs(fplot - vmax))
vmin_ind = np.argmin(np.abs(fplot + vmax))
sigplot = 10 * np.log10(np.abs(Aout_v[:, vmin_ind:vmax_ind])**2)
sigplot -= np.max(sigplot)
sigplot[sigplot < -40] = -40
plt.pcolormesh(fplot[vmin_ind:vmax_ind] / 1e12, delta0_plot / alpha, sigplot, shading='auto')
plt.title("Spectral (Log Scale)")
plt.xlabel("Frequency (THz)")
plt.ylabel(r"$\Delta_0$")
plt.colorbar()
plt.show()

plt.figure(13)
tmax_ind = 4095
tmin_ind = 0
sigplot = np.abs(Aout_t[:, tmin_ind:tmax_ind])**2
plt.pcolormesh(T[tmin_ind:tmax_ind] / 1e-12, delta0_plot / alpha, sigplot, shading='auto')
plt.title("Temporal")
plt.xlabel("Time (ps)")
plt.ylabel(r"$\Delta_0$")
plt.colorbar()
plt.show()

plt.figure(14)
sigplotT = 10 * np.log10(np.abs(Aout_t[:, tmin_ind:tmax_ind])**2)
sigplotT -= np.max(sigplotT)
sigplotT[sigplotT < -60] = -60
plt.pcolormesh(T[tmin_ind:tmax_ind] / 1e-12, delta0_plot / alpha, sigplotT, shading='auto')
plt.title("Temporal (Log Scale)")
plt.xlabel("Time (ps)")
plt.ylabel(r"$\Delta_0$")
plt.colorbar()
plt.show()

print(f"Execution time: {time.time() - start_time:.2f} seconds")
