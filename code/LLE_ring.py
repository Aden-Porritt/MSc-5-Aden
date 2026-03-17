import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift
import time

start_time = time.time()
np.random.seed(42)  # Fixed seed for reproducibility

# ============================================================
# Ring Resonator LLE Parameters
# ============================================================
# For a ring resonator (as opposed to FP):
#   - Light travels in ONE direction only → L_eff = L (not 2*L)
#   - Round-trip length = circumference = L
#   - Coupling and loss defined per round trip
# ============================================================

diameter = 400e-6                      # Ring circumference (m)
c = 3e8                      # Speed of light (m/s)
n0 = 2                    # Group refractive index
finesse = 1440

# Mirror/coupler parameters
T1 = 0.01                    # Input coupler power transmission
T2 = 0.01                    # Output coupler power transmission (if present)
# For a ring resonator, round-trip loss coefficient:
#   alpha = total power lost per round trip / 2  (amplitude loss)
# Includes coupling loss + any internal propagation loss
loss = 0.001
alpha = np.pi / finesse     # Effective round-trip amplitude loss

# Input coupling amplitude coefficient
theta = alpha                   # Power coupled in from input waveguide

# For a ring: effective propagation length per round trip = L (not 2*L)
L_eff = np.pi * diameter                   # ← KEY CHANGE from FP (was 2*L)

# Free spectral range for ring cavity
FSR = c / (n0 * L_eff)           # ← KEY CHANGE: FSR = c/(n0*L) for ring (not c/(2*n0*L))
t_R = 1 / FSR                # Round-trip time

print(f"FSR: {FSR/1e9:.2f} GHz")
print(f"Round-trip time t_R: {t_R*1e9:.4f} ns")

# ============================================================
# Nonlinear / Dispersion Parameters
# ============================================================
Pin = 0.3                      # Input pump power (W)
t_pulse = 4e-12           # Pump pulse duration (s) — effectively CW for large t_pulse
gamma = 1.4                # Nonlinear coefficient (1/W/m)

# Group velocity dispersion coefficients
beta2 = 392e-27             # (s^2/m) — anomalous dispersion
beta3 = 0.0                  # (s^3/m)
beta4 = 0.0                  # (s^4/m)

lambda_pump = 1550e-9        # Pump wavelength (m)

# ============================================================
# Temporal / Spectral Grid
# ============================================================
nt = 2**14                   # Number of time/frequency points

# Temporal window: must span at least one round trip with sufficient resolution.
# For a ring, a natural choice is a window that covers many cavity modes.
# Here we use a physically motivated window: ~10x the pump pulse duration,
# but capped to be at least a few times t_R.
Tmax = t_R * 10
dT = Tmax / nt

T = np.linspace(-nt / 2, nt / 2 - 1, nt) * dT   # Symmetric time axis

# Frequency axis (NOT fftshifted — raw FFT ordering for use with fft/ifft)
f = np.fft.fftfreq(nt, d=dT)                      # Frequency grid (Hz)
omega = 2 * np.pi * f                              # Angular frequency grid

print(f"Temporal window: {Tmax*1e12:.2f} ps")
print(f"Frequency resolution: {(f[1]-f[0])/1e9:.4f} GHz")

# ============================================================
# Pump Field (temporal envelope)
# ============================================================
# Sech-shaped pump; for large t_pulse this approximates CW
At_pump = np.sqrt(Pin) * (1.0 / np.cosh(T / t_pulse))


# ============================================================
# Simulation Parameters
# ============================================================
nloops = int(5e6)            # Total number of round trips
nplot = int(1e2)             # Number of snapshots to save

n_start = 0
n_step = 1             # Round-trip step size (fractional — 1/20 of a round trip)
n_stop = nloops
num_steps = int((n_stop - n_start) / n_step)

t_slow_start = n_start * t_R
t_slow_step  = n_step  * t_R
t_slow_stop  = n_stop  * t_R

# Detuning scan: from below resonance to above
delta0_start = -10 * alpha
delta0_stop  =  20 * alpha

# ============================================================
# Dispersive Operator
# ============================================================
# Phase accumulated per (fractional) round trip due to dispersion.
# Uses L_eff = L for ring (single-pass per round trip).
# The factor (t_slow_step / t_R) normalises to a fractional round trip.
dispersive_op = np.exp(
    (
        1j / 2  * beta2 * omega**2 +
        1j / 6  * beta3 * omega**3 +
        1j / 24 * beta4 * omega**4
    ) * (t_slow_step / t_R) * L_eff
)

# ============================================================
# Initial Intracavity Field
# ============================================================
# Start with vacuum (zero field) + tiny noise to seed MI
Anl = np.zeros(nt, dtype=complex)

# ============================================================
# Storage Arrays
# ============================================================
co = 0
co_plot = 0

Aout_t      = []
Aout_v      = []
t_slow_plot = []
delta0_plot = []


@nb.jit()
def fast_hepler(theta, t_R, At_pump, dispersive_op, Anl, alpha, gamma, L_eff, delta0, t_slow_step):
    noise = 1.0 + 1e-6 * np.random.randn(*T.shape)
    S = np.sqrt(theta) / t_R * At_pump * noise

    # --- Step 1: Dispersive propagation (frequency domain) ---
    Ad = fft(dispersive_op * ifft(Anl))

    # --- Step 2: Nonlinear + loss + detuning (time domain) ---
    # LLE round-trip operator:
    #   K = (1/t_R) * [-alpha + i*gamma*L_eff*|A|² - i*delta0]
    # Exact integration of dA/dt_slow = K*A + S gives:
    #   A(t+dt) = (A + S/K)*exp(K*dt) - S/K
    # This is valid when K ≠ 0 (always true here since alpha > 0)
    K = (1.0 / t_R) * (-alpha + 1j * gamma * L_eff * np.abs(Ad) ** 2 - 1j * delta0)
    Anl = (Ad + S / K) * np.exp(K * t_slow_step) - S / K
    return Anl

for tslow in np.arange(t_slow_start, t_slow_stop, t_slow_step):
    co += 1

    # Current detuning (linear ramp)
    delta0 = delta0_start + (delta0_stop - delta0_start) * tslow / t_slow_stop


    Anl = fast_hepler(theta, t_R, At_pump, dispersive_op, Anl, alpha, gamma, L_eff, delta0, t_slow_step)



    # Pump injection term (with small quantum noise seed)
    

        # noise = 1.0 + 1e-6 * np.random.randn(*T.shape)
        # S = np.sqrt(theta) / t_R * At_pump * noise

        # # --- Step 1: Dispersive propagation (frequency domain) ---
        # Ad = fft(dispersive_op * ifft(Anl))

        # # --- Step 2: Nonlinear + loss + detuning (time domain) ---
        # # LLE round-trip operator:
        # #   K = (1/t_R) * [-alpha + i*gamma*L_eff*|A|² - i*delta0]
        # # Exact integration of dA/dt_slow = K*A + S gives:
        # #   A(t+dt) = (A + S/K)*exp(K*dt) - S/K
        # # This is valid when K ≠ 0 (always true here since alpha > 0)
        # K = (1.0 / t_R) * (-alpha + 1j * gamma * L_eff * np.abs(Ad) ** 2 - 1j * delta0)
        # Anl = (Ad + S / K) * np.exp(K * t_slow_step) - S / K

    # --------------------------------------------------------
    # Save snapshot for plotting
    # --------------------------------------------------------
    if (co / num_steps * nplot) > co_plot + 1:
        co_plot += 1

        Aout = Anl.copy()
        Aout_t.append(Aout)
        Aout_v.append(fftshift(ifft(Aout)))
        t_slow_plot.append(tslow)
        delta0_plot.append(delta0)

        print(f"Step {co_plot}/{nplot} | delta0/alpha = {delta0/alpha:.3f}")

        # Live preview
        plt.figure(99)
        plt.clf()

        plt.subplot(2, 1, 1)
        plt.plot(fftshift(f) / 1e12, 10 * np.log10(np.abs(fftshift(ifft(Aout)))**2 + 1e-30))
        plt.xlabel("Frequency (THz)")
        plt.ylabel("Power (dB)")
        plt.title(f"Spectrum — delta0/alpha = {delta0/alpha:.3f}")

        plt.subplot(2, 1, 2)
        plt.plot(T * 1e12, np.abs(Aout)**2 + 1e-30)
        plt.xlabel("Time (ps)")
        plt.ylabel("Power (W)")
        plt.title("Temporal profile")

        plt.tight_layout()
        plt.pause(0.01)

plt.close(99)

# ============================================================
# Post-Processing
# ============================================================
Aout_t      = np.array(Aout_t)
Aout_v      = np.array(Aout_v)
t_slow_plot = np.array(t_slow_plot)
delta0_plot = np.array(delta0_plot)
fplot       = fftshift(f)

print("\nSimulation complete. Generating summary plots...")

# --- Figure 11: Spectral map (wide window) ---
plt.figure(11, figsize=(8, 5))
vmax = 20e12
vmax_ind = np.argmin(np.abs(fplot - vmax))
vmin_ind = np.argmin(np.abs(fplot + vmax))
sigplot = 10 * np.log10(np.abs(Aout_v[:, vmin_ind:vmax_ind])**2 + 1e-30)
plt.pcolormesh(fplot[vmin_ind:vmax_ind] / 1e12, delta0_plot / alpha, sigplot, shading='auto')
plt.title("Spectral Evolution (Ring Resonator LLE)")
plt.xlabel("Frequency (THz)")
plt.ylabel(r"$\Delta_0 / \alpha$")
plt.colorbar(label="Power (dB)")
plt.tight_layout()
plt.show()

# --- Figure 12: Spectral map (narrow window, normalised) ---
plt.figure(12, figsize=(8, 5))
vmax = 6e12
vmax_ind = np.argmin(np.abs(fplot - vmax))
vmin_ind = np.argmin(np.abs(fplot + vmax))
sigplot = 10 * np.log10(np.abs(Aout_v[:, vmin_ind:vmax_ind])**2 + 1e-30)
sigplot = sigplot - np.max(sigplot)
sigplot = np.clip(sigplot, -40, 0)
plt.pcolormesh(fplot[vmin_ind:vmax_ind] / 1e12, delta0_plot / alpha, sigplot, shading='auto')
plt.title("Spectral Evolution — Normalised (Ring Resonator LLE)")
plt.xlabel("Frequency (THz)")
plt.ylabel(r"$\Delta_0 / \alpha$")
plt.colorbar(label="Relative Power (dB)")
plt.tight_layout()
plt.show()

# --- Figure 13: Temporal map (linear) ---
plt.figure(13, figsize=(8, 5))
# Use full temporal grid, but only the central portion
tmax_plot_ind = nt - 1
tmin_plot_ind = 0
sigplot = np.abs(Aout_t[:, tmin_plot_ind:tmax_plot_ind])**2
plt.pcolormesh(T[tmin_plot_ind:tmax_plot_ind] / 1e-12, delta0_plot / alpha, sigplot, shading='auto')
plt.title("Temporal Evolution (Ring Resonator LLE)")
plt.xlabel("Time (ps)")
plt.ylabel(r"$\Delta_0 / \alpha$")
plt.colorbar(label="Power (W)")
plt.tight_layout()
plt.show()

# --- Figure 14: Temporal map (log scale) ---
plt.figure(14, figsize=(8, 5))
sigplotT = 10 * np.log10(np.abs(Aout_t[:, tmin_plot_ind:tmax_plot_ind])**2 + 1e-30)
sigplotT = sigplotT - np.max(sigplotT)
sigplotT = np.clip(sigplotT, -60, 0)
plt.pcolormesh(T[tmin_plot_ind:tmax_plot_ind] / 1e-12, delta0_plot / alpha, sigplotT, shading='auto')
plt.title("Temporal Evolution — Log Scale (Ring Resonator LLE)")
plt.xlabel("Time (ps)")
plt.ylabel(r"$\Delta_0 / \alpha$")
plt.colorbar(label="Relative Power (dB)")
plt.tight_layout()
plt.show()

plt.show()

print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")