import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.optimize import brentq
import oscilloscope as osc


def read_file(file_name):

    data = np.loadtxt(file_name, skiprows=2, delimiter=',')
    
    time = data[:, 0]
    voltage = data[:, 1]
    
    return time, voltage

def ring_out_put(x, I, c, a, fsr, x0):
    x = (x + x0) / fsr
    p = np.sqrt(1 - c) * np.sqrt(1 - a)
    return I * (1 - c) + (I * c ** 2 - 2 * I * c * np.sqrt(1 - c) * (1 - p * np.cos(2 * np.pi * x))) / (1 - 2 * p * np.cos(2 * np.pi * x) + p ** 2)

def finesse(c, a):
    p = np.sqrt(1 - c) * np.sqrt(1 - a)
    return np.pi * p / (1 - p)

# time, data = read_file("C:/Users/adenp/Desktop/MSc 5 Aden/code/3_peak_test.csv")
time, data = osc.read_waveform(osc.connect(), 2)


plt.plot(time, data)
plt.show()

x = np.linspace(0, 1, len(data))

# y = ring_out_put(x, 1, 0.005, 0.001, 50, -0.5)
# y += ring_out_put(x, 0.2, 0.005, 0.001, 50, -0.25)
# y += ring_out_put(x, 0.2, 0.005, 0.001, 50, -0.75)

# # noise
# y += np.random.normal(0, 0.01, size=y.shape)


# plt.plot(x, y)
# plt.show()

y = data

def find_peaks(data):
    data = np.array(data)
    avg_ = np.average(data)
    peaks_index = []
    for _ in range(3):
        peaks_index.append(np.argmin(data))
        peak = peaks_index[-1]
        for i in range(round(len(data) / 10)):
            data[peak + i] = avg_
            data[peak - i] = avg_

    return sorted(peaks_index)

peaks_index = find_peaks(y)

plt.plot(x, y)
plt.scatter([x[i] for i in peaks_index], [y[i] for i in peaks_index], c="red")
plt.show()

print(peaks_index)
d_peak = peaks_index[2] - peaks_index[0]
print(d_peak)

freq_peaks = 5 * 10 ** 9 * 2

df = freq_peaks / d_peak
print("df", df)

x = np.arange(len(y)) * df

def rind_fsr_fixed(x, I, c, a, x0):
    fsr = 3.75 * 10 ** 12
    return ring_out_put(x, I, c, a, fsr, x0)

bounds = (
    [ 0,      0,    0,    -x[peaks_index[1]] * 1.01],
    [10,   0.99, 0.99, -x[peaks_index[1]] * 0.99   ]
    )

popt, pcov = curve_fit(rind_fsr_fixed, x, y, p0=[1, 0.005, 0.001, -x[peaks_index[1]]], bounds=bounds, maxfev=10000)
print(popt)
print("finesse", finesse(popt[1], popt[2]))

plt.plot(x, y)
plt.plot(x, rind_fsr_fixed(x, *popt), c="red")
plt.scatter([x[i] for i in peaks_index], [y[i] for i in peaks_index], c="red")
plt.show()



    

