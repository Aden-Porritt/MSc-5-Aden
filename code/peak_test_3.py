import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.optimize import brentq


def ring_out_put(x, I, c, a, fsr, x0):
    x = (x + x0) / fsr
    p = np.sqrt(1 - c) * np.sqrt(1 - a)
    return I * (1 - c) + (I * c ** 2 - 2 * I * c * np.sqrt(1 - c) * (1 - p * np.cos(2 * np.pi * x))) / (1 - 2 * p * np.cos(2 * np.pi * x) + p ** 2)


x = np.linspace(0, 1, 10000)

y = ring_out_put(x, 1, 0.005, 0.001, 50, -0.5)
y += ring_out_put(x, 0.2, 0.005, 0.001, 50, -0.25)
y += ring_out_put(x, 0.2, 0.005, 0.001, 50, -0.75)

# noise
y += np.random.normal(0, 0.01, size=y.shape)


# plt.plot(x, y)
# plt.show()

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

freq_peaks = 100 * 10 ** 9

df = freq_peaks / d_peak
print("df", df)

x = np.arange(len(y)) * df

plt.plot(x, y)
plt.show()



    

