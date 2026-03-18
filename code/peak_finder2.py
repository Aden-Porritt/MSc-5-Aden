import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.optimize import brentq

def read_file(file_name):

    data = np.loadtxt(file_name, skiprows=2, delimiter=',')
    
    time = data[:, 0]
    voltage = data[:, 1]
    
    return time, voltage

def high_pass_filter(signal, cutoff_hz, sample_rate):

    fft_data = np.fft.rfft(signal)
    
    freqs = np.fft.rfftfreq(len(signal), d=1/sample_rate)
    
    fft_data[freqs < cutoff_hz] = 0
    
    filtered_signal = np.fft.irfft(fft_data, n=len(signal))
    
    return filtered_signal

def find_peaks(data):
    avg_data = np.average(data)
    min_data = np.min(data)

    cut_off = (avg_data + min_data) * 0.33 + avg_data

    i = 100
    peaks = []
    while i < len(data) and len(peaks) < 2:
        if data[i] < cut_off:
            new_data = data[i:i+100]
            peak_index_1 = np.argmin(new_data)
            for ii in range(3):
                new_data[peak_index_1 + ii] = 0.0
                new_data[peak_index_1 - ii] = 0.0
            peak_index_2 = np.argmin(new_data)
            peaks.append(round((i + (peak_index_1 + peak_index_2) / 2)))
            i += 100

        i += 1

    return peaks

def get_peaks(data, peak_index):
    new_data = data[peak_index - 100:peak_index + 100]

    start_peak_end = [0, 0, 0]

    # find peak 


    last_min = new_data[100]

    for i in range(1, 20):
        if np.min(new_data[100 - i * 4:100 + i * 4]) == last_min:
            start_peak_end[1] = np.argmin(new_data[100 - i * 4:100 + i * 4]) + 100 - i * 4
            break
        last_min = np.min(new_data[100 - i * 4:100 + i * 4])

    peak = start_peak_end[1]

    last_max = new_data[peak]

    for i in range(1, 20):
        if np.max(new_data[peak - i * 4:peak]) == last_max:
            start_peak_end[0] = np.argmax(new_data[peak - i * 4:peak]) + peak - i * 4
            break
        last_max = np.max(new_data[peak - i * 4:peak])

    last_max = new_data[peak]

    for i in range(1, 20):
        if np.max(new_data[peak:peak + i * 4]) == last_max:
            start_peak_end[2] = np.argmax(new_data[peak:peak + i * 4]) + peak
            break
        last_max = np.max(new_data[peak:peak + i * 4])
        
    half_indexs = get_half(new_data, start_peak_end)

    return start_peak_end, half_indexs

def get_half(data, start_peak_end):
    start_peak_end = np.array(start_peak_end)
    half_index = [0, 0]
    start = start_peak_end[0]
    peak = start_peak_end[1]
    end = start_peak_end[2]
    half = (data[start] + data[peak]) / 2
    while start < peak:
        if (data[start - 1] + data[start] + data[start + 1]) / 3 < half:
            half_index[0] = start
            break
        start += 1
    half = (data[end] + data[peak]) / 2
    while end > peak:
        if (data[end - 1] + data[end] + data[end + 1]) / 3 < half:
            half_index[1] = end
            break
        end -= 1
    return half_index

def foo(x, I, c, a, fsr, x0):
    x = (x + x0) / fsr
    p = np.sqrt(1 - c) * np.sqrt(1 - a)
    return I * (1 - c) + (I * c ** 2 - 2 * I * c * np.sqrt(1 - c) * (1 - p * np.cos(2 * np.pi * x))) / (1 - 2 * p * np.cos(2 * np.pi * x) + p ** 2)

# time, data = read_file("C:/Users/adenp/Desktop/MSc 5 Aden/code/g_072_1_1.csv")
time, data = read_file("C:/Users/adenp/Desktop/MSc 5 Aden/code/g_082_1_1.csv")

print("data length", len(data))

plt.figure(1)
plt.plot([i / len(data) for i in range(len(data))], data)

data = high_pass_filter(data, 100, len(data))


g_data = np.gradient(data)
g_avg_data = []
n = 3
for i in range(n, len(g_data) - n):
    g_avg_data.append(np.average(g_data[i - n: i + n]))


peak_indexs = find_peaks(-np.abs(g_avg_data))

# plt.figure(4)
# plt.plot(np.abs(g_avg_data))
# plt.scatter(peak_indexs, [np.abs(g_avg_data)[i] for i in peak_indexs], c="red")



# plt.figure(6)
# plt.plot(data[peak_indexs[0] - 100:peak_indexs[0] + 100])
# 
# plt.figure(7)
# plt.plot(data[peak_indexs[1] - 100:peak_indexs[1] + 100])

start_peak_end, half_indexs = get_peaks(data, peak_indexs[0])


plt.figure(8)
plot_data = data[peak_indexs[0] - 100:peak_indexs[0] + 100]
plt.plot(plot_data)
# plt.scatter(start_peak_end, [plot_data[i] for i in start_peak_end], c="red")
# plt.scatter(half_indexs, [plot_data[i] for i in half_indexs], c="green")

fsr = (peak_indexs[1] - peak_indexs[0]) / (len(data)) / 2
df = (half_indexs[1] - half_indexs[0]) / (len(data)) / 2
print("F", fsr / df)
peak_indexs[0] += start_peak_end[1] - 100

start_peak_end, half_indexs = get_peaks(data, peak_indexs[1])

plt.figure(9)
plot_data = data[peak_indexs[1] - 100:peak_indexs[1] + 100]
plt.plot(plot_data)
# plt.scatter(start_peak_end, [plot_data[i] for i in start_peak_end], c="red")
# plt.scatter(half_indexs, [plot_data[i] for i in half_indexs], c="green")

df = (half_indexs[1] - half_indexs[0]) / (len(data)) / 2
print("F", fsr / df)
peak_indexs[1] += start_peak_end[1] - 100

plt.figure(5)
plt.plot([i / len(data) for i in range(len(data))], data)
plt.scatter(np.array(peak_indexs) / len(data), [data[i] for i in peak_indexs], c="red")


time, data = read_file("C:/Users/adenp/Desktop/MSc 5 Aden/code/g_082_1_4.csv")
# time, data = read_file("C:/Users/adenp/Desktop/MSc 5 Aden/code/g_2_72_3.csv")

x = np.linspace(0, 1, len(data))

plt.figure(11)
plt.plot(x, data)

avg_ = (np.average(data))

x = np.linspace(0, 1, len(data))
# for i in range(len(x)):
#     if x[i] < 0.4:
#         data[i] = avg_
#     if x[i] > 0.6:
#         data[i] = avg_

print(fsr, fsr * 25)

def foo_fixed_fsr(x, I, c, a, x0):
        return foo(x, I, c, a, fsr * 25, x0)

bounds = ([0,    0.001,    0.001,    np.argmin(data) / len(data) - 0.01], [1000,   0.01, 0.01,  np.argmin(data) / len(data) + 0.01])

popt, pcov = curve_fit(foo_fixed_fsr, x, data, p0=[np.max(data), 0.001,  0.001, -np.argmin(data) / len(data)])
print(popt)

plt.figure(13)
plt.plot(x, data)
plt.plot(x, foo_fixed_fsr(x, *popt))

print(1 - (foo_fixed_fsr(0, popt[0], popt[1], popt[2], 0)) / (foo_fixed_fsr(1.0, popt[0], popt[1], popt[2], 0)))
print(((foo_fixed_fsr(0, popt[0], popt[1], popt[2], 0) + foo_fixed_fsr(1.0, popt[0], popt[1], popt[2], 0)) / 2))


def my_function(x):
    return foo_fixed_fsr(x, popt[0], popt[1], popt[2], 0) - ((foo_fixed_fsr(0, popt[0], popt[1], popt[2], 0) + foo_fixed_fsr(1.0, popt[0], popt[1], popt[2], 0)) / 2)

print(my_function(0), my_function(1.0))

root = brentq(my_function, 0.0, 1.0)

print(root * 2)

print(25 * fsr / (root * 2))

plt.show()



