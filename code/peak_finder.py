import numpy as np
import matplotlib.pyplot as plt

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

    cut_off = (avg_data + min_data) * 0.333 + avg_data
    print(cut_off)

    i = 100
    peaks = []
    while i < len(data) and len(peaks) < 2:
        if data[i] < cut_off:
            peak_index = np.argmin(data[i:i+100])
            peaks.append(i + peak_index)
            i += 100

        i += 1

    return peaks

def find_half(data):
    avg_data = np.average(data)
    min_data = np.min(data)

    half = ((avg_data + min_data) / 2) + avg_data
    print(half)

    half_index = [-1, -1]

    for i in range(len(data)):
        if data[i] < half:
            if half_index[0] == -1:
                half_index[0] = i
        if data[len(data) - 1 - i] < half:
            if half_index[1] == -1:
                half_index[1] = len(data) - 1 - i
    
    return half_index

if False:
    test_data = np.zeros(100000)
    FSR = 0.5

    start = 0.25

    F = 10000
    for i in range(len(test_data)):
        test_data[i] = -0.05 / (1 + F * np.sin(np.pi * (i / (len(test_data)) + start) / FSR) ** 2)

    test_data += np.random.rand(len(test_data)) / 100

    for i in range(len(test_data)):
        test_data[i] += -0.4 / (1 + 2 * np.sin(np.pi * (i / (len(test_data)) + start) / 0.1823973) ** 2)


test_data = read_file("C:/Users/adenp/Desktop/5/code/test12.csv")[1]

plt.plot(test_data)
plt.show()

test_data = high_pass_filter(test_data, 200, len(test_data))

plt.plot(test_data)
plt.show()

peak_indexs = find_peaks(test_data)

# print(peak_indexs)

plt.plot(test_data)
plt.scatter(peak_indexs, [0 for i in range(len(peak_indexs))], c="red")
plt.show()

peak_data = test_data[peak_indexs[0] - 100: peak_indexs[0] + 100]

half_indexs = find_half(peak_data)
print(half_indexs)
print(half_indexs[1] - half_indexs[0])

# print("test finesse", np.pi * np.sqrt(F) / 2)
print("finesse", (peak_indexs[1] - peak_indexs[0]) / (half_indexs[1] - half_indexs[0]))

plt.plot(test_data[peak_indexs[0] - 100: peak_indexs[0] + 100])
plt.scatter(half_indexs, [-0.038 for i in range(len(half_indexs))], c="red")
plt.show()






