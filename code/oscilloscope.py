import pyvisa
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import time

# ── Connection ──────────────────────────────────────────────────────────────
def connect_lan(usb_address=None):
    """
    Connect to Keysight oscilloscope over USB.
    
    Parameters:
        usb_address : optional VISA resource string e.g. 'USB0::0x0957::0x1796::MY12345678::INSTR'
                      if None, will auto-detect the first USB instrument found
    
    Returns:
        scope : pyvisa resource object
    """
    rm = pyvisa.ResourceManager()

    if usb_address is None:
        # Auto-detect first USB instrument
        resources = rm.list_resources('USB?*')
        if len(resources) == 0:
            raise RuntimeError("No USB instruments found")
        usb_address = resources[0]
        print(f"Auto-detected: {usb_address}")

    scope = rm.open_resource(usb_address)
    scope.timeout = 10000
    scope.write_termination = '\n'
    scope.read_termination = '\n'

    idn = scope.query('*IDN?')
    print(f"Connected to: {idn}")

    return scope

def connect(ip_address):
    """
    Connect to Keysight oscilloscope over LAN.
    
    Parameters:
        ip_address : string e.g. '192.168.1.100'
    
    Returns:
        scope : pyvisa resource object
    """
    rm = pyvisa.ResourceManager()
    scope = rm.open_resource(f'TCPIP0::{ip_address}::inst0::INSTR')
    scope.timeout = 10000  # 10 second timeout
    scope.write_termination = '\n'
    scope.read_termination = '\n'

    idn = scope.query('*IDN?')
    print(f"Connected to: {idn}")

    return scope

# ── Oscilloscope Setup ───────────────────────────────────────────────────────

def setup(scope, channel=1, time_scale=1e-3, voltage_scale=0.1):
    """
    Basic oscilloscope setup.
    
    Parameters:
        scope         : pyvisa resource object
        channel       : channel number to read (1-4)
        time_scale    : time per division in seconds
        voltage_scale : voltage per division in volts
    """
    scope.write('*RST')                                 # reset to default
    time.sleep(2)                                       # wait for reset

    scope.write(f':CHANnel{channel}:DISPlay ON')        # turn channel on
    scope.write(f':CHANnel{channel}:SCALe {voltage_scale}')  # volts/div
    scope.write(f':TIMebase:SCALe {time_scale}')        # time/div
    scope.write(f':TIMebase:MODE MAIN')
    scope.write(':ACQuire:TYPE NORMal')
    scope.write(':ACQuire:COMPlete 100')                # 100% complete acquisition

    print(f"Setup complete — CH{channel}, {voltage_scale} V/div, {time_scale} s/div")

# ── Data Acquisition ─────────────────────────────────────────────────────────

def read_waveform(scope, channel=1):
    """
    Read waveform data from oscilloscope.
    
    Parameters:
        scope   : pyvisa resource object
        channel : channel number to read (1-4)
    
    Returns:
        time    : time array in seconds
        voltage : voltage array in volts
    """
    # Tell scope which channel to transfer
    scope.write(f':WAVeform:SOURce CHANnel{channel}')
    scope.write(':WAVeform:FORMat ASCii')       # ASCII for simplicity
    scope.write(':WAVeform:POINts:MODE RAW')
    # scope.write(':WAVeform:POINts 1000')        # number of points
    scope.write(':WAVeform:POINts:MODE NORMal')
    max_points = int(scope.query(':WAVeform:POINts:MAXimum?'))

    # Get scaling factors from scope
    x_increment = float(scope.query(':WAVeform:XINCrement?'))
    x_origin    = float(scope.query(':WAVeform:XORigin?'))
    y_increment = float(scope.query(':WAVeform:YINCrement?'))
    y_origin    = float(scope.query(':WAVeform:YORigin?'))
    y_reference = float(scope.query(':WAVeform:YREFerence?'))

    # Trigger single acquisition and wait
    # scope.write(':SINGle')
    # scope.query('*OPC?')    # waits until operation complete

    scope.write(':STOP')        # make sure scope is not still acquiring
    scope.query('*OPC?')        # wait until it has fully stopped

    # Pull data
    raw = scope.query(':WAVeform:DATA?')

    # Parse ASCII data
    raw = raw.strip()
    if raw.startswith('#'):
        # Strip IEEE header if present
        n_digits = int(raw[1])
        raw = raw[2 + n_digits:]

    values = np.array([float(v) for v in raw.split(',')])

    # Apply scaling
    voltage = (values - y_reference) * y_increment + y_origin
    time_arr = x_origin + np.arange(len(voltage)) * x_increment

    print(f"Read {len(voltage)} points from CH{channel}")

    return time_arr, voltage

# ── Save Data ────────────────────────────────────────────────────────────────

def save_data(time_arr, voltage, filename=None):
    """
    Save waveform to CSV, matching your existing read_file() format.
    
    Parameters:
        time_arr : time array in seconds
        voltage  : voltage array in volts
        filename : optional filename, auto-generated if None
    """
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'waveform_{timestamp}.csv'

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Keysight oscilloscope data'])
        writer.writerow(['time (s)', 'voltage (V)'])
        for t, v in zip(time_arr, voltage):
            writer.writerow([t, v])

    print(f"Saved to {filename}")
    return filename

# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_waveform(time_arr, voltage, filename=''):
    plt.figure()
    plt.plot(time_arr, voltage)
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Waveform — {filename}')
    plt.tight_layout()
    plt.show()

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    IP_ADDRESS   = '192.168.1.100'  # ← change to your oscilloscope's IP
    CHANNEL      = 1
    TIME_SCALE   = 1e-3             # 1 ms/div
    VOLTAGE_SCALE = 0.1             # 100 mV/div

    scope = connect(IP_ADDRESS)
    # setup(scope, CHANNEL, TIME_SCALE, VOLTAGE_SCALE)

    time_arr, voltage = read_waveform(scope, CHANNEL)

    filename = save_data(time_arr, voltage)
    plot_waveform(time_arr, voltage, filename)

    scope.close()