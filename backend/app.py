import time
import serial
import threading
import os
import math
import csv
import datetime
import numpy as np
from scipy.fft import rfft, rfftfreq
from flask import Flask, jsonify, render_template, request
from ml.pipeline import ml_pipeline

# Create dataset directory if it doesn't exist
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)

# --- Configuration ---
SERIAL_PORT = 'COM9'
BAUD_RATE = 115200

app = Flask(__name__)

# --- Thread-Safe Data Storage ---
sensor_data = {
    'pitch': 0.0, 'roll': 0.0, 'yaw': 0.0,
    'raw_ax': 0, 'raw_ay': 0, 'raw_az': 0,
    'raw_gx': 0, 'raw_gy': 0, 'raw_gz': 0,
    'ax_g': 0.0, 'ay_g': 0.0, 'az_g': 0.0,
    'gx_dps': 0.0, 'gy_dps': 0.0, 'gz_dps': 0.0,
    'dsp_freq': 0.0, 'dsp_amp': 0.0, 'dsp_axis': 'None',
    'needs_label': False
}
data_lock = threading.Lock()

# --- Active Learning Labels ---
pending_label_window = {'x': [], 'y': [], 'z': []}

# --- DSP Buffers ---
BUFFER_SIZE = 256
SAMPLING_RATE = 50.0 # 50 Hz
buffer_x = []
buffer_y = []
buffer_z = []

# --- Temporal Consistency Filter Variables ---
last_dom_freq = 0.0
freq_streak_count = 0

# --- Complementary Filter Variables ---
# dt is roughly 0.02s since we read at 50Hz
dt = 0.02  
pitch = 0.0
roll = 0.0
yaw = 0.0

# Accelerometer sensitivity (16384 LSB/g for +/- 2g)
# Gyroscope sensitivity (131 LSB/deg/s for +/- 250 deg/s)
ACCEL_SENSITIVITY = 16384.0
GYRO_SENSITIVITY = 131.0

def serial_reader():
    global sensor_data, pitch, roll, yaw
    
    # We loop indefinitely in case the serial port disconnects and reconnects
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"Successfully connected to {SERIAL_PORT}.")
            
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Ignore initial log messages from the ESP32
                    if "Initialized" in line or not line:
                        continue
                    
                    try:
                        # Expected format: ax,ay,az,gx,gy,gz
                        values = [int(v) for v in line.split(',')]
                        
                        if len(values) == 6:
                            raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz = values
                            
                            # Convert raw values to physical units
                            # Accel to 'g'
                            ax_g = raw_ax / ACCEL_SENSITIVITY
                            ay_g = raw_ay / ACCEL_SENSITIVITY
                            az_g = raw_az / ACCEL_SENSITIVITY
                            
                            # Gyro to deg/s
                            gx_dps = raw_gx / GYRO_SENSITIVITY
                            gy_dps = raw_gy / GYRO_SENSITIVITY
                            gz_dps = raw_gz / GYRO_SENSITIVITY

                            # Calculate pitch and roll from accelerometer
                            # Using math.atan2 for full quadrant coverage
                            accel_pitch = -math.degrees(math.atan2(ay_g, math.sqrt(ax_g**2 + az_g**2)))
                            accel_roll = math.degrees(math.atan2(-ax_g, az_g))

                            # Apply Complementary Filter
                            # 96% trust in gyro, 4% trust in accelerometer for drift correction
                            alpha = 0.96
                            pitch = alpha * (pitch + gx_dps * dt) + (1.0 - alpha) * accel_pitch
                            roll = alpha * (roll + gy_dps * dt) + (1.0 - alpha) * accel_roll
                            
                            # Yaw estimation from gyro only (will drift over time without magnetometer)
                            yaw = yaw + gz_dps * dt

                            # Update DSP buffers
                            if len(buffer_x) >= BUFFER_SIZE:
                                buffer_x.pop(0)
                                buffer_y.pop(0)
                                buffer_z.pop(0)
                            buffer_x.append(ax_g)
                            buffer_y.append(ay_g)
                            buffer_z.append(az_g)

                            with data_lock:
                                sensor_data['pitch'] = pitch
                                sensor_data['roll'] = roll
                                sensor_data['yaw'] = yaw
                                sensor_data['raw_ax'] = raw_ax
                                sensor_data['raw_ay'] = raw_ay
                                sensor_data['raw_az'] = raw_az
                                sensor_data['raw_gx'] = raw_gx
                                sensor_data['raw_gy'] = raw_gy
                                sensor_data['raw_gz'] = raw_gz
                                sensor_data['ax_g'] = ax_g
                                sensor_data['ay_g'] = ay_g
                                sensor_data['az_g'] = az_g
                                sensor_data['gx_dps'] = gx_dps
                                sensor_data['gy_dps'] = gy_dps
                                sensor_data['gz_dps'] = gz_dps

                            # Feed into ML pipeline (non-blocking)
                            ml_pipeline.feed(ax_g, ay_g, az_g, gx_dps, gy_dps, gz_dps)

                    except (ValueError, IndexError):
                        pass # Ignore malformed packets quietly to avoid flooding the console
                        
        except serial.SerialException as e:
            print(f"Waiting for {SERIAL_PORT}... is the ESP32 closed in the Arduino IDE? Retrying in 3s.")
            time.sleep(3)

def dsp_worker():
    """Runs FFT on the sliding window every 0.5s to find dominant tremor frequencies."""
    global sensor_data, buffer_x, buffer_y, buffer_z
    
    while True:
        time.sleep(0.5)
        
        # We need a full buffer to do meaningful FFT
        if len(buffer_x) < BUFFER_SIZE:
            continue
            
        # Make a copy of the lists to avoid mutation during FFT
        x_data = np.array(buffer_x)
        y_data = np.array(buffer_y)
        z_data = np.array(buffer_z)
        
        # Remove DC Offset (mean) before FFT to prevent a massive spike at 0 Hz
        x_data = x_data - np.mean(x_data)
        y_data = y_data - np.mean(y_data)
        z_data = z_data - np.mean(z_data)

        # Frequencies corresponding to the FFT bins
        freqs = rfftfreq(BUFFER_SIZE, d=1.0/SAMPLING_RATE)
        
        # We only care about the typical tremor band (3 Hz to 12 Hz)
        valid_idx = np.where((freqs >= 3.0) & (freqs <= 12.0))[0]
        
        if len(valid_idx) == 0:
            continue
            
        # Calculate FFT magnitudes for each axis
        fft_x = np.abs(rfft(x_data))
        fft_y = np.abs(rfft(y_data))
        fft_z = np.abs(rfft(z_data))
        
        # Search for peaks strictly within the 3-12 Hz band
        band_freqs = freqs[valid_idx]
        band_x = fft_x[valid_idx]
        band_y = fft_y[valid_idx]
        band_z = fft_z[valid_idx]
        
        highest_amp = 0
        dom_freq = 0
        dom_axis = 'None'
        
        max_x_idx = np.argmax(band_x)
        if band_x[max_x_idx] > highest_amp:
            highest_amp = band_x[max_x_idx]
            dom_freq = band_freqs[max_x_idx]
            dom_axis = 'X'
            
        max_y_idx = np.argmax(band_y)
        if band_y[max_y_idx] > highest_amp:
            highest_amp = band_y[max_y_idx]
            dom_freq = band_freqs[max_y_idx]
            dom_axis = 'Y'
            
        max_z_idx = np.argmax(band_z)
        if band_z[max_z_idx] > highest_amp:
            highest_amp = band_z[max_z_idx]
            dom_freq = band_freqs[max_z_idx]
            dom_axis = 'Z'
            
        # Threshold: Only register it as a tremor if the amplitude crosses a noise floor.
        global last_dom_freq, freq_streak_count
        
        # User confirmed observation: An impulse stays mathematically stable in the FFT for the ENTIRE duration of the window (~5.1s).
        # To filter out a single bump, the streak MUST be longer than the sliding window.
        # 5.12s sliding window / 0.5s cycle = ~10 cycles. 
        # By setting it to 11, we ensure the vibration outlasts the window.
        STREAK_THRESHOLD = 11 
        
        # NEW FIX: The 0.5s Instantaneous Energy Gate
        # We only check the very last 0.5 seconds (25 samples at 50Hz) of the buffer.
        # If the peak-to-peak difference in this tiny recent chunk is physically near zero, the patient is currently sitting still.
        # This instantly mutes the "ghost trail" of the FFT.
        activity_window = 25
        recent_x = x_data[-activity_window:]
        recent_y = y_data[-activity_window:]
        recent_z = z_data[-activity_window:]
        
        # Peak-to-peak check (Max - Min). A threshold of 0.1g effectively filters out heartbeat/breathing but catches tremors.
        is_moving_now = False
        if dom_axis == 'X':
            is_moving_now = (np.max(recent_x) - np.min(recent_x)) > 0.1
        elif dom_axis == 'Y':
            is_moving_now = (np.max(recent_y) - np.min(recent_y)) > 0.1
        elif dom_axis == 'Z':
            is_moving_now = (np.max(recent_z) - np.min(recent_z)) > 0.1

        if highest_amp > 1.5 and is_moving_now: 
            # Consistency filter
            if abs(dom_freq - last_dom_freq) <= 0.6: # Allow 0.6 Hz fluctuation
                freq_streak_count += 1
            else:
                freq_streak_count = 1 # Reset streak with the new dominant frequency
                
            last_dom_freq = dom_freq
            
            if freq_streak_count >= STREAK_THRESHOLD:
                with data_lock:
                    sensor_data['dsp_freq'] = dom_freq
                    sensor_data['dsp_amp'] = highest_amp / (BUFFER_SIZE/2) # Normalize FFT amplitude
                    sensor_data['dsp_axis'] = 'Accel ' + dom_axis # Explicitly state data source
                    
                    # TRIGGER ACTIVE LEARNING: We detected a confirmed uniform periodic signal
                    # If we haven't already asked the user for this specific event, trigger the UI
                    if not sensor_data['needs_label'] and freq_streak_count == STREAK_THRESHOLD:
                        sensor_data['needs_label'] = True
                        global pending_label_window
                        pending_label_window['x'] = list(x_data)
                        pending_label_window['y'] = list(y_data)
                        pending_label_window['z'] = list(z_data)
                        print("TRIGGERED ACTIVE LEARNING NOTIFICATION")
            else:
                # Still stabilizing
                with data_lock:
                    sensor_data['dsp_freq'] = 0.0
                    sensor_data['dsp_amp'] = 0.0
                    sensor_data['dsp_axis'] = 'None'
        else:
            freq_streak_count = 0
            last_dom_freq = 0.0
            with data_lock:
                sensor_data['dsp_freq'] = 0.0
                sensor_data['dsp_amp'] = 0.0
                sensor_data['dsp_axis'] = 'None'

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/log_data', methods=['POST'])
def log_data():
    global pending_label_window
    data = request.json
    label     = data.get('label', 'Unknown')
    window_id = data.get('window_id')   # present only for ML-triggered requests

    # --- ML pipeline label routing ---
    if window_id:
        ml_label = label if label in ('normal', 'tremor') else None
        if ml_label is None:
            return jsonify({"status": "error", "message": f"Unknown label '{label}'"})
        result = ml_pipeline.submit_label(window_id, ml_label)
        with data_lock:
            sensor_data['needs_label'] = False
        return jsonify(result)

    # --- Original DSP-based label routing (unchanged) ---
    with data_lock:
        sensor_data['needs_label'] = False # Reset the UI flag

    if not pending_label_window['x']:
        return jsonify({"status": "error", "message": "No pending data"})

    # Save the window to a CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.csv"
    filepath = os.path.join(DATASET_DIR, filename)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ax_g', 'ay_g', 'az_g'])
        for i in range(len(pending_label_window['x'])):
            writer.writerow([
                pending_label_window['x'][i],
                pending_label_window['y'][i],
                pending_label_window['z'][i]
            ])

    # Clear the pending window
    pending_label_window = {'x': [], 'y': [], 'z': []}
    print(f"Data successfully saved to {filepath}")
    return jsonify({"status": "success", "message": f"Saved {filename}"})

@app.route('/data')
def get_data():
    with data_lock:
        response = dict(sensor_data)
    # Merge ML predictions — always present, null when not yet active
    response.update(ml_pipeline.latest_prediction())
    return jsonify(response)


@app.route('/ml_status')
def ml_status():
    """Debug endpoint — returns full ML pipeline status."""
    return jsonify(ml_pipeline.status())

if __name__ == '__main__':
    # Start the serial reading loop in a daemon thread
    thread = threading.Thread(target=serial_reader, daemon=True)
    thread.start()

    # Start the FFT DSP worker thread
    dsp_thread = threading.Thread(target=dsp_worker, daemon=True)
    dsp_thread.start()

    ml_pipeline.start()

    # Run the web server
    app.run(host='0.0.0.0', port=5000, debug=False)
