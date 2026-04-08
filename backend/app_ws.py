import os
import csv
import datetime
import time
import threading
import math
import numpy as np
from scipy.fft import rfft, rfftfreq
from flask import Flask, jsonify, render_template, request
import websocket  # websocket-client library
from ml.pipeline import ml_pipeline

# --- Configuration ---
# The ESP32 hotspot assigns itself 192.168.4.1 by default
ESP32_WS_URL = 'ws://192.168.4.1:81'

app = Flask(__name__)

# Create dataset directory if it doesn't exist
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)

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
dt = 0.02  
pitch = 0.0
roll = 0.0
yaw = 0.0

ACCEL_SENSITIVITY = 16384.0
GYRO_SENSITIVITY = 131.0


def process_sensor_line(line):
    """Processes a single CSV line of sensor data."""
    global sensor_data, pitch, roll, yaw

    try:
        values = [int(v) for v in line.split(',')]
        
        if len(values) == 6:
            raw_ax, raw_ay, raw_az, raw_gx, raw_gy, raw_gz = values
            
            ax_g = raw_ax / ACCEL_SENSITIVITY
            ay_g = raw_ay / ACCEL_SENSITIVITY
            az_g = raw_az / ACCEL_SENSITIVITY
            
            gx_dps = raw_gx / GYRO_SENSITIVITY
            gy_dps = raw_gy / GYRO_SENSITIVITY
            gz_dps = raw_gz / GYRO_SENSITIVITY

            accel_pitch = -math.degrees(math.atan2(ay_g, math.sqrt(ax_g**2 + az_g**2)))
            accel_roll = math.degrees(math.atan2(-ax_g, az_g))

            alpha = 0.96
            pitch = alpha * (pitch + gx_dps * dt) + (1.0 - alpha) * accel_pitch
            roll = alpha * (roll + gy_dps * dt) + (1.0 - alpha) * accel_roll
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
        pass


def websocket_reader():
    """Connects to the ESP32 WebSocket server and reads sensor data."""
    def on_message(ws_conn, message):
        line = message.strip()
        if "Initialized" in line or not line:
            return
        process_sensor_line(line)

    def on_error(ws_conn, error):
        print(f"WebSocket error: {error}")

    def on_close(ws_conn, close_status_code, close_msg):
        print("WebSocket connection closed. Reconnecting in 3s...")

    def on_open(ws_conn):
        print(f"Successfully connected to ESP32 at {ESP32_WS_URL}")

    while True:
        try:
            ws_conn = websocket.WebSocketApp(
                ESP32_WS_URL,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            ws_conn.run_forever()
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
        
        print(f"Waiting for ESP32 hotspot... Retrying in 3s.")
        time.sleep(3)


def dsp_worker():
    """Runs FFT on the sliding window every 0.5s to find dominant tremor frequencies."""
    global sensor_data, buffer_x, buffer_y, buffer_z
    
    while True:
        time.sleep(0.5)
        
        if len(buffer_x) < BUFFER_SIZE:
            continue
            
        x_data = np.array(buffer_x)
        y_data = np.array(buffer_y)
        z_data = np.array(buffer_z)
        
        x_data = x_data - np.mean(x_data)
        y_data = y_data - np.mean(y_data)
        z_data = z_data - np.mean(z_data)

        freqs = rfftfreq(BUFFER_SIZE, d=1.0/SAMPLING_RATE)
        valid_idx = np.where((freqs >= 4.0) & (freqs <= 12.0))[0]
        
        if len(valid_idx) == 0:
            continue
            
        fft_x = np.abs(rfft(x_data))
        fft_y = np.abs(rfft(y_data))
        fft_z = np.abs(rfft(z_data))
        
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
            
        global last_dom_freq, freq_streak_count
        STREAK_THRESHOLD = 11 
        
        activity_window = 25
        recent_x = x_data[-activity_window:]
        recent_y = y_data[-activity_window:]
        recent_z = z_data[-activity_window:]
        
        is_moving_now = False
        if dom_axis == 'X':
            is_moving_now = (np.max(recent_x) - np.min(recent_x)) > 0.1
        elif dom_axis == 'Y':
            is_moving_now = (np.max(recent_y) - np.min(recent_y)) > 0.1
        elif dom_axis == 'Z':
            is_moving_now = (np.max(recent_z) - np.min(recent_z)) > 0.1

        if highest_amp > 1.5 and is_moving_now: 
            if abs(dom_freq - last_dom_freq) <= 0.6: 
                freq_streak_count += 1
            else:
                freq_streak_count = 1 
                
            last_dom_freq = dom_freq
            
            if freq_streak_count >= STREAK_THRESHOLD:
                with data_lock:
                    sensor_data['dsp_freq'] = dom_freq
                    sensor_data['dsp_amp'] = highest_amp / (BUFFER_SIZE/2)
                    sensor_data['dsp_axis'] = 'Accel ' + dom_axis
                    
                    # TRIGGER ACTIVE LEARNING
                    if not sensor_data['needs_label'] and freq_streak_count == STREAK_THRESHOLD:
                        sensor_data['needs_label'] = True
                        global pending_label_window
                        pending_label_window['x'] = list(x_data)
                        pending_label_window['y'] = list(y_data)
                        pending_label_window['z'] = list(z_data)
                        print("TRIGGERED ACTIVE LEARNING NOTIFICATION")
            else:
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
    # If a window_id is present this label came from the ML system.
    # Route it through feedback manager (baseline + storage) and return early.
    if window_id:
        # Normalise label to what ML system expects
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
    thread = threading.Thread(target=websocket_reader, daemon=True)
    thread.start()

    dsp_thread = threading.Thread(target=dsp_worker, daemon=True)
    dsp_thread.start()

    ml_pipeline.start()

    app.run(host='0.0.0.0', port=5000, debug=False)
