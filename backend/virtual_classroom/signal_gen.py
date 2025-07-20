import csv
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import shutil
from scipy.signal import butter, filtfilt

EEG_CSV_FILE = Path("virtual_classroom_eeg.csv")
PLOTS_DIR = Path("virtual_classroom_eeg_plots")

# --- Helper: Bandpass filter ---
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# --- Generate realistic EEG signal ---
def generate_realistic_eeg(params, length=1024, fs=128):
    t = np.arange(length) / fs
    noise_level = params["noise_level"]
    lighting = params["lighting"]
    temp = params["temperature"]
    seat = params["seating_comfort"]
    session = params["session_duration"]
    difficulty = params["task_difficulty"]

    eeg = np.zeros_like(t)

    def band_noise(low, high, amplitude):
        noise = np.random.normal(0, 1, len(t))
        band = bandpass_filter(noise, low, high, fs)
        return band * amplitude

    eeg += band_noise(0.5, 4, 5 + 0.5 * temp)                  # Delta
    eeg += band_noise(4, 8, 7 + 0.4 * session)                 # Theta
    eeg += band_noise(8, 14, 10 + 0.5 * (10 - difficulty))     # Alpha
    eeg += band_noise(14, 30, 6 + 0.6 * difficulty)            # Beta
    eeg += band_noise(30, 63, 4 + 0.3 * lighting)              # Gamma

    # Add artifacts (random bursts)
    for _ in range(random.randint(1, 3)):
        idx = random.randint(0, len(t)-20)
        eeg[idx:idx+20] += np.random.normal(30, 5, 20)

    # Add baseline drift and white noise
    drift = np.cumsum(np.random.normal(0, 0.05, len(t)))
    white = np.random.normal(0, 1 + 0.2 * noise_level, len(t))
    eeg += drift + white

    return eeg

# --- Decompose EEG into bands ---
def decompose_eeg_bands(eeg, fs=128):
    return {
        "Delta": bandpass_filter(eeg, 0.5, 4, fs),
        "Theta": bandpass_filter(eeg, 4, 8, fs),
        "Alpha": bandpass_filter(eeg, 8, 14, fs),
        "Beta": bandpass_filter(eeg, 14, 30, fs),
        "Gamma": bandpass_filter(eeg, 30, 63, fs),
    }

# --- Per-student jittering for realism ---
def generate_student_signal(params, length=1024, fs=128):
    jittered = params.copy()
    for key in ["noise_level", "lighting", "temperature", "seating_comfort", "session_duration", "task_difficulty"]:
        jittered[key] += random.uniform(-0.5, 0.5)

    eeg = generate_realistic_eeg(jittered, length=length, fs=fs)
    bands = decompose_eeg_bands(eeg, fs=fs)

    # Attention index: beta / (alpha + theta)
    ai = 100 * np.mean(np.abs(bands["Beta"])) / (
        np.mean(np.abs(bands["Alpha"])) + np.mean(np.abs(bands["Theta"])) + 1e-6
    )
    ai = np.clip(ai, 0, 100)

    return {
        "raw_eeg": eeg.round(2).tolist(),
        **{band: data.round(2).tolist() for band, data in bands.items()},
        "attention_index": round(ai, 2)
    }

# --- Plot and save ---
def plot_and_save_signals(signals, student_id, base_dir):
    student_dir = base_dir / f"student_{student_id}"
    student_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(signals["raw_eeg"])) / 128  # time axis in seconds

    # Raw EEG plot
    plt.figure(figsize=(10, 3))
    plt.plot(x, signals["raw_eeg"], color="black")
    plt.title("Raw EEG")
    plt.xlabel("Time (sec)")
    plt.tight_layout()
    plt.savefig(student_dir / "raw_eeg.png")
    plt.close()

    # Combined subplot of all bands
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, signals["raw_eeg"])
    axes[0].set_ylabel("rawEEG")
    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    for i, band in enumerate(band_names):
        axes[i+1].plot(x, signals[band])
        axes[i+1].set_ylabel(band)
    axes[-1].set_xlabel("time (sec)")
    plt.tight_layout()
    plt.savefig(student_dir / "eeg_decomposition.png")
    plt.close()

def plot_and_save_mean_signals(mean_signals, base_dir):
    mean_dir = base_dir / "MEAN"
    mean_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(mean_signals["raw_eeg"])) / 128  # time axis in seconds

    # Raw EEG mean plot
    plt.figure(figsize=(10, 3))
    plt.plot(x, mean_signals["raw_eeg"], color="black")
    plt.title("Mean Raw EEG (All Students)")
    plt.xlabel("Time (sec)")
    plt.tight_layout()
    plt.savefig(mean_dir / "raw_eeg.png")
    plt.close()

    # Combined subplot of all mean bands
    fig, axes = plt.subplots(6, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, mean_signals["raw_eeg"])
    axes[0].set_ylabel("rawEEG")
    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    for i, band in enumerate(band_names):
        axes[i+1].plot(x, mean_signals[band])
        axes[i+1].set_ylabel(band)
    axes[-1].set_xlabel("time (sec)")
    plt.tight_layout()
    plt.savefig(mean_dir / "eeg_decomposition.png")
    plt.close()

# --- Prepare plots directory ---
def wipe_and_prepare_plots_dir(plots_dir):
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(exist_ok=True)

# --- Main generator ---
def process_parameters_and_generate_eeg(params):
    wipe_and_prepare_plots_dir(PLOTS_DIR)
    all_signals = {"raw_eeg": [], "Delta": [], "Theta": [], "Alpha": [], "Beta": [], "Gamma": []}
    with open(EEG_CSV_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "timestamp", "student_id",
            "noise_level", "lighting", "temperature", "seating_comfort",
            "teaching_method", "time_of_day", "session_duration", "task_difficulty", "class_strength",
            "raw_eeg", "delta", "theta", "alpha", "beta", "gamma", "attention_index"
        ])
        num_students = int(params.get("class_strength", 50))
        for student_id in range(1, num_students + 1):
            signals = generate_student_signal(params)
            writer.writerow([
                params.get("timestamp"),
                student_id,
                params.get("noise_level"),
                params.get("lighting"),
                params.get("temperature"),
                params.get("seating_comfort"),
                params.get("teaching_method"),
                params.get("time_of_day"),
                params.get("session_duration"),
                params.get("task_difficulty"),
                params.get("class_strength"),
                signals["raw_eeg"],
                signals["Delta"],
                signals["Theta"],
                signals["Alpha"],
                signals["Beta"],
                signals["Gamma"],
                signals["attention_index"]
            ])
            plot_and_save_signals(signals, student_id, PLOTS_DIR)
            # Collect for mean
            for key in all_signals.keys():
                all_signals[key].append(np.array(signals[key], dtype=float))
    # Compute and plot mean signals
    mean_signals = {}
    for key, arrs in all_signals.items():
        stacked = np.stack(arrs)
        mean_signals[key] = np.mean(stacked, axis=0).round(2).tolist()
    plot_and_save_mean_signals(mean_signals, PLOTS_DIR)
    print(f"✅ EEG data written to {EEG_CSV_FILE}")
    print(f"✅ Plots saved in {PLOTS_DIR}")
