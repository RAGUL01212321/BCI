import csv
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import shutil

EEG_CSV_FILE = Path("virtual_classroom_eeg.csv")
PLOTS_DIR = Path("virtual_classroom_eeg_plots")

# --- Function to generate synthetic EEG signal based on class parameters ---
def generate_synthetic_eeg_signals(params, length=32):
    def normalize(val, max_val=10):
        return min(max(val / max_val, 0.0), 1.0)  # Clamp between 0 and 1

    # Normalize inputs
    noise = normalize(params["noise_level"])
    lighting = normalize(params["lighting"])
    temp = normalize(params["temperature"])
    seat = normalize(params["seating_comfort"])
    session = normalize(params["session_duration"])
    difficulty = normalize(params["task_difficulty"])
    strength = normalize(params["class_strength"])

    # Base signal values influenced by class environment
    base_beta = 60 + 10 * (difficulty + seat - noise)
    base_alpha = 40 + 10 * (lighting + noise - difficulty)
    base_theta = 30 + 10 * (session + temp - seat)

    # Time axis
    x = np.linspace(0, 2 * np.pi, length)

    # Generate signals with sinusoidal trends and Gaussian noise
    noise_std = max(noise * 10, 0)  # Prevent negative std deviation
    beta = base_beta + 5 * np.sin(x) + np.random.normal(0, noise_std, length)
    alpha = base_alpha + 4 * np.sin(x + 1) + np.random.normal(0, noise_std, length)
    theta = base_theta + 3 * np.sin(x + 2) + np.random.normal(0, noise_std, length)

    # Clip to realistic EEG range
    beta = np.clip(beta, 0, 100)
    alpha = np.clip(alpha, 0, 100)
    theta = np.clip(theta, 0, 100)

    # Calculate attention index
    attention_index = np.clip(100 * beta / (alpha + theta + 1e-6), 0, 100)

    return {
        "beta": beta.round(2).tolist(),
        "alpha": alpha.round(2).tolist(),
        "theta": theta.round(2).tolist(),
        "attention_index": attention_index.round(2).tolist()
    }

# --- Per-student jittering for realism ---
def generate_student_signal(params, noise_scale=1.0):
    jittered = params.copy()
    # Clamp noise_level ≥ 0 after jitter
    jittered["noise_level"] = max(jittered["noise_level"] + random.uniform(-0.5, 0.5) * noise_scale, 0)
    jittered["lighting"] += random.uniform(-0.5, 0.5)
    jittered["temperature"] += random.uniform(-0.5, 0.5)
    jittered["seating_comfort"] += random.uniform(-0.5, 0.5)
    jittered["session_duration"] += random.uniform(-0.5, 0.5)
    jittered["task_difficulty"] += random.uniform(-0.5, 0.5)
    return generate_synthetic_eeg_signals(jittered)

def plot_and_save_signals(signals, student_id, output_dir):
    x = np.arange(len(signals["beta"]))
    plt.figure(figsize=(8, 5))
    plt.plot(x, signals["beta"], label="Beta")
    plt.plot(x, signals["alpha"], label="Alpha")
    plt.plot(x, signals["theta"], label="Theta")
    plt.plot(x, signals["attention_index"], label="Attention Index", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"Student {student_id} EEG Signals")
    plt.legend()
    plt.tight_layout()
    img_path = output_dir / f"student_{student_id}.png"
    plt.savefig(img_path)
    plt.close()

def wipe_and_prepare_plots_dir(plots_dir):
    if plots_dir.exists():
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(exist_ok=True)

def process_parameters_and_generate_eeg(params):
    # Wipe plots directory
    wipe_and_prepare_plots_dir(PLOTS_DIR)
    # Overwrite CSV file
    with open(EEG_CSV_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "timestamp", "student_id",
            "noise_level", "lighting", "temperature", "seating_comfort",
            "teaching_method", "time_of_day", "session_duration", "task_difficulty", "class_strength",
            "beta", "alpha", "theta", "attention_index"
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
                signals["beta"],
                signals["alpha"],
                signals["theta"],
                signals["attention_index"]
            ])
            plot_and_save_signals(signals, student_id, PLOTS_DIR)
    print(f"✅ EEG data written to {EEG_CSV_FILE}")
    print(f"✅ Plots saved in {PLOTS_DIR}")
