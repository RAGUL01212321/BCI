from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from typing import Any
from .signal_gen import process_parameters_and_generate_eeg

import csv
from pathlib import Path
import ast
import numpy as np

router = APIRouter()

class VirtualClassroomInput(BaseModel):
    noise_level: float
    lighting: float
    temperature: float
    seating_comfort: float
    teaching_method: str
    time_of_day: str
    session_duration: float
    task_difficulty: float
    class_strength: int

EEG_CSV_FILE = Path("virtual_classroom_eeg.csv")

@router.post("/virtual-classroom/generate-signals")
def save_parameters(data: VirtualClassroomInput):
    record = data.dict()
    record["timestamp"] = datetime.utcnow().isoformat()
    # Directly generate EEG and plots from parameters (no JSON file)
    process_parameters_and_generate_eeg(record)
    return {"message": "Parameters received and EEG generated.", "data": record}

@router.get("/virtual-classroom/get-signals")
def get_average_signals():
    if not EEG_CSV_FILE.exists():
        raise HTTPException(status_code=404, detail="EEG data not found. Please generate signals first.")
    signals = {"raw_eeg": [], "delta": [], "theta": [], "alpha": [], "beta": [], "gamma": []}
    classroom_info = {}
    timestamps = []
    num_students = 0
    with open(EEG_CSV_FILE, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            num_students += 1
            timestamps.append(row["timestamp"])
            for key in signals.keys():
                # Convert string list to list of floats
                arr = np.array(ast.literal_eval(row[key]), dtype=float)
                signals[key].append(arr)
            # Save classroom info from first row
            if not classroom_info:
                classroom_info = {
                    "noise_level": float(row["noise_level"]),
                    "lighting": float(row["lighting"]),
                    "temperature": float(row["temperature"]),
                    "seating_comfort": float(row["seating_comfort"]),
                    "teaching_method": row["teaching_method"],
                    "time_of_day": row["time_of_day"],
                    "session_duration": float(row["session_duration"]),
                    "task_difficulty": float(row["task_difficulty"]),
                    "class_strength": int(row["class_strength"])
                }
    if num_students == 0:
        raise HTTPException(status_code=404, detail="No student data found.")
    avg_signals = {}
    for key, arrs in signals.items():
        stacked = np.stack(arrs)
        avg = np.mean(stacked, axis=0)
        avg_signals[key] = avg.round(2).tolist()
    return {
        "timestamp": timestamps[0] if timestamps else None,
        "classroom_info": classroom_info,
        "average_signals": avg_signals
    }

@router.get("/virtual-classroom/get-signals/student_{student_id}")
def get_student_signals(student_id: int):
    if not EEG_CSV_FILE.exists():
        raise HTTPException(status_code=404, detail="EEG data not found. Please generate signals first.")
    with open(EEG_CSV_FILE, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if int(row["student_id"]) == student_id:
                signals = {
                    "raw_eeg": ast.literal_eval(row["raw_eeg"]),
                    "delta": ast.literal_eval(row["delta"]),
                    "theta": ast.literal_eval(row["theta"]),
                    "alpha": ast.literal_eval(row["alpha"]),
                    "beta": ast.literal_eval(row["beta"]),
                    "gamma": ast.literal_eval(row["gamma"]),
                    "attention_index": float(row["attention_index"])
                }
                classroom_info = {
                    "noise_level": float(row["noise_level"]),
                    "lighting": float(row["lighting"]),
                    "temperature": float(row["temperature"]),
                    "seating_comfort": float(row["seating_comfort"]),
                    "teaching_method": row["teaching_method"],
                    "time_of_day": row["time_of_day"],
                    "session_duration": float(row["session_duration"]),
                    "task_difficulty": float(row["task_difficulty"]),
                    "class_strength": int(row["class_strength"])
                }
                return {
                    "timestamp": row["timestamp"],
                    "student_id": student_id,
                    "classroom_info": classroom_info,
                    "signals": signals
                }
    raise HTTPException(status_code=404, detail=f"Student ID {student_id} not found.")
