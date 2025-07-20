from fastapi import APIRouter
from pydantic import BaseModel
import json
from pathlib import Path
from datetime import datetime
from .signal_gen import process_json_and_generate_eeg

router = APIRouter()

DATA_FILE = Path("virtual_classroom_data.json")

class VirtualClassroomInput(BaseModel):
    noise_level: float
    lighting: float
    temperature: float
    seating_comfort: float
    teaching_method: str
    time_of_day: str
    session_duration: float
    task_difficulty: float
    class_strength: int  # <--

@router.post("/virtual-classroom/save-parameters")
def save_parameters(data: VirtualClassroomInput):
    record = data.dict()
    record["timestamp"] = datetime.utcnow().isoformat()
    # Load existing data, handle empty or invalid file
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r") as f:
                content = f.read().strip()
                all_data = json.loads(content) if content else []
        except Exception:
            all_data = []
    else:
        all_data = []
    # Append new record
    all_data.append(record)
    # Save back to file
    with open(DATA_FILE, "w") as f:
        json.dump(all_data, f, indent=2)
    # Automatically generate EEG CSV
    process_json_and_generate_eeg()
    return {"message": "Parameters saved and EEG generated.", "data": record}
