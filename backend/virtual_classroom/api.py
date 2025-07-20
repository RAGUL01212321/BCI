from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from .signal_gen import process_parameters_and_generate_eeg

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

@router.post("/virtual-classroom/save-parameters")
def save_parameters(data: VirtualClassroomInput):
    record = data.dict()
    record["timestamp"] = datetime.utcnow().isoformat()
    # Directly generate EEG and plots from parameters (no JSON file)
    process_parameters_and_generate_eeg(record)
    return {"message": "Parameters received and EEG generated.", "data": record}
