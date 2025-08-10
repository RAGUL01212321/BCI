import csv
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import UploadFile, HTTPException
import numpy as np

EEG_CSV_FILE = Path("virtual_classroom_eeg.csv")

class CSVUploadHandler:
    """Handler for CSV file uploads and validation"""
    
    REQUIRED_COLUMNS = [
        'timestamp', 'student_id', 'noise_level', 'lighting', 'temperature', 
        'seating_comfort', 'teaching_method', 'time_of_day', 'session_duration', 
        'task_difficulty', 'class_strength', 'raw_eeg', 'delta', 'theta', 
        'alpha', 'beta', 'gamma', 'attention_index'
    ]
    
    def __init__(self):
        self.backup_file = Path("virtual_classroom_eeg_backup.csv")
    
    def validate_csv_format(self, file_content: str) -> Dict[str, Any]:
        """Validate the uploaded CSV file format and content"""
        try:
            # Parse CSV content using StringIO for better handling
            from io import StringIO
            csv_file = StringIO(file_content)
            reader = csv.DictReader(csv_file)
            
            # Check header
            if not reader.fieldnames:
                raise ValueError("CSV file has no header")
                
            missing_columns = set(self.REQUIRED_COLUMNS) - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Parse data rows
            rows = list(reader)
            
            if not rows:
                raise ValueError("CSV file contains no data rows")
            
            # Basic validation results
            validation_results = {
                'total_rows': len(rows),
                'students': set(),
                'validation_errors': []
            }
            
            # Collect student IDs (basic validation only)
            for row in rows:
                if 'student_id' in row and row['student_id']:
                    try:
                        student_id = int(float(row['student_id']))
                        validation_results['students'].add(student_id)
                    except ValueError:
                        pass
            
            # Generate data summary
            validation_results['data_summary'] = {
                'unique_students': len(validation_results['students']),
                'student_ids': sorted(list(validation_results['students']))
            }
            validation_results['has_errors'] = len(validation_results['validation_errors']) > 0
            
            return validation_results
            
        except Exception as e:
            raise ValueError(f"Error parsing CSV file: {str(e)}")
    
    def create_backup(self) -> bool:
        """Create backup of current CSV file"""
        try:
            if EEG_CSV_FILE.exists():
                shutil.copy2(EEG_CSV_FILE, self.backup_file)
                return True
            return False
        except Exception:
            return False
    
    def restore_backup(self) -> bool:
        """Restore CSV file from backup"""
        try:
            if self.backup_file.exists():
                shutil.copy2(self.backup_file, EEG_CSV_FILE)
                return True
            return False
        except Exception:
            return False
    
    def replace_csv_file(self, file_content: str) -> Dict[str, Any]:
        """Replace the current CSV file with uploaded content"""
        try:
            # Validate the file first
            validation_results = self.validate_csv_format(file_content)
            
            if validation_results['has_errors']:
                return {
                    'success': False,
                    'error': 'Validation failed',
                    'validation_results': validation_results
                }
            
            # Create backup of current file
            backup_created = self.create_backup()
            
            # Write new content to file
            with open(EEG_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                f.write(file_content)
            
            return {
                'success': True,
                'message': 'CSV file replaced successfully',
                'backup_created': backup_created,
                'validation_results': validation_results
            }
            
        except Exception as e:
            # Try to restore backup if something went wrong
            if self.backup_file.exists():
                self.restore_backup()
            
            return {
                'success': False,
                'error': f'Failed to replace CSV file: {str(e)}',
                'validation_results': None
            }

def process_uploaded_csv(upload_file: UploadFile) -> Dict[str, Any]:
    """Process and validate uploaded CSV file"""
    handler = CSVUploadHandler()
    
    try:
        # Read file content
        content = upload_file.file.read()
        
        # Decode content (handle different encodings)
        try:
            file_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                file_content = content.decode('utf-8-sig')  # Handle BOM
            except UnicodeDecodeError:
                file_content = content.decode('latin-1')  # Fallback encoding
        
        # Validate and replace file
        result = handler.replace_csv_file(file_content)
        
        # Add file information to result
        result['file_info'] = {
            'filename': upload_file.filename,
            'content_type': upload_file.content_type,
            'size': len(content)
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to process uploaded file: {str(e)}',
            'file_info': {
                'filename': upload_file.filename,
                'content_type': upload_file.content_type
            }
        }
    
    def validate_csv_format(self, file_content: str) -> Dict[str, Any]:
        """Validate the uploaded CSV file format and content"""
        try:
            # Parse CSV content using StringIO for better handling
            from io import StringIO
            csv_file = StringIO(file_content)
            reader = csv.DictReader(csv_file)
            
            # Check header
            if not reader.fieldnames:
                raise ValueError("CSV file has no header")
                
            missing_columns = set(self.REQUIRED_COLUMNS) - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Parse data rows
            rows = list(reader)
            
            if not rows:
                raise ValueError("CSV file contains no data rows")
            
            # Basic validation results
            validation_results = {
                'total_rows': len(rows),
                'students': set(),
                'validation_errors': []
            }
            
            # Collect student IDs (basic validation only)
            for row in rows:
                if 'student_id' in row and row['student_id']:
                    try:
                        student_id = int(float(row['student_id']))
                        validation_results['students'].add(student_id)
                    except ValueError:
                        pass
            
            # Generate data summary
            validation_results['data_summary'] = {
                'unique_students': len(validation_results['students']),
                'student_ids': sorted(list(validation_results['students']))
            }
            validation_results['has_errors'] = len(validation_results['validation_errors']) > 0
            
            return validation_results
            
        except Exception as e:
            raise ValueError(f"Error parsing CSV file: {str(e)}")
    
    def create_backup(self) -> bool:
        """Create backup of current CSV file"""
        try:
            if EEG_CSV_FILE.exists():
                shutil.copy2(EEG_CSV_FILE, self.backup_file)
                return True
            return False
        except Exception:
            return False
    
    def restore_backup(self) -> bool:
        """Restore CSV file from backup"""
        try:
            if self.backup_file.exists():
                shutil.copy2(self.backup_file, EEG_CSV_FILE)
                return True
            return False
        except Exception:
            return False
    
    def replace_csv_file(self, file_content: str) -> Dict[str, Any]:
        """Replace the current CSV file with uploaded content"""
        try:
            # Validate the file first
            validation_results = self.validate_csv_format(file_content)
            
            if validation_results['has_errors']:
                return {
                    'success': False,
                    'error': 'Validation failed',
                    'validation_results': validation_results
                }
            
            # Create backup of current file
            backup_created = self.create_backup()
            
            # Write new content to file
            with open(EEG_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                f.write(file_content)
            
            return {
                'success': True,
                'message': 'CSV file replaced successfully',
                'backup_created': backup_created,
                'validation_results': validation_results
            }
            
        except Exception as e:
            # Try to restore backup if something went wrong
            if self.backup_file.exists():
                self.restore_backup()
            
            return {
                'success': False,
                'error': f'Failed to replace CSV file: {str(e)}',
                'validation_results': None
            }

def process_uploaded_csv(upload_file: UploadFile) -> Dict[str, Any]:
    """Process and validate uploaded CSV file"""
    handler = CSVUploadHandler()
    
    try:
        # Read file content
        content = upload_file.file.read()
        
        # Decode content (handle different encodings)
        try:
            file_content = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                file_content = content.decode('utf-8-sig')  # Handle BOM
            except UnicodeDecodeError:
                file_content = content.decode('latin-1')  # Fallback encoding
        
        # Validate and replace file
        result = handler.replace_csv_file(file_content)
        
        # Add file information to result
        result['file_info'] = {
            'filename': upload_file.filename,
            'content_type': upload_file.content_type,
            'size': len(content)
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Failed to process uploaded file: {str(e)}',
            'file_info': {
                'filename': upload_file.filename,
                'content_type': upload_file.content_type
            }
        }
