from fastapi import APIRouter, HTTPException
from .dashboard import generate_dashboard_data

router = APIRouter(prefix="/virtual_classroom", tags=["dashboard"])

@router.get("/dashboard")
async def get_dashboard():
    """
    Get comprehensive EEG analysis dashboard with detailed insights, statistics, and AI-powered recommendations.
    
    Returns:
    - Basic statistics for all parameters
    - EEG band power analysis
    - Attention score analysis and distribution
    - Environmental factor correlations
    - Outlier detection and analysis
    - Session duration insights
    - AI-generated insights and recommendations
    - Chart configuration for frontend visualization
    - Summary metrics and actionable recommendations
    """
    try:
        dashboard_data = await generate_dashboard_data()
        return {
            "status": "success",
            "message": "Dashboard data generated successfully",
            "data": dashboard_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating dashboard: {str(e)}")
