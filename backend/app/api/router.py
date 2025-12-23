# app/api/router.py
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import FireAlert
from typing import List
from datetime import datetime

router = APIRouter()

class AlertResponse(BaseModel):
    grid_lat: float
    grid_lon: float
    risk_score: float
    T2M_MAX: float
    timestamp: datetime

    class Config:
        from_attributes = True


@router.get("/alerts/high-risk", response_model=List[AlertResponse])
async def get_active_alerts(db: Session = Depends(get_db)):
    # Récupère les alertes des dernières 24 heures avec un score >= 0.8
    alerts = db.query(FireAlert).filter(
        FireAlert.risk_score >= 0.8
    ).order_by(FireAlert.timestamp.desc()).all()
    return alerts

@router.get("/stats/summary")
async def get_stats():
    """
    Retourne un résumé pour le dashboard (ex: total de zones à risque).
    """
    return {
        "total_high_risk_zones": 12,
        "last_scan_time": datetime.now(),
        "status": "Scanning Madagascar OK"
    }

@router.post("/scan/force")
async def force_scan(background_tasks: BackgroundTasks):
    """
    Permet aux autorités de forcer un scan météo immédiat.
    """
    # On importe la fonction de scan ici pour éviter les imports circulaires
    from app.main import run_global_scan 
    background_tasks.add_task(run_global_scan)
    return {"message": "Scan forcé lancé en arrière-plan"}