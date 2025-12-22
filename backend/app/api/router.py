from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
import pandas as pd
import uuid
import json
import asyncio
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field
import os

from app.services.prediction_service import PredictionService
from app.services.weather_service import WeatherService

# Configuration du logging
logger = logging.getLogger(__name__)

# Modèles Pydantic pour la validation
class ScanRequest(BaseModel):
    """Modèle pour les requêtes de scan."""
    grid_file: str = Field(default="dataset/weather_master_grille.csv", description="Chemin vers le fichier de grille")
    risk_threshold: float = Field(default=0.80, ge=0.0, le=1.0, description="Seuil de risque (0.0 à 1.0)")
    max_concurrent: int = Field(default=10, ge=1, le=50, description="Nombre maximum de requêtes concurrentes")
    cluster_distance_km: float = Field(default=5.0, ge=0.1, le=100.0, description="Distance pour le clustering des alertes")
    notify_critical: bool = Field(default=True, description="Envoyer des notifications pour les alertes critiques")

class GridPoint(BaseModel):
    """Modèle pour un point de grille."""
    grid_lat: float = Field(..., ge=-90, le=90)
    grid_lon: float = Field(..., ge=-180, le=180)
    name: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=10)

class ScanResult(BaseModel):
    """Modèle pour les résultats de scan."""
    scan_id: str
    status: str
    alerts_count: int
    critical_alerts: int
    processing_time: float
    timestamp: datetime
    grid_points_processed: int

# Stockage des tâches en cours (en production, utiliser Redis ou une base de données)
active_scans: Dict[str, Dict] = {}
scan_results: Dict[str, Dict] = {}

router = APIRouter(prefix="/api/v1", tags=["prediction"])

# Initialisation du service de prédiction
MODEL_PATH = os.getenv("MODEL_PATH", "model/model_expert_98.json")
predictor = PredictionService(MODEL_PATH)

def validate_grid_file(grid_file: str) -> pd.DataFrame:
    """Valide et charge le fichier de grille."""
    if not os.path.exists(grid_file):
        raise HTTPException(
            status_code=404,
            detail=f"Fichier de grille non trouvé: {grid_file}"
        )
    
    try:
        grid = pd.read_csv(grid_file)
        
        # Vérifier les colonnes requises
        required_columns = ['grid_lat', 'grid_lon']
        missing_columns = [col for col in required_columns if col not in grid.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Colonnes manquantes dans le fichier de grille: {missing_columns}"
            )
        
        # Valider les coordonnées
        invalid_coords = grid[
            (grid['grid_lat'] < -90) | (grid['grid_lat'] > 90) |
            (grid['grid_lon'] < -180) | (grid['grid_lon'] > 180)
        ]
        
        if not invalid_coords.empty:
            raise HTTPException(
                status_code=400,
                detail=f"{len(invalid_coords)} points ont des coordonnées invalides"
            )
        
        # Supprimer les doublons
        grid = grid[required_columns].drop_duplicates()
        
        logger.info(f"Grille chargée: {len(grid)} points valides")
        return grid
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la grille: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du chargement du fichier de grille: {str(e)}"
        )

async def run_scan_task(scan_id: str, grid: pd.DataFrame, request: ScanRequest):
    """Exécute la tâche de scan en arrière-plan."""
    try:
        # Marquer la tâche comme en cours
        active_scans[scan_id] = {
            "status": "running",
            "start_time": datetime.now(),
            "total_points": len(grid),
            "processed_points": 0,
            "alerts_found": 0
        }
        
        logger.info(f"Début du scan {scan_id} sur {len(grid)} points")
        
        # Configuration du prédicteur
        predictor.risk_threshold = request.risk_threshold
        
        # Exécution du scan
        start_time = datetime.now()
        alerts = await predictor.run_island_scan(
            grid_points=grid,
            max_concurrent=request.max_concurrent,
            cluster_distance_km=request.cluster_distance_km
        )
        end_time = datetime.now()
        
        # Calcul des statistiques
        critical_alerts = len([a for a in alerts if a.get('risk_level') == 'CRITICAL'])
        processing_time = (end_time - start_time).total_seconds()
        
        # Stocker les résultats
        scan_results[scan_id] = {
            "scan_id": scan_id,
            "status": "completed",
            "alerts": alerts,
            "alerts_count": len(alerts),
            "critical_alerts": critical_alerts,
            "processing_time": processing_time,
            "timestamp": end_time,
            "grid_points_processed": len(grid),
            "parameters": request.dict(),
            "summary": {
                "total_points": len(grid),
                "high_risk_zones": len([a for a in alerts if a.get('risk') >= 80]),
                "medium_risk_zones": len([a for a in alerts if 65 <= a.get('risk', 0) < 80]),
                "clusters_found": len([a for a in alerts if a.get('type') == 'CLUSTER'])
            }
        }
        
        # Mettre à jour le statut
        active_scans[scan_id].update({
            "status": "completed",
            "end_time": end_time,
            "alerts_found": len(alerts),
            "processing_time": processing_time
        })
        
        logger.info(f"Scan {scan_id} terminé: {len(alerts)} alertes détectées en {processing_time:.2f}s")
        
        # Envoyer des notifications si configuré
        if request.notify_critical and critical_alerts > 0:
            await send_critical_alerts_notification(scan_id, alerts)
            
    except Exception as e:
        logger.error(f"Erreur lors du scan {scan_id}: {str(e)}")
        active_scans[scan_id] = {
            "status": "failed",
            "error": str(e),
            "end_time": datetime.now()
        }
        scan_results[scan_id] = {
            "scan_id": scan_id,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now()
        }

async def send_critical_alerts_notification(scan_id: str, alerts: List[Dict]):
    """Envoie une notification pour les alertes critiques."""
    critical_alerts = [a for a in alerts if a.get('risk_level') == 'CRITICAL']
    
    if critical_alerts:
        notification = {
            "scan_id": scan_id,
            "timestamp": datetime.now().isoformat(),
            "critical_alerts_count": len(critical_alerts),
            "alerts": critical_alerts[:10],  # Limiter à 10 pour la notification
            "message": f"🚨 {len(critical_alerts)} ALERTES CRITIQUES DÉTECTÉES"
        }
        
        # Log pour simulation
        logger.warning(f"NOTIFICATION: {notification['message']}")
        
        # Ici, vous pourriez ajouter:
        # - Envoi d'email
        # - Webhook vers un dashboard
        # - Notification Slack/Discord
        # - SMS via Twilio

@router.post("/scan", response_model=Dict)
async def start_scan(
    background_tasks: BackgroundTasks,
    request: ScanRequest = None
):
    """
    Lance l'analyse globale de l'île en tâche de fond.
    
    Args:
        request: Paramètres du scan (optionnel, valeurs par défaut utilisées si non fourni)
    
    Returns:
        Informations sur la tâche lancée
    """
    if request is None:
        request = ScanRequest()
    
    try:
        # Valider et charger la grille
        grid = validate_grid_file(request.grid_file)
        
        # Générer un ID unique pour la tâche
        scan_id = str(uuid.uuid4())
        
        # Lancer la tâche en arrière-plan
        background_tasks.add_task(
            run_scan_task,
            scan_id=scan_id,
            grid=grid,
            request=request
        )
        
        # Enregistrer la tâche
        active_scans[scan_id] = {
            "status": "pending",
            "start_time": datetime.now(),
            "parameters": request.dict()
        }
        
        logger.info(f"Nouveau scan lancé: {scan_id}")
        
        return {
            "status": "success",
            "message": "Le scan de Madagascar a démarré en arrière-plan.",
            "scan_id": scan_id,
            "details": {
                "grid_points": len(grid),
                "risk_threshold": request.risk_threshold,
                "estimated_time": f"~{len(grid) * 2} secondes",  # Estimation grossière
                "monitor_url": f"/api/v1/scans/{scan_id}"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors du lancement du scan: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne lors du lancement du scan: {str(e)}"
        )

@router.get("/scans/{scan_id}", response_model=Dict)
async def get_scan_status(scan_id: str):
    """
    Récupère le statut d'une tâche de scan.
    
    Args:
        scan_id: ID du scan
    
    Returns:
        Statut et résultats du scan
    """
    # Vérifier si le scan existe
    if scan_id not in active_scans and scan_id not in scan_results:
        raise HTTPException(
            status_code=404,
            detail=f"Scan {scan_id} non trouvé"
        )
    
    # Récupérer les informations
    scan_info = active_scans.get(scan_id, {})
    result_info = scan_results.get(scan_id, {})
    
    response = {
        "scan_id": scan_id,
        **scan_info,
        **result_info
    }
    
    # Ajouter des informations de progression si le scan est en cours
    if scan_info.get("status") == "running":
        response["progress"] = {
            "percentage": (
                scan_info.get("processed_points", 0) / 
                scan_info.get("total_points", 1) * 100
            ) if scan_info.get("total_points", 0) > 0 else 0,
            "processed": scan_info.get("processed_points", 0),
            "total": scan_info.get("total_points", 0)
        }
    
    return response

@router.get("/scans", response_model=List[Dict])
async def list_scans(
    limit: int = 10,
    status: Optional[str] = None
):
    """
    Liste les scans récents.
    
    Args:
        limit: Nombre maximum de scans à retourner
        status: Filtrer par statut (running, completed, failed, pending)
    
    Returns:
        Liste des scans
    """
    all_scans = []
    
    # Ajouter les scans actifs
    for scan_id, scan_info in list(active_scans.items())[:limit]:
        all_scans.append({
            "scan_id": scan_id,
            **scan_info,
            "type": "active"
        })
    
    # Ajouter les scans terminés
    for scan_id, result_info in list(scan_results.items())[:limit]:
        all_scans.append({
            "scan_id": scan_id,
            **result_info,
            "type": "completed"
        })
    
    # Trier par date (les plus récents d'abord)
    all_scans.sort(key=lambda x: x.get("timestamp", datetime.min), reverse=True)
    
    # Filtrer par statut si spécifié
    if status:
        all_scans = [s for s in all_scans if s.get("status") == status]
    
    return all_scans[:limit]

@router.delete("/scans/{scan_id}")
async def cancel_scan(scan_id: str):
    """
    Annule un scan en cours.
    
    Args:
        scan_id: ID du scan à annuler
    
    Returns:
        Confirmation de l'annulation
    """
    if scan_id not in active_scans:
        raise HTTPException(
            status_code=404,
            detail=f"Scan {scan_id} non trouvé ou déjà terminé"
        )
    
    scan_info = active_scans[scan_id]
    
    if scan_info.get("status") != "running":
        raise HTTPException(
            status_code=400,
            detail=f"Le scan {scan_id} n'est pas en cours d'exécution"
        )
    
    # Marquer comme annulé
    active_scans[scan_id]["status"] = "cancelled"
    active_scans[scan_id]["end_time"] = datetime.now()
    
    logger.info(f"Scan {scan_id} annulé par l'utilisateur")
    
    return {
        "status": "success",
        "message": f"Scan {scan_id} annulé avec succès"
    }

@router.post("/predict/single", response_model=Dict)
async def predict_single_point(point: GridPoint):
    """
    Effectue une prédiction pour un point unique.
    
    Args:
        point: Coordonnées du point à analyser
    
    Returns:
        Risque de dengue pour le point spécifié
    """
    try:
        # Récupérer les données météo
        weather_df = await WeatherService.get_last_35_days_async(
            point.grid_lat,
            point.grid_lon
        )
        
        if weather_df is None or len(weather_df) < 30:
            raise HTTPException(
                status_code=400,
                detail="Données météo insuffisantes pour ce point"
            )
        
        # Préparer les features et faire la prédiction
        input_data = predictor.prepare_features(
            weather_df,
            point.grid_lat,
            point.grid_lon
        )
        
        # Obtenir la prédiction
        prob = predictor.model.predict_proba(input_data)[0][1]
        risk_level = predictor._get_risk_level(prob)
        
        # Obtenir l'explication
        explanation = predictor.get_prediction_explanation(input_data)
        
        return {
            "lat": point.grid_lat,
            "lon": point.grid_lon,
            "risk": round(float(prob) * 100, 2),
            "risk_level": risk_level,
            "threshold_exceeded": prob > predictor.risk_threshold,
            "weather_data_points": len(weather_df),
            "prediction_explanation": explanation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction pour ({point.grid_lat}, {point.grid_lon}): {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction: {str(e)}"
        )

@router.get("/system/health", response_model=Dict)
async def health_check():
    """
    Vérifie l'état du système et des services.
    
    Returns:
        État de santé du système
    """
    checks = {
        "api": "healthy",
        "model": "unknown",
        "weather_service": "unknown",
        "cache": "unknown"
    }
    
    # Vérifier le modèle
    try:
        if hasattr(predictor.model, 'feature_importances_'):
            checks["model"] = "healthy"
        else:
            checks["model"] = "warning (no feature importance)"
    except:
        checks["model"] = "unhealthy"
    
    # Vérifier le service météo
    try:
        stats = WeatherService.get_cache_stats()
        checks["weather_service"] = "healthy"
        checks["cache"] = f"healthy ({stats.get('valid_entries', 0)} entries)"
    except:
        checks["weather_service"] = "unhealthy"
    
    # Vérifier les fichiers nécessaires
    required_files = [
        MODEL_PATH,
        "dataset/weather_master_grille.csv"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        checks["files"] = f"missing: {missing_files}"
    else:
        checks["files"] = "all_present"
    
    overall_status = "healthy" if all(
        "healthy" in str(v).lower() or "warning" in str(v).lower() 
        for k, v in checks.items() if k != "files"
    ) else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
        "active_scans": len([s for s in active_scans.values() if s.get("status") == "running"]),
        "completed_scans": len(scan_results)
    }

@router.get("/alerts/recent", response_model=List[Dict])
async def get_recent_alerts(
    hours: int = 24,
    min_risk: float = 70.0
):
    """
    Récupère les alertes récentes.
    
    Args:
        hours: Nombre d'heures à regarder en arrière
        min_risk: Niveau de risque minimum
    
    Returns:
        Liste des alertes récentes
    """
    cutoff_time = datetime.now() - timedelta(hours=hours)
    
    recent_alerts = []
    
    for scan_id, result in scan_results.items():
        if result.get("status") == "completed" and result.get("timestamp") >= cutoff_time:
            alerts = result.get("alerts", [])
            filtered_alerts = [
                {**alert, "scan_id": scan_id, "scan_time": result.get("timestamp")}
                for alert in alerts
                if alert.get("risk", 0) >= min_risk
            ]
            recent_alerts.extend(filtered_alerts)
    
    # Trier par risque (le plus élevé d'abord)
    recent_alerts.sort(key=lambda x: x.get("risk", 0), reverse=True)
    
    # Limiter le nombre de résultats
    return recent_alerts[:100]

# Endpoint pour nettoyer les vieux scans
@router.delete("/scans/cleanup/{days_old}")
async def cleanup_old_scans(days_old: int = 7):
    """
    Supprime les scans plus vieux que X jours.
    
    Args:
        days_old: Âge minimum en jours
    
    Returns:
        Nombre de scans supprimés
    """
    cutoff_time = datetime.now() - timedelta(days=days_old)
    
    scans_deleted = 0
    
    # Nettoyer les scans actifs
    for scan_id in list(active_scans.keys()):
        scan_info = active_scans[scan_id]
        scan_time = scan_info.get("end_time") or scan_info.get("start_time")
        
        if scan_time and scan_time < cutoff_time:
            del active_scans[scan_id]
            scans_deleted += 1
    
    # Nettoyer les résultats
    for scan_id in list(scan_results.keys()):
        result_info = scan_results[scan_id]
        scan_time = result_info.get("timestamp")
        
        if scan_time and scan_time < cutoff_time:
            del scan_results[scan_id]
            scans_deleted += 1
    
    logger.info(f"Nettoyage effectué: {scans_deleted} scans supprimés")
    
    return {
        "status": "success",
        "message": f"{scans_deleted} scans supprimés",
        "cutoff_date": cutoff_time.isoformat()
    }