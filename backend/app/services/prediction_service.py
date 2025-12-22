import pandas as pd
import xgboost as xgb
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
from app.services.weather_service import WeatherService
import logging

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model_path: str):
        """Initialise le service de prédiction avec le modèle XGBoost."""
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        
    def prepare_features(self, df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
        """Calcule les variables expertes (features engineering) pour la prédiction."""
        df = df.copy()
        df = df.sort_values('date')
        
        # Calcul des agrégations temporelles
        df['rain_7d'] = df['PRECTOTCORR'].rolling(7, min_periods=1).sum()
        df['rain_30d'] = df['PRECTOTCORR'].rolling(30, min_periods=1).sum()
        df['temp_7d_avg'] = df['T2M_MAX'].rolling(7, min_periods=1).mean()
        df['temp_7d_max'] = df['T2M_MAX'].rolling(7, min_periods=1).max()
        df['humidity_7d_avg'] = df['RH2M'].rolling(7, min_periods=1).mean()
        df['wind_7d_max'] = df['WS10M_MAX'].rolling(7, min_periods=1).max()
        
        # Caractéristiques temporelles
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['season'] = (df['month'] % 12 + 3) // 3  # 1:hiver, 2:printemps, 3:été, 4:automne
        
        # Features supplémentaires
        df['temp_humidity_interaction'] = df['T2M_MAX'] * df['RH2M'] / 100
        df['rain_intensity_7d'] = df['rain_7d'] / 7
        
        # Utiliser la dernière ligne (données les plus récentes)
        latest = df.iloc[[-1]].copy()
        latest['grid_lat'] = lat
        latest['grid_lon'] = lon
        
        # Sélection des features pour le modèle
        features = [
            'grid_lat', 'grid_lon', 'T2M_MAX', 'RH2M', 'PRECTOTCORR',
            'WS10M_MAX', 'rain_7d', 'rain_30d', 'temp_7d_avg',
            'temp_7d_max', 'humidity_7d_avg', 'wind_7d_max',
            'month', 'day_of_year', 'season',
            'temp_humidity_interaction', 'rain_intensity_7d'
        ]
        
        # S'assurer que toutes les colonnes existent
        available_features = [f for f in features if f in latest.columns]
        return latest[available_features]
    
    async def _process_grid_point(self, point: Dict, semaphore: asyncio.Semaphore) -> Optional[Dict]:
        """Traite un point de grille individuel de manière asynchrone."""
        async with semaphore:
            try:
                # Récupérer les données météo
                weather_df = await WeatherService.get_last_35_days(
                    point['grid_lat'], 
                    point['grid_lon']
                )
                
                if weather_df is None or len(weather_df) < 30:
                    logger.warning(f"Données insuffisantes pour le point ({point['grid_lat']}, {point['grid_lon']})")
                    return None
                
                # Préparer les features et faire la prédiction
                input_data = self.prepare_features(
                    weather_df, 
                    point['grid_lat'], 
                    point['grid_lon']
                )
                
                # Prédiction dans un thread séparé pour ne pas bloquer le event loop
                loop = asyncio.get_event_loop()
                prob = await loop.run_in_executor(
                    None, 
                    lambda: self.model.predict_proba(input_data)[0][1]
                )
                
                # Seuil de risque ajustable
                risk_threshold = 0.80
                if prob > risk_threshold:
                    return {
                        "lat": point['grid_lat'],
                        "lon": point['grid_lon'],
                        "risk": round(float(prob) * 100, 2),
                        "risk_level": self._get_risk_level(prob),
                        "date": datetime.now().isoformat(),
                        "features": input_data.iloc[0].to_dict() if not input_data.empty else {}
                    }
                    
            except Exception as e:
                logger.error(f"Erreur lors du traitement du point ({point['grid_lat']}, {point['grid_lon']}): {str(e)}")
                return None
            
            return None
    
    def _get_risk_level(self, probability: float) -> str:
        """Détermine le niveau de risque basé sur la probabilité."""
        if probability >= 0.90:
            return "CRITICAL"
        elif probability >= 0.80:
            return "HIGH"
        elif probability >= 0.65:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def run_island_scan(self, grid_points: pd.DataFrame, 
                            max_concurrent: int = 10) -> List[Dict]:
        """
        Scanne les points de la grille et identifie les zones à risque.
        
        Args:
            grid_points: DataFrame contenant les points de grille (lat, lon)
            max_concurrent: Nombre maximum de requêtes concurrentes
            
        Returns:
            Liste des alertes détectées
        """
        logger.info(f"Début du scan de {len(grid_points)} points...")
        
        # Limiteur de concurrence pour éviter de surcharger l'API météo
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Traiter tous les points en parallèle
        tasks = [
            self._process_grid_point(point, semaphore)
            for _, point in grid_points.iterrows()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Filtrer les résultats None
        alerts = [alert for alert in results if alert is not None]
        
        # Trier par niveau de risque
        alerts.sort(key=lambda x: x['risk'], reverse=True)
        
        # Regrouper les alertes proches géographiquement
        clustered_alerts = self._cluster_alerts(alerts)
        
        logger.info(f"Scan terminé : {len(alerts)} alertes détectées, {len(clustered_alerts)} clusters identifiés")
        
        # Envoyer les notifications si nécessaire
        if alerts:
            await self._send_notifications(clustered_alerts)
        
        return clustered_alerts
    
    def _cluster_alerts(self, alerts: List[Dict], distance_km: float = 5.0) -> List[Dict]:
        """
        Regroupe les alertes proches géographiquement.
        
        Args:
            alerts: Liste des alertes
            distance_km: Distance maximale pour le regroupement (en km)
            
        Returns:
            Alertes regroupées avec le centre du cluster
        """
        if not alerts:
            return []
        
        # Conversion simple en degrés (approximatif)
        distance_deg = distance_km / 111.0
        
        clusters = []
        visited = set()
        
        for i, alert in enumerate(alerts):
            if i in visited:
                continue
                
            cluster = [alert]
            visited.add(i)
            
            # Chercher les alertes voisines
            for j, other_alert in enumerate(alerts):
                if j in visited:
                    continue
                    
                dist = np.sqrt(
                    (alert['lat'] - other_alert['lat'])**2 + 
                    (alert['lon'] - other_alert['lon'])**2
                )
                
                if dist <= distance_deg:
                    cluster.append(other_alert)
                    visited.add(j)
            
            # Calculer le centre du cluster
            if len(cluster) > 1:
                avg_lat = np.mean([a['lat'] for a in cluster])
                avg_lon = np.mean([a['lon'] for a in cluster])
                max_risk = max([a['risk'] for a in cluster])
                
                clusters.append({
                    "lat": round(avg_lat, 4),
                    "lon": round(avg_lon, 4),
                    "risk": max_risk,
                    "risk_level": self._get_risk_level(max_risk / 100),
                    "cluster_size": len(cluster),
                    "points": cluster,
                    "type": "CLUSTER"
                })
            else:
                clusters.append({**alert, "cluster_size": 1, "type": "SINGLE"})
        
        return clusters
    
    async def _send_notifications(self, alerts: List[Dict]):
        """Envoie des notifications pour les alertes critiques."""
        critical_alerts = [a for a in alerts if a['risk'] >= 90]
        
        if critical_alerts:
            # Log pour simulation - à remplacer par un vrai service de notification
            logger.warning(f"🚨 ALERTES CRITIQUES DÉTECTÉES: {len(critical_alerts)} zones")
            for alert in critical_alerts:
                logger.warning(
                    f"  - Position: ({alert['lat']}, {alert['lon']}), "
                    f"Risque: {alert['risk']}%, "
                    f"Taille cluster: {alert.get('cluster_size', 1)}"
                )
            
            # Ici, vous pourriez ajouter:
            # 1. Envoi d'email via SMTP
            # 2. Notification SMS via Twilio
            # 3. Webhook vers un dashboard
            # 4. Notification push
    
    def get_prediction_explanation(self, features: pd.DataFrame) -> Dict:
        """
        Fournit une explication de la prédiction (importance des features).
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_names = self.model.get_booster().feature_names
            
            explanation = {
                'feature_importance': dict(zip(feature_names, importance.tolist())),
                'top_features': sorted(
                    zip(feature_names, importance),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            }
            return explanation
        
        return {"message": "Feature importance non disponible"}