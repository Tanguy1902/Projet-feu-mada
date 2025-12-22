import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
import hashlib
from functools import lru_cache
import logging
import time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class WeatherParameter(Enum):
    """Paramètres météo disponibles via l'API NASA POWER."""
    TEMPERATURE_MAX = "T2M_MAX"
    HUMIDITY = "RH2M"
    PRECIPITATION = "PRECTOTCORR"
    WIND_SPEED_MAX = "WS10M_MAX"
    SOLAR_RADIATION = "ALLSKY_SFC_SW_DWN"
    PRESSURE = "PS"
    TEMPERATURE_MIN = "T2M_MIN"
    WIND_SPEED = "WS10M"

@dataclass
class WeatherCacheEntry:
    """Entrée de cache pour les données météo."""
    data: pd.DataFrame
    timestamp: datetime
    lat: float
    lon: float

class WeatherService:
    # Cache en mémoire (peut être remplacé par Redis en production)
    _cache: Dict[str, WeatherCacheEntry] = {}
    _cache_duration_hours = 24  # Durée de vie du cache
    
    # Configuration de l'API NASA POWER
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    DEFAULT_PARAMETERS = [
        WeatherParameter.TEMPERATURE_MAX.value,
        WeatherParameter.HUMIDITY.value,
        WeatherParameter.PRECIPITATION.value,
        WeatherParameter.WIND_SPEED_MAX.value,
        WeatherParameter.SOLAR_RADIATION.value,
        WeatherParameter.PRESSURE.value
    ]
    
    @classmethod
    def _get_cache_key(cls, lat: float, lon: float, start_date: str, end_date: str) -> str:
        """Génère une clé unique pour le cache."""
        key_str = f"{lat}_{lon}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @classmethod
    def _is_cache_valid(cls, cache_entry: WeatherCacheEntry) -> bool:
        """Vérifie si une entrée de cache est encore valide."""
        cache_age = datetime.now() - cache_entry.timestamp
        return cache_age.total_seconds() < (cls._cache_duration_hours * 3600)
    
    @classmethod
    def clear_cache(cls):
        """Vide le cache."""
        cls._cache.clear()
        logger.info("Cache météo vidé.")
    
    @classmethod
    def get_cache_stats(cls) -> Dict:
        """Retourne les statistiques du cache."""
        valid_entries = sum(1 for entry in cls._cache.values() 
                          if cls._is_cache_valid(entry))
        return {
            "total_entries": len(cls._cache),
            "valid_entries": valid_entries,
            "invalid_entries": len(cls._cache) - valid_entries
        }
    
    @staticmethod
    def get_last_n_days(lat: float, lon: float, days: int = 35) -> Optional[pd.DataFrame]:
        """
        Récupère l'historique météo des N derniers jours.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Nombre de jours d'historique (35 par défaut)
            
        Returns:
            DataFrame avec les données météo ou None en cas d'erreur
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return WeatherService.get_weather_data(
            lat=lat,
            lon=lon,
            start_date=start_date,
            end_date=end_date
        )
    
    @classmethod
    def get_weather_data(cls, lat: float, lon: float, 
                        start_date: datetime, end_date: datetime,
                        parameters: List[str] = None) -> Optional[pd.DataFrame]:
        """
        Récupère les données météo pour une période donnée.
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Date de début
            end_date: Date de fin
            parameters: Liste des paramètres à récupérer
            
        Returns:
            DataFrame avec les données météo
        """
        # Validation des entrées
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            logger.error(f"Coordonnées invalides: lat={lat}, lon={lon}")
            return None
        
        if start_date > end_date:
            logger.error(f"Dates invalides: start_date={start_date} > end_date={end_date}")
            return None
        
        # Formatage des dates
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # Vérification du cache
        cache_key = cls._get_cache_key(lat, lon, start_str, end_str)
        if cache_key in cls._cache and cls._is_cache_valid(cls._cache[cache_key]):
            logger.debug(f"Données récupérées du cache pour ({lat}, {lon})")
            return cls._cache[cache_key].data.copy()
        
        # Paramètres par défaut si non spécifiés
        if parameters is None:
            parameters = cls.DEFAULT_PARAMETERS
        
        # Construction de l'URL
        params_str = ",".join(parameters)
        
        url = (f"{cls.BASE_URL}?"
               f"parameters={params_str}&"
               f"community=AG&"
               f"longitude={lon}&"
               f"latitude={lat}&"
               f"start={start_str}&"
               f"end={end_str}&"
               f"format=JSON")
        
        logger.info(f"Requête API NASA pour ({lat}, {lon}): {start_str} à {end_str}")
        
        # Tentative avec retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Validation de la réponse
                    if 'properties' not in data or 'parameter' not in data['properties']:
                        logger.error(f"Format de réponse invalide pour ({lat}, {lon})")
                        return None
                    
                    # Extraction et transformation des données
                    df = cls._process_api_response(data, parameters)
                    
                    if df is not None and not df.empty:
                        # Mise en cache
                        cls._cache[cache_key] = WeatherCacheEntry(
                            data=df.copy(),
                            timestamp=datetime.now(),
                            lat=lat,
                            lon=lon
                        )
                        
                        logger.info(f"Données récupérées avec succès pour ({lat}, {lon}): {len(df)} jours")
                        return df
                    
                elif response.status_code == 429:
                    # Trop de requêtes - attendre avant de réessayer
                    wait_time = (attempt + 1) * 10  # 10, 20, 30 secondes
                    logger.warning(f"Rate limit atteint. Nouvelle tentative dans {wait_time}s")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    logger.error(f"Erreur API (code {response.status_code}) pour ({lat}, {lon})")
                    
            except requests.exceptions.Timeout:
                logger.error(f"Timeout sur l'API NASA pour ({lat}, {lon}) - tentative {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                continue
                
            except requests.exceptions.ConnectionError:
                logger.error(f"Erreur de connexion à l'API NASA pour ({lat}, {lon})")
                if attempt < max_retries - 1:
                    time.sleep(10)
                continue
                
            except Exception as e:
                logger.error(f"Erreur inattendue pour ({lat}, {lon}): {str(e)}")
                break
        
        logger.error(f"Échec de récupération des données après {max_retries} tentatives pour ({lat}, {lon})")
        return None
    
    @classmethod
    def _process_api_response(cls, data: Dict, parameters: List[str]) -> Optional[pd.DataFrame]:
        """
        Traite la réponse de l'API et la transforme en DataFrame.
        
        Args:
            data: Données brutes de l'API
            parameters: Liste des paramètres
            
        Returns:
            DataFrame structuré
        """
        try:
            raw_data = data['properties']['parameter']
            dates = list(raw_data[parameters[0]].keys())
            
            # Création du DataFrame
            records = []
            for date_str in dates:
                record = {'date': date_str}
                for param in parameters:
                    if param in raw_data and date_str in raw_data[param]:
                        record[param] = raw_data[param][date_str]
                    else:
                        record[param] = np.nan
                records.append(record)
            
            df = pd.DataFrame(records)
            
            # Conversion des types
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            
            # Tri par date
            df = df.sort_values('date').reset_index(drop=True)
            
            # Vérification des données manquantes
            missing_percentage = df.isnull().mean() * 100
            
            for col, percent in missing_percentage.items():
                if percent > 20:  # Plus de 20% de données manquantes
                    logger.warning(f"Colonne {col}: {percent:.1f}% de données manquantes")
            
            # Interpolation des valeurs manquantes (linéaire pour la météo)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(
                method='linear', 
                limit_direction='both'
            )
            
            # Vérification finale
            if df.isnull().any().any():
                logger.warning("Certaines valeurs sont toujours manquantes après interpolation")
                # Remplissage avec la moyenne
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la réponse API: {str(e)}")
            return None
    
    @classmethod
    def get_weather_stats(cls, lat: float, lon: float, days: int = 35) -> Dict:
        """
        Calcule des statistiques sur les données météo récentes.
        
        Args:
            lat: Latitude
            lon: Longitude
            days: Nombre de jours
            
        Returns:
            Dictionnaire de statistiques
        """
        df = cls.get_last_n_days(lat, lon, days)
        
        if df is None or df.empty:
            return {"error": "Données non disponibles"}
        
        stats = {
            "location": {"lat": lat, "lon": lon},
            "period": f"Last {days} days",
            "data_points": len(df),
            "date_range": {
                "start": df['date'].min().strftime('%Y-%m-%d'),
                "end": df['date'].max().strftime('%Y-%m-%d')
            }
        }
        
        # Statistiques par paramètre
        for param in cls.DEFAULT_PARAMETERS:
            if param in df.columns:
                stats[param] = {
                    "min": float(df[param].min()),
                    "max": float(df[param].max()),
                    "mean": float(df[param].mean()),
                    "std": float(df[param].std()),
                    "last_value": float(df[param].iloc[-1])
                }
        
        # Statistiques de précipitation cumulées
        if 'PRECTOTCORR' in df.columns:
            stats['precipitation_cumulative'] = {
                "last_7d": float(df['PRECTOTCORR'].tail(7).sum()),
                "last_14d": float(df['PRECTOTCORR'].tail(14).sum()),
                "last_30d": float(df['PRECTOTCORR'].tail(30).sum()),
                "total_period": float(df['PRECTOTCORR'].sum())
            }
        
        return stats
    
    @classmethod
    async def get_last_35_days_async(cls, lat: float, lon: float) -> Optional[pd.DataFrame]:
        """
        Version asynchrone pour récupérer les données météo.
        À utiliser avec asyncio dans les services asynchrones.
        """
        # Note: requests n'est pas asynchrone, on utilise run_in_executor
        import asyncio
        loop = asyncio.get_event_loop()
        
        try:
            df = await loop.run_in_executor(
                None, 
                lambda: cls.get_last_n_days(lat, lon, 35)
            )
            return df
        except Exception as e:
            logger.error(f"Erreur asynchrone pour ({lat}, {lon}): {str(e)}")
            return None

# Fonction de compatibilité pour l'ancien code
def get_last_35_days(lat: float, lon: float) -> Optional[pd.DataFrame]:
    """Fonction wrapper pour la compatibilité avec l'ancien code."""
    return WeatherService.get_last_n_days(lat, lon, 35)