import pandas as pd
import requests
from datetime import datetime, timedelta

def fetch_nasa_weather_realtime(lat, lon):
    """
    Récupère les 35 derniers jours de météo pour calculer les moyennes glissantes.
    """
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=35)).strftime('%Y%m%d')
    
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M_MAX,RH2M,PRECTOTCORR,WS10M_MAX&community=AG&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['properties']['parameter']
        df = pd.DataFrame(data)
        df.index.name = 'date'
        df = df.reset_index()
        return df
    return None

def prepare_features_for_scan(df, lat, lon):
    """
    Calcule les variables rain_7d, rain_30d, temp_7d et month 
    pour correspondre au modèle à 98%.
    """
    df = df.sort_values('date')
    
    # Calcul des lags (fenêtres glissantes)
    df['rain_7d'] = df['PRECTOTCORR'].rolling(7).sum()
    df['rain_30d'] = df['PRECTOTCORR'].rolling(30).sum()
    df['temp_7d'] = df['T2M_MAX'].rolling(7).mean()
    df['month'] = pd.to_datetime(df['date']).dt.month
    
    # On ne garde que la ligne la plus récente (aujourd'hui)
    latest = df.iloc[-1:].copy()
    latest['grid_lat'] = lat
    latest['grid_lon'] = lon
    
    return latest[['grid_lat', 'grid_lon', 'T2M_MAX', 'RH2M', 'PRECTOTCORR', 
                   'WS10M_MAX', 'rain_7d', 'rain_30d', 'temp_7d', 'month']]