# app/services/prediction_service.py
import xgboost as xgb
import pandas as pd
import os

class PredictionService:
    def __init__(self, model_path: str):
        # Initialisation du Booster XGBoost
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        self.threshold = 0.8  # Votre seuil de validation

    def predict(self, data_list: list):
        """
        Prend une liste de dictionnaires (données météo) 
        et retourne les zones où le risque >= 0.8
        """
        if not data_list:
            return []

        df = pd.DataFrame(data_list)
        
        # L'ordre doit correspondre exactement au JSON :
        # ["grid_lat", "grid_lon", "T2M_MAX", "RH2M", "PRECTOTCORR", 
        #  "WS10M_MAX", "rain_7d", "rain_30d", "temp_7d", "month"]
        features = ["grid_lat", "grid_lon", "T2M_MAX", "RH2M", "PRECTOTCORR", 
                    "WS10M_MAX", "rain_7d", "rain_30d", "temp_7d", "month"]
        
        dmatrix = xgb.DMatrix(df[features])
        predictions = self.model.predict(dmatrix)
        
        df['risk_score'] = predictions
        
        # Filtrage selon votre critère de 0.8
        high_risk = df[df['risk_score'] >= self.threshold]
        
        return high_risk.to_dict(orient="records")