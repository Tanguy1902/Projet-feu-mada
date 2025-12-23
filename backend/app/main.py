# app/main.py
from fastapi import FastAPI, BackgroundTasks
from app.services.prediction_service import PredictionService
from app.services.weather_service import WeatherService
from app.database import SessionLocal
from app.database import engine, Base
import app.models as models
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import router as api_router
import asyncio
import pandas as pd

app = FastAPI(title="Mada Fire Detection API")

models.Base.metadata.create_all(bind=engine)

# Configuration CORS pour Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # URL de votre frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion des routes
app.include_router(api_router, prefix="/api/v1")

# Initialisation des services
predictor = PredictionService("model/model_expert_98.json")
weather_provider = WeatherService()

async def run_global_scan():
    while True:
        print("--- Début du scan national (Madagascar) ---")
        db = SessionLocal()
        try:
            # 1. Charger votre grille de coordonnées
            grille_df = pd.read_csv("dataset/weather_master_grille.csv")
            
            # 2. On récupère les données météo pour chaque point
            all_weather_data = []
            for _, row in grille_df.iterrows():
                # Appel à weather_service (NASA)
                data = await weather_provider.get_weather_data(row['grid_lat'], row['grid_lon'])
                all_weather_data.append(data)
            
            # 3. Lancer l'IA et sauvegarder si > 0.8
            alerts = predictor.run_inference_and_save(all_weather_data, db)
            
            if alerts:
                print(f"Succès : {len(alerts)} alertes critiques enregistrées.")
                
        except Exception as e:
            print(f"Erreur lors du scan : {e}")
        finally:
            db.close()
        
        # Attendre 1 heure (3600 secondes)
        await asyncio.sleep(3600)

@app.on_event("startup")
async def startup_event():
    # Lance le scan automatique dès que le serveur démarre
    asyncio.create_task(run_global_scan())

async def save_alerts_to_db(alerts_list):
    db = SessionLocal()
    for item in alerts_list:
        new_alert = FireAlert(
            latitude=item['grid_lat'],
            longitude=item['grid_lon'],
            risk_score=item['risk_score'],
            temp_max=item['T2M_MAX'],
            humidity=item['RH2M'],
            precipitations=item['PRECTOTCORR']
        )
        db.add(new_alert)
    db.commit()
    db.close()

@app.get("/")
def read_root():
    return {"status": "Système de surveillance actif"}