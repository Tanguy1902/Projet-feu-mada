# app/models.py
from sqlalchemy import Column, Integer, Float, DateTime, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class FireAlert(Base):
    __tablename__ = "fire_alerts"

    id = Column(Integer, primary_key=True, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False) # Ex: 0.82
    temp_max = Column(Float)
    humidity = Column(Float)
    precipitations = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    is_confirmed = Column(Boolean, default=False) # Pour validation manuelle