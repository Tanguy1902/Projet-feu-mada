from fastapi import FastAPI
from app.api.router import router as api_router

app = FastAPI(
    title="Madagascar Fire Watch API",
    description="Système d'alerte précoce contre les feux de brousse"
)

# Inclusion du routeur avec versioning
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"message": "Mada Fire Watch Backend is running"}