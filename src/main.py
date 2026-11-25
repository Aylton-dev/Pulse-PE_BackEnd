from fastapi import FastAPI
from src.service.user_facial import router as facial_router

app = FastAPI(title="PulsePE API")
app.include_router(facial_router)


@app.get("/")
def root():
    return {"status": "ok"}