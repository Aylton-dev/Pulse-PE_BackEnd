"""endpoints"""
from fastapi import FastAPI, Depends, HTTPException, status

app = FastAPI(title = "PulsePE")


@app.get("/")
def status():
    return {"status": "API is running", "project": "PulsePE"}