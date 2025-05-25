from pydantic import BaseModel

class WeatherInput(BaseModel):
    temperature: float
    pressure: float
    humidity: float
    wind_sped: float

class PredictionOutput(BaseModel):
    energy: float