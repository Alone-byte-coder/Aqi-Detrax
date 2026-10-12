
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aqi_agent import generate_report

app = FastAPI(
    title="Location-Aware AQI AI Agent",
    description=(
        "Provides current air-quality advice and future precautions "
        "using existing ML outputs and internal location context."
    )
)


class AgentRequest(BaseModel):
    question: str = Field(
        default=(
            "Explain current air quality, current precautions, "
            "future predictions, future risk and precautions."
        ),
        min_length=3,
        max_length=1000
    )


@app.get("/")
def home():
    return {
        "status": "running",
        "message": "AQI AI Agent is ready"
    }


@app.post("/analyze")
def analyze(request: AgentRequest):
    try:
        return generate_report(request.question)


    except Exception as error:
        print("Agent error:", repr(error))

        raise HTTPException(
            status_code=500,
            detail=(
                "Agent failed. Check the sensor backend API, "
                "Ollama, the model installation, and GeoJSON map files."
            )
        ) from error
