import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from intent_router.model import IntentRouter


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=10)


def create_app(router=None):
    @asynccontextmanager
    async def lifespan(app):
        path = Path(os.environ.get("INTENT_ROUTER_MODEL", "artifacts/router.joblib"))
        app.state.router = (
            router
            if router is not None
            else IntentRouter.load(path)
            if path.exists()
            else None
        )
        yield

    app = FastAPI(title="Banking Intent Router", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_ready": app.state.router is not None}

    @app.get("/ready")
    def ready():
        if app.state.router is None:
            raise HTTPException(503, "Train a router before serving requests.")
        return {"status": "ready"}

    @app.post("/predict")
    def predict(request: Request):
        if app.state.router is None:
            raise HTTPException(503, "No model loaded.")
        try:
            return app.state.router.predict(request.text, request.top_k)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
