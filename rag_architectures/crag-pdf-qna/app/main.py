from fastapi import FastAPI

import initiate_logger
from app.routers import health, ingestion, query

# Importing the routers above pulls in RapidOCR, which forces its own logger to
# INFO at import time. Re-apply the silencing now so it actually takes effect.
initiate_logger.silence_noisy_loggers()

app = FastAPI(title="CRAG PDF QnA", version="0.1.0")

app.include_router(health.router)
app.include_router(ingestion.router)
app.include_router(query.router)
