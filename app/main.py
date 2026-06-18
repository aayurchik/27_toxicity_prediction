from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, Header, Depends
from fastapi.responses import JSONResponse
import numpy as np
import json

from app.config import settings
from app.models.tox_model import ToxModel
from app.schemas.requests import ForwardItem, HealthResponse
from app.database.db import SessionLocal, init_db, close_db
from app.database.models import History
from app.services.prediction import PredictionService
from app.services.history import HistoryService
from app.middleware.history import HistoryMiddleware


# Глобальные переменные
model = None
prediction_service = None


def get_model() -> ToxModel:
    """Получение экземпляра модели (синглтон)"""
    global model
    if model is None:
        model = ToxModel()
    return model


def get_prediction_service() -> PredictionService:
    """Получение сервиса предсказаний"""
    global prediction_service
    if prediction_service is None:
        prediction_service = PredictionService(get_model())
    return prediction_service


def get_db():
    """Получение сессии БД (синхронно)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    # Старт
    init_db()  # Синхронный вызов
    get_model()  # Инициализируем модель
    yield
    # Завершение
    close_db()


# Создаем приложение
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# Добавляем middleware
app.add_middleware(HistoryMiddleware)


# ================= ЭНДПОИНТЫ =================

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to FastAPI Toxicity Prediction",
        "documentation": "/docs",
        "version": settings.API_VERSION
    }


@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    """Проверка работоспособности"""
    return HealthResponse(
        status="healthy",
        service="FastAPI Toxicity Prediction",
        model_loaded=get_model().is_loaded
    )


@app.post("/forward", status_code=status.HTTP_200_OK, tags=["Predict"])
async def forward(
    request_body: list[ForwardItem],
    db: SessionLocal = Depends(get_db)
):
    """
    Предсказание токсичности для списка молекул
    """
    if not request_body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="bad request: пустой массив данных"
        )
    
    # Проверяем валидность данных
    for item in request_body:
        if not item.smiles:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="bad request: smiles не может быть пустым"
            )
        if len(item.features) != 78:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"bad request: требуется 78 признаков, получено {len(item.features)}"
            )
    
    try:
        service = get_prediction_service()
        results = service.predict_batch(request_body)
        return JSONResponse(content=[r.model_dump() for r in results])
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"модель не смогла обработать данные: {str(e)}"
        )


@app.get("/history", status_code=status.HTTP_200_OK, tags=["History"])
async def get_history(db: SessionLocal = Depends(get_db)):
    """Получение истории всех запросов"""
    events = HistoryService.get_all(db)
    return [
        {
            "id": event.id,
            "time": event.time.isoformat(),
            "endpoint": event.endpoint,
            "request_body": event.request_body,
            "response_body": event.response_body,
            "code": event.code
        }
        for event in events
    ]


@app.get("/stats", status_code=status.HTTP_200_OK, tags=["Stats"])
async def get_stats(db: SessionLocal = Depends(get_db)):
    """Статистика по запросам"""
    events = HistoryService.get_all(db)
    
    times = []
    msg_lengths = []
    token_counts = []
    
    for event in events:
        if not event.request_body:
            continue
        try:
            body = json.loads(event.request_body)
            if "processing_time" in body:
                times.append(body["processing_time"])
            
            msg = json.dumps(body.get("data", ""))
            msg_lengths.append(len(msg))
            token_counts.append(len(msg.split()))
        except Exception:
            continue
    
    def percentile_safe(data, p):
        return float(np.percentile(data, p)) if data else 0.0
    
    return {
        "processing_time": {
            "mean": float(np.mean(times)) if times else 0.0,
            "p50": percentile_safe(times, 50),
            "p95": percentile_safe(times, 95),
            "p99": percentile_safe(times, 99),
        },
        "input_stats": {
            "average_message_length": float(np.mean(msg_lengths)) if msg_lengths else 0.0,
            "average_token_count": float(np.mean(token_counts)) if token_counts else 0.0,
        },
        "total_requests": len(events)
    }


@app.delete("/history", status_code=status.HTTP_200_OK, tags=["History"])
async def delete_history(
    token: str = Header(..., description="Токен администратора"),
    db: SessionLocal = Depends(get_db)
):
    """Удаление всей истории запросов (требуется токен администратора)"""
    if token != settings.ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: invalid token"
        )
    
    try:
        deleted_count = HistoryService.clear_all(db)
        return {"message": f"История запросов успешно очищена. Удалено {deleted_count} записей."}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка удаления истории: {str(e)}"
        )