from fastapi import FastAPI, HTTPException, Request, status,Header
from fastapi.responses import JSONResponse
from typing import Any, Optional
import json
import numpy as np
from typing import Dict, List
from fastapi import Body
import pickle
import json
from pathlib import Path
import numpy as np

from typing import AsyncGenerator, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import String, Integer, Text, select, delete, text
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import Column, Integer, String, Text, DateTime
from datetime import datetime
import time


class Base(DeclarativeBase):
    pass


# Инициализация приложения FastAPI
app = FastAPI(
    title="FastAPI Toxicity Prediction",
    description="Prediction of toxicity of chemical compounds based on their physicochemical properties",
    version="1.0.0"
)
class Model:
    def  __init__(self, model_path: str = "trained_models/knn_neuro_sensory_toxicity.pkl"):
        self.path = Path(model_path)
        self.model = None
        self.metadata = None
        self.load_model()
    
    def load_model(self):
        try:
            with open(self.path, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            raise
        except Exception as e:
            raise

    def predict(self, data: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Модель не загружена")
        try:
            return self.model.predict(data)
        except Exception as e:
            raise
async def save_history(
    db: AsyncSession,
    endpoint: str,
    request_body: Optional[dict],
    response_body: Optional[dict],
    code: int
):
    record = History(
        endpoint=endpoint,
        request_body=json.dumps(request_body, ensure_ascii=False) if request_body else None,
        response_body=json.dumps(response_body, ensure_ascii=False) if response_body else None,
        code=code
    )
    db.add(record)
    await db.commit()
class Features:
    def json_parse(data):
        # data = json.loads(json_line)
        smile = data.get('smiles')
        features = []
        for i in range(1,len(data)):  
            feature_key = f'feature{i}'
            if feature_key in data:
                features.append(data[feature_key])
            else:
                features.append(0.0)
        return smile, np.array([features])
    
# DATABASE SETUP / НАСТРОЙКА БАЗЫ ДАННЫХ

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = f"sqlite+aiosqlite:///{BASE_DIR}/test.db"

engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Log all SQL queries / Логировать все SQL запросы
    future=True  # Use SQLAlchemy 2.0 style / Использовать стиль 2.0
)

# Create async session factory / Фабрика асинхронных сессий
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit / Не истекать объекты после commit
    autocommit=False,
    autoflush=False
)
class History(Base):
    __tablename__ = "History"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True
    )

    time: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
        index=True
    )

    endpoint: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True
    )

    request_body: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True  
    )
    response_body: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True  
    )
    code: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )


model = None

# History.__table__.create(bind=engine, checkfirst=True)

def get_model():
    global model
    if model is None:
        model = Model()
    return model


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session 

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to FastAPI Toxicity Prediction",
        "documentation": "/docs",
        "version": "1.0.0"
    }

@app.get("/health", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_check():
    # Используется для проверки работоспособности API
    return {
        "status": "healthy",
        "service": "FastAPI Toxicity Prediction"
    }

# Pydantic-модель для входных данных
class ForwardItem(BaseModel):
    smiles: str
    features: Dict[str, float]  # все feature1, feature2, ..., featureN

# В FastAPI должен быть route типа POST на /forward, который должен принимать один из двух форматов:
# Если данные не содержат изображений, то необходимо передавать входные данные в теле запроса в формате JSON
@app.post("/forward", status_code=status.HTTP_200_OK, tags=["Predict"])
async def forward(
    request_body: List[ForwardItem] = Body(...),
    db: AsyncSession = Depends(get_db)):
    start_time = time.time()  # измеряем время обработки запроса
    # Преобразуем Pydantic-модель в обычный список словарей
    # Каждый словарь содержит smiles + все фичи
    body = []
    for item in request_body:
        d = {"smiles": item.smiles}
        d.update(item.features)
        body.append(d)
    # Если запрос неверного формата, то должен возвращаться код ошибки 400 с текстом ‘bad request’
    if not body:
        await save_history(
            db=db,
            endpoint="/forward",
            request_body=None,
            response_body=None,
            code=400
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="bad request")
    
    model = get_model()
    
    try:
        results = []
        # Прогоняем все данные через модель
        # Для каждого элемента создаём результат с smiles и toxity
        for j in body:
            smile, input_data = Features.json_parse(j)
            prediction = model.predict(input_data)
            results.append({
                "smile": smile,
                "toxity": int(prediction[0])
            })
        
        processing_time = time.time() - start_time  # измеряем фактическое время обработки
        await save_history(
            db=db,
            endpoint="/forward",
            request_body={"data": body, "processing_time": processing_time},
            response_body=results,
            code=200
        )
        # Если модель отработала успешно, возвращаем результаты в формате JSON
        return JSONResponse(content=results)
    
    # Если модель не смогла выполнить работу, то необходимо вернуть код ошибки 403 и вернуть сообщение: “модель не смогла обработать данные”
    except Exception:
        processing_time = time.time() - start_time
        await save_history(
            db=db,
            endpoint="/forward",
            request_body={"data": body, "processing_time": processing_time},
            response_body=None,
            code=status.HTTP_403_FORBIDDEN
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='модель не смогла обработать данные')

# Реализуйте GET-запрос /history, в котором будет показываться история всех запросов. История всех запросов должна находиться в базе данных.   
@app.get("/history", status_code=status.HTTP_200_OK, tags=["History"])
async def history(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(History).order_by(History.time.desc())
    )
    events = result.scalars().all()
    return [
        {
            "id": event.id,
            "time": event.time,
            "endpoint": event.endpoint,
            "request_body": event.request_body,
            "response_body": event.response_body,
            "code": event.code
        }
        for event in events
    ]

@app.get("/stats", status_code=status.HTTP_200_OK, tags=["Stats"])
async def stats(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(History))
    records = result.scalars().all()

    processing_times = []
    message_lengths = []
    token_counts = []

    for record in records:
        if record.request_body is None:
            continue

        try:
            body = json.loads(record.request_body)

            # время обработки
            if "processing_time" in body:
                processing_times.append(body["processing_time"])

            # длина сообщения
            message = json.dumps(body.get("data", ""))
            message_lengths.append(len(message))

            # количество "токенов"
            token_counts.append(len(message.split()))

        except Exception:
            continue

# (5 баллов) Реализуйте запрос /stats, который возвращает статистику запросов:
# среднее время обработки, квантили распределения (mean/50%/95%/99%)
# характеристики входных запросов (длина сообщения/количество токенов, если работаем с текстом; размеры изображений)

    def safe_percentile(data, q):
        return float(np.percentile(data, q)) if data else 0.0

    return {
        "processing_time": {
            "mean": float(np.mean(processing_times)) if processing_times else 0.0,
            "p50": safe_percentile(processing_times, 50),
            "p95": safe_percentile(processing_times, 95),
            "p99": safe_percentile(processing_times, 99),
        },
        "input_stats": {
            "avg_message_length": float(np.mean(message_lengths)) if message_lengths else 0.0,
            "avg_token_count": float(np.mean(token_counts)) if token_counts else 0.0,
        },
        "total_requests": len(records)
    }

       
                
