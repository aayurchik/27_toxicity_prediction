import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select, delete
from app.database.models import History


class HistoryService:
    """Сервис для работы с историей запросов"""
    
    @staticmethod
    def save(
        db: Session,
        endpoint: str,
        request_body: Optional[Dict],
        response_body: Optional[Any],
        code: int
    ):
        """Сохранение записи в историю"""
        record = History(
            endpoint=endpoint,
            request_body=json.dumps(request_body, ensure_ascii=False) if request_body else None,
            response_body=json.dumps(response_body, ensure_ascii=False) if response_body else None,
            code=code
        )
        db.add(record)
        db.commit()
    
    @staticmethod
    def get_all(db: Session) -> List[History]:
        """Получение всей истории"""
        result = db.execute(
            select(History).order_by(History.time.desc())
        )
        return result.scalars().all()
    
    @staticmethod
    def clear_all(db: Session) -> int:
        """Очистка истории"""
        result = db.execute(delete(History))
        db.commit()
        return result.rowcount