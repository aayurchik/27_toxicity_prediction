import json
import time
from typing import Optional
from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.services.history import HistoryService
from app.database.db import SessionLocal


class HistoryMiddleware(BaseHTTPMiddleware):
    """Middleware для логирования запросов"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_body = None
        
        # Читаем тело запроса для POST/PUT/PATCH
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                request_body = await request.json()
            except Exception:
                request_body = None
        
        response_body = None
        status_code = 500
        
        try:
            response = await call_next(request)
            status_code = getattr(response, "status_code", 200)
            
            # Извлекаем тело ответа
            if hasattr(response, "body") and response.body:
                try:
                    body_bytes = response.body
                    if isinstance(body_bytes, bytes):
                        body_bytes = body_bytes.decode()
                    response_body = json.loads(body_bytes)
                except Exception:
                    response_body = str(getattr(response, "body", None))
                    
        except RequestValidationError as e:
            status_code = 400
            response_body = {"error": "bad request", "details": str(e)}
            response = JSONResponse(content=response_body, status_code=status_code)
            
        except Exception as e:
            status_code = 500
            response_body = {"error": str(e)}
            response = JSONResponse(content=response_body, status_code=status_code)
            
        finally:
            # Сохраняем в историю (синхронно)
            processing_time = time.time() - start_time
            try:
                db = SessionLocal()
                try:
                    log_data = {"processing_time": processing_time}
                    if request_body:
                        log_data["data"] = request_body
                    
                    HistoryService.save(
                        db=db,
                        endpoint=request.url.path,
                        request_body=log_data,
                        response_body=response_body,
                        code=status_code
                    )
                finally:
                    db.close()
            except Exception as e:
                # Логирование не должно ломать ответ
                print(f"Error saving history: {e}")
                pass
        
        return response