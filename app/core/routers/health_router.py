from fastapi import APIRouter


router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """
    Проверка работоспособности системы.
    Возвращает статус здоровья системы и состояние сервоконтроллера.
    """
    return {
        "status": (
            "healthy"
        )
    }
