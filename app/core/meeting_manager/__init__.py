"""
Модуль для управления встречами с людьми.
Отслеживает известные и неизвестные лица, отправляет уведомления в LLM.
"""

from app.core.meeting_manager.manager import MeetingManager

__all__ = ["MeetingManager"]
