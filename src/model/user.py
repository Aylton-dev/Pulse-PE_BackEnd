import uuid
from datetime import datetime
from pgvector.sqlalchemy import VECTOR
from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class FacialTemplate(Base):
    __tablename__ = "facial_templates"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )
    embedding: Mapped[list[float]] = mapped_column(VECTOR(512), nullable=False)
    model_version: Mapped[str] = mapped_column(
        String(50), nullable=False, default="facenet512_v1"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )