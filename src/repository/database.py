import os
import uuid
from dotenv import load_dotenv
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from src.model.user import FacialTemplate

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class FacialRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_user_embeddings(
        self,
        user_id: str,
        model_version: str = "facenet512_v1",
    ) -> list[list[float]]:
        uid = uuid.UUID(user_id)
        stmt = select(FacialTemplate.embedding).where(
            FacialTemplate.user_id == uid,
            FacialTemplate.model_version == model_version,
        )
        rows = self.db.scalars(stmt).all()
        return [list(emb) for emb in rows]

    def save_template(
        self,
        user_id: str,
        embedding: list[float],
        model_version: str = "facenet512_v1",
        max_templates: int = 3,
    ) -> FacialTemplate:
        uid = uuid.UUID(user_id)

        existing = self.db.scalars(
            select(FacialTemplate)
            .where(
                FacialTemplate.user_id == uid,
                FacialTemplate.model_version == model_version,
            )
            .order_by(FacialTemplate.created_at.asc())
        ).all()

        while len(existing) >= max_templates:
            oldest = existing.pop(0)
            self.db.delete(oldest)

        template = FacialTemplate(
            user_id=uid,
            embedding=embedding,
            model_version=model_version,
        )
        self.db.add(template)
        self.db.commit()
        self.db.refresh(template)
        return template