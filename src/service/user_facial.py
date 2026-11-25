from dataclasses import dataclass
import cv2
from deepface import DeepFace
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
import numpy as np
from sqlalchemy.orm import Session

from src.repository.database import FacialRepository, get_db

router = APIRouter(prefix="/facial", tags=["Reconhecimento Facial"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024
MODEL_VERSION = "facenet512_v1"
DUPLICATE_THRESHOLD = 0.20  # calibre com amostras reais do Facenet512
VERIFY_THRESHOLD = 0.40     # calibre com amostras reais do Facenet512


@dataclass(frozen=True)
class FacialValidationError(Exception):
    message: str
    status_code: int = 422


def _cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return max(0.0, min(1.0, 1.0 - similarity))


class FacialService:
    def __init__(
        self,
        model_name: str = "Facenet512",
        detector_backend: str = "opencv",
        min_face_size: int = 80,
        min_blur_score: float = 40.0,
        min_brightness: float = 45.0,
        max_brightness: float = 220.0,
    ) -> None:
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.min_face_size = min_face_size
        self.min_blur_score = min_blur_score
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    def extract_embedding(self, image_bytes: bytes, check_quality: bool = True) -> list[float]:
        if not image_bytes:
            raise FacialValidationError("Arquivo de imagem vazio.", status_code=400)

        bgr_img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise FacialValidationError("Imagem inválida ou corrompida.", status_code=400)

        try:
            results = DeepFace.represent(
                img_path=bgr_img,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
            )
        except ValueError as exc:
            raise FacialValidationError("Nenhum rosto detectado na imagem.") from exc

        if not results:
            raise FacialValidationError("Nenhum rosto detectado na imagem.")
        if len(results) > 1:
            raise FacialValidationError("Mais de um rosto detectado. Envie uma foto individual.")

        face_data = results[0]
        area = face_data.get("facial_area", {})
        x = int(area.get("x", 0))
        y = int(area.get("y", 0))
        w = int(area.get("w", 0))
        h = int(area.get("h", 0))

        if check_quality:
            if w < self.min_face_size or h < self.min_face_size:
                raise FacialValidationError("Rosto muito pequeno na imagem.")

            crop = bgr_img[max(0, y) : max(0, y + h), max(0, x) : max(0, x + w)]
            if crop.size == 0:
                raise FacialValidationError("Recorte facial inválido na imagem.")

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < self.min_blur_score:
                raise FacialValidationError("Imagem desfocada.")

            mean_brightness = float(np.mean(gray))
            if mean_brightness < self.min_brightness or mean_brightness > self.max_brightness:
                raise FacialValidationError("Iluminação inadequada na imagem.")

        embedding = face_data.get("embedding")
        if not embedding:
            raise FacialValidationError("Não foi possível extrair o embedding facial.")

        return [float(val) for val in embedding]

    def min_distance_to(
        self,
        candidate_embedding: list[float],
        existing_embeddings: list[list[float]],
    ) -> float | None:
        if not existing_embeddings:
            return None

        candidate = np.asarray(candidate_embedding, dtype=np.float64)
        distances = [
            _cosine_distance(candidate, np.asarray(emb, dtype=np.float64))
            for emb in existing_embeddings
        ]
        return float(min(distances))


def get_facial_service() -> FacialService:
    return FacialService()


def get_current_user_id() -> str:
   
    return "00000000-0000-0000-0000-000000000001"

async def _read_and_extract_embedding(
    file: UploadFile,
    facial_service: FacialService,
    check_quality: bool = True,
) -> list[float]:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato inválido. Envie JPEG ou PNG.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Imagem acima de 5 MB.",
        )

    try:
        return facial_service.extract_embedding(image_bytes, check_quality=check_quality)
    except FacialValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)

@router.post("/enroll", status_code=status.HTTP_201_CREATED)
async def enroll_facial(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
    facial_service: FacialService = Depends(get_facial_service),
):
    embedding = await _read_and_extract_embedding(file, facial_service, check_quality=True)

    repo = FacialRepository(db)
    existing_embeddings = repo.get_user_embeddings(
        user_id=current_user_id,
        model_version=MODEL_VERSION,
    )

    min_dist = facial_service.min_distance_to(embedding, existing_embeddings)
    if min_dist is not None and min_dist <= DUPLICATE_THRESHOLD:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Essa biometria facial já está cadastrada (distância: {min_dist:.4f}).",
        )

    saved = repo.save_template(
        user_id=current_user_id,
        embedding=embedding,
        model_version=MODEL_VERSION,
        max_templates=3,
    )

    return {
        "message": "Biometria facial cadastrada com sucesso.",
        "template_id": str(saved.id),
        "model_version": saved.model_version,
    }


@router.post("/verify")
async def verify_facial(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
    facial_service: FacialService = Depends(get_facial_service),
):
    embedding = await _read_and_extract_embedding(file, facial_service, check_quality=True)

    repo = FacialRepository(db)
    existing_embeddings = repo.get_user_embeddings(
        user_id=current_user_id,
        model_version=MODEL_VERSION,
    )

    if not existing_embeddings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nenhuma biometria facial cadastrada para este usuário.",
        )

    min_dist = facial_service.min_distance_to(embedding, existing_embeddings)
    matched = min_dist is not None and min_dist <= VERIFY_THRESHOLD

    return {
        "matched": matched,
        "distance": round(min_dist, 4) if min_dist is not None else None,
        "threshold": VERIFY_THRESHOLD,
        "model_version": MODEL_VERSION,
    }