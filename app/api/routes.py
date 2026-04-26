from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pathlib import Path
import tempfile
import zipfile
import uuid
import shutil
import cv2

# Импорты
from app.pipeline.anpr_pipeline import run_anpr
from app.services.detector.yolo_detector import detect_plate
from app.services.preprocess.variants import generate_variants
from app.services.ocr.ensemble import read_plate_ensemble
from app.services.postprocess.plate_rules import normalize_plate

router = APIRouter()


# ====================== ОДИНОЧНЫЙ РЕЖИМ (твой первый) ======================
@router.post("/anpr/read")
async def read_plate(file: UploadFile = File(...)):
    result = await run_anpr(file)
    return result


# ====================== МАССОВАЯ РАЗМЕТКА (ZIP) ======================
def cleanup_temp_dir(tmp_dir: Path):
    """Очистка после отправки файла"""
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception as e:
        print(f"Не удалось очистить временную папку: {e}")


@router.post("/anpr/label-dataset")
async def label_dataset(
    archive: UploadFile = File(..., description="Загрузи ZIP с фотографиями машин"),
    background_tasks: BackgroundTasks = None,
):
    if not archive.filename.lower().endswith(".zip"):
        raise HTTPException(400, detail="Файл должен быть .zip архивом")

    if archive.size and archive.size > 800_000_000:  # 800 МБ
        raise HTTPException(413, detail="Архив слишком большой (макс. 800 МБ)")

    tmp_dir = Path(tempfile.mkdtemp())
    extract_dir = tmp_dir / "extracted"
    crops_dir = tmp_dir / "plates"
    extract_dir.mkdir(parents=True)
    crops_dir.mkdir(parents=True)

    result_zip = tmp_dir / "labeled_plates.zip"

    try:
        # Сохраняем zip
        zip_path = tmp_dir / f"upload_{uuid.uuid4().hex}.zip"
        with open(zip_path, "wb") as f:
            shutil.copyfileobj(archive.file, f)

        # Распаковка
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        labels = []
        processed = 0
        skipped = 0

        for img_path in extract_dir.rglob("*.*"):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                continue

            try:
                plate_crop, _ = detect_plate(str(img_path))
                if plate_crop is None:
                    skipped += 1
                    continue

                variants = generate_variants(plate_crop)
                if not variants:
                    skipped += 1
                    continue

                raw_text = read_plate_ensemble(variants)
                if not raw_text:
                    skipped += 1
                    continue

                final_text = normalize_plate(raw_text)
                if not final_text:
                    skipped += 1
                    continue

                fname = f"{final_text}_{uuid.uuid4().hex[:8]}.jpg"
                cv2.imwrite(str(crops_dir / fname), plate_crop)
                labels.append(f"{fname} {final_text}")
                processed += 1

            except Exception as e:
                print(f"Ошибка обработки {img_path.name}: {e}")
                skipped += 1
                continue

        if not labels:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return {
                "status": "error",
                "message": "Не удалось обнаружить ни одного номера",
                "processed": 0,
                "skipped": skipped
            }

        # labels.txt
        (crops_dir / "labels.txt").write_text("\n".join(labels), encoding="utf-8")

        # Финальный zip
        with zipfile.ZipFile(result_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in crops_dir.iterdir():
                zf.write(file, file.name)

        if background_tasks:
            background_tasks.add_task(cleanup_temp_dir, tmp_dir)

        return FileResponse(
            path=str(result_zip),
            filename="labeled_plates.zip",
            media_type="application/zip"
        )

    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, detail=f"Ошибка: {str(e)}")