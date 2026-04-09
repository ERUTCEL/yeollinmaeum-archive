#!/usr/bin/env python3
"""
열린마음 아카이브 - 1단계: 이미지 전처리
-----------------------------------------
소스 폴더의 중첩 구조를 탐색하여 input/ 디렉토리로 정규화된 이름으로 복사.
HEIC 파일은 JPG로 변환. manifest.json(볼륨 목록)을 생성.

소스 구조 예:
  Desktop/열린마음/
    1991년 열린마음-TIMESTAMP/
      1991년 열린마음/
        1.jpg, 2.jpg, ...

출력:
  input/  → 1991열린마음_cover_front.jpg, 1991열린마음_p001.jpg, ...
  data/manifest.json → 볼륨 목록 및 메타데이터

사용법:
  python scripts/1_preprocess.py
  python scripts/1_preprocess.py --source /mnt/c/Users/최강연/Desktop/열린마음
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORTED = True
except ImportError:
    HEIF_SUPPORTED = False
    print("[경고] pillow-heif 미설치. HEIC 파일 변환 불가. 'pip install pillow-heif' 실행 필요.")

BASE_DIR = Path(__file__).parent.parent
DEFAULT_SOURCE = Path("/mnt/c/Users/최강연/Desktop/열린마음")
INPUT_DIR = BASE_DIR / "input"
DATA_DIR = BASE_DIR / "data"
MANIFEST_FILE = DATA_DIR / "manifest.json"

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def extract_year(name: str) -> int | None:
    """폴더명에서 연도 추출."""
    m = re.search(r"(19|20)\d{2}", name)
    return int(m.group()) if m else None


def make_volume_id(folder_name: str) -> str:
    """볼륨 ID 생성 (파일명 안전한 문자열)."""
    # 타임스탬프 제거
    name = re.sub(r"-\d{8}T\d{6}Z.*$", "", folder_name).strip()
    # 특수문자를 언더스코어로
    safe = re.sub(r"[^\w가-힣]", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe


def get_page_number(filename: str) -> tuple[int, int]:
    """파일명에서 페이지 정렬 키 반환 (cover_front=0, cover_back=999, 나머지 숫자)."""
    name = filename.lower()
    stem = Path(filename).stem.lower()

    if "앞표지" in stem or "front" in stem:
        return (0, 0)
    if "뒷표지" in stem or "back" in stem:
        return (999, 0)

    # 파일명에서 숫자 추출 (p.1, p1, 1, 95_1 등)
    numbers = re.findall(r"\d+", stem)
    if numbers:
        # 마지막 숫자를 페이지 번호로 사용
        return (1, int(numbers[-1]))
    return (500, 0)


def convert_to_jpg(src: Path, dst: Path):
    """이미지를 JPG로 변환하여 저장."""
    with Image.open(src) as img:
        # RGBA → RGB 변환 (HEIC가 RGBA일 수 있음)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 최대 2000px로 리사이즈 (품질 유지)
        max_size = 2000
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)

        img.save(dst, format="JPEG", quality=88, optimize=True)


def find_image_folder(top: Path) -> Path | None:
    """볼륨 최상위 폴더에서 실제 이미지가 있는 하위 폴더 탐색."""
    # 직접 이미지가 있으면 그 폴더
    images = [f for f in top.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
    if images:
        return top

    # 하위 폴더 탐색 (한 단계)
    for sub in sorted(top.iterdir()):
        if sub.is_dir():
            images = [f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
            if images:
                return sub
    return None


def process_source(source_dir: Path):
    """소스 폴더를 탐색하여 input/에 정규화된 이미지 복사."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not source_dir.exists():
        print(f"[오류] 소스 폴더를 찾을 수 없습니다: {source_dir}")
        return

    # 볼륨 폴더 목록 (연도 기준 정렬)
    volume_dirs = sorted(
        [d for d in source_dir.iterdir() if d.is_dir()],
        key=lambda d: (extract_year(d.name) or 9999, d.name)
    )

    if not volume_dirs:
        print(f"[오류] 소스 폴더에 하위 디렉토리가 없습니다: {source_dir}")
        return

    print(f"발견된 볼륨: {len(volume_dirs)}권")
    manifest = []
    total_copied = 0
    total_converted = 0

    for vol_dir in volume_dirs:
        vol_name = vol_dir.name
        vol_id = make_volume_id(vol_name)
        year = extract_year(vol_name)

        # 제목 추출 (연도와 타임스탬프 제거)
        title = re.sub(r"-\d{8}T\d{6}Z.*$", "", vol_name).strip()
        title = re.sub(r"^\d{4}년\s*", "", title).strip()
        if not title:
            title = f"{year}년 열린마음"

        img_folder = find_image_folder(vol_dir)
        if not img_folder:
            print(f"  [건너뜀] {vol_name} — 이미지 없음")
            continue

        image_files = sorted(
            [f for f in img_folder.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS],
            key=lambda f: get_page_number(f.name)
        )

        if not image_files:
            print(f"  [건너뜀] {vol_name} — 이미지 없음")
            continue

        print(f"\n  {vol_name}")
        print(f"    → {len(image_files)}장 처리 중...")

        volume_entry = {
            "id": vol_id,
            "year": year,
            "title": f"{year}년 {title}" if year else title,
            "short_title": title,
            "source_folder": vol_name,
            "pages": [],
        }

        for i, img_path in enumerate(tqdm(image_files, desc=f"    복사", unit="장", leave=False)):
            sort_key = get_page_number(img_path.name)

            # 출력 파일명 결정
            if sort_key[0] == 0:
                page_label = "cover_front"
            elif sort_key[0] == 999:
                page_label = "cover_back"
            else:
                page_label = f"p{i:03d}"

            out_name = f"{vol_id}_{page_label}.jpg"
            out_path = INPUT_DIR / out_name

            # 이미 변환된 파일 스킵
            if out_path.exists():
                volume_entry["pages"].append({
                    "file": out_name,
                    "label": page_label,
                    "original": img_path.name,
                })
                continue

            try:
                is_heic = img_path.suffix.lower() in (".heic", ".heif")

                if is_heic:
                    if not HEIF_SUPPORTED:
                        print(f"\n    [건너뜀] HEIC 지원 없음: {img_path.name}")
                        continue
                    convert_to_jpg(img_path, out_path)
                    total_converted += 1
                elif img_path.suffix.lower() not in (".jpg", ".jpeg"):
                    # PNG 등 → JPG 변환
                    convert_to_jpg(img_path, out_path)
                    total_converted += 1
                else:
                    # JPG는 그냥 복사 (리사이즈 필요시 변환)
                    with Image.open(img_path) as img:
                        if max(img.size) > 2000:
                            convert_to_jpg(img_path, out_path)
                            total_converted += 1
                        else:
                            shutil.copy2(img_path, out_path)
                            total_copied += 1

                volume_entry["pages"].append({
                    "file": out_name,
                    "label": page_label,
                    "original": img_path.name,
                })

            except Exception as e:
                print(f"\n    [오류] {img_path.name}: {e}")

        # 앞표지 파일 기록
        cover_file = next(
            (p["file"] for p in volume_entry["pages"] if p["label"] == "cover_front"),
            volume_entry["pages"][0]["file"] if volume_entry["pages"] else None
        )
        volume_entry["cover"] = cover_file
        volume_entry["page_count"] = len([p for p in volume_entry["pages"]
                                          if p["label"] not in ("cover_front", "cover_back")])
        manifest.append(volume_entry)

    # manifest.json 저장
    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    total_volumes = len(manifest)
    total_files = sum(len(v["pages"]) for v in manifest)
    print(f"\n전처리 완료!")
    print(f"  볼륨: {total_volumes}권")
    print(f"  이미지: {total_files}장 (복사 {total_copied}, 변환 {total_converted})")
    print(f"  manifest: {MANIFEST_FILE}")
    print(f"\n다음 단계: python process.py --mode batch  (또는 --mode live)")


def main():
    parser = argparse.ArgumentParser(description="열린마음 이미지 전처리")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"소스 폴더 경로 (기본: {DEFAULT_SOURCE})"
    )
    args = parser.parse_args()
    process_source(args.source)


if __name__ == "__main__":
    main()
