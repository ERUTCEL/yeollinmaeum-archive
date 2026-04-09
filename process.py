#!/usr/bin/env python3
"""
열린마음 아카이빙 프로세서
-------------------------------
손으로 쓴 노트 사진 500장을 Claude Vision API로 디지털화합니다.

사용법:
  python process.py --mode batch   # Batches API (50% 절감, 비동기, 권장)
  python process.py --mode live    # 실시간 처리 (진행 상황 즉시 확인)
  python process.py --mode check   # 배치 작업 완료 여부 확인 및 결과 수집
  python process.py --mode build   # 결과로부터 HTML 뷰어 생성
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import anthropic
from PIL import Image
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# ── 설정 ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
JSON_DIR = OUTPUT_DIR / "json"
BATCH_DIR = OUTPUT_DIR / "batch_results"
ARCHIVE_FILE = OUTPUT_DIR / "archive.json"
STATE_FILE = OUTPUT_DIR / ".state.json"   # 처리 완료된 파일 추적
BATCH_STATE_FILE = OUTPUT_DIR / ".batch_state.json"  # 배치 작업 ID 추적

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif"}
MODEL = "claude-haiku-4-5-20251001"
MANIFEST_FILE = BASE_DIR / "data" / "manifest.json"

EXTRACTION_PROMPT = """이 사진은 "열린마음"이라는 학과 방명록/일상 노트의 페이지입니다 (1990년대~2000년대).
손으로 쓴 한국어 텍스트를 분석하여 아래 JSON 형식으로만 응답해주세요. 다른 설명은 불필요합니다.

{
  "date": "작성 날짜 (예: 1995-03-15, 1998년 봄, null)",
  "author": "작성자 이름 또는 별명 (없으면 null)",
  "content": "손으로 쓴 텍스트 전체를 최대한 정확하게 옮긴 내용",
  "summary": "2-3문장 요약",
  "tags": ["일상", "고민", "졸업", "신입생", "행사", "연애", "공부", "MT" 등 해당하는 태그들],
  "mood": "positive 또는 neutral 또는 negative",
  "decade": "1990s 또는 2000s 또는 unknown",
  "image_quality": "good 또는 fair 또는 poor"
}

텍스트가 흐리거나 읽기 어려운 부분은 [불명확] 으로 표기하세요.
사진에 텍스트가 없거나 노트 페이지가 아닌 경우 content를 "텍스트 없음"으로 설정하세요."""


# ── 유틸리티 ─────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"processed": [], "errors": []}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_batch_state() -> dict:
    if BATCH_STATE_FILE.exists():
        return json.loads(BATCH_STATE_FILE.read_text(encoding="utf-8"))
    return {"batch_ids": [], "submitted_at": None}


def save_batch_state(state: dict):
    BATCH_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_image_files() -> list[Path]:
    files = sorted([
        p for p in INPUT_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTS
    ])
    return files


def load_manifest() -> dict:
    """manifest.json에서 볼륨 정보 로드. 파일명 → 볼륨 정보 맵 반환."""
    if not MANIFEST_FILE.exists():
        return {}
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    file_map = {}
    for vol in manifest:
        for page in vol.get("pages", []):
            file_map[page["file"]] = {
                "volume_id": vol["id"],
                "volume_title": vol["title"],
                "year": vol["year"],
                "label": page["label"],
            }
    return file_map


def encode_image(path: Path) -> tuple[str, str]:
    """이미지를 base64로 인코딩하고 미디어 타입 반환 (HEIC 포함)."""
    import io
    suffix = path.suffix.lower()

    # HEIC/HEIF는 항상 JPEG로 변환하여 전송
    is_heic = suffix in (".heic", ".heif")

    with Image.open(path) as img:
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")

        # 최대 1000px로 제한 (API 전송 크기 최소화)
        max_size = 1000
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.LANCZOS)

        if is_heic or max(img.size) > 1000:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            data = base64.standard_b64encode(buf.read()).decode("utf-8")
            return data, "image/jpeg"

    # 일반 이미지: 원본 그대로 전송
    media_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
    }
    media_type = media_map.get(suffix, "image/jpeg")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, media_type


_manifest_cache: dict | None = None

def get_manifest() -> dict:
    global _manifest_cache
    if _manifest_cache is None:
        _manifest_cache = load_manifest()
    return _manifest_cache


def parse_response(text: str, filename: str) -> dict:
    """Claude 응답에서 JSON 추출."""
    text = text.strip()
    # JSON 블록 추출 시도
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # JSON 파싱 실패시 원문 보존
        data = {
            "date": None,
            "author": None,
            "content": text,
            "summary": "파싱 오류 - 원문 보존",
            "tags": ["파싱오류"],
            "mood": "neutral",
            "decade": "unknown",
            "image_quality": "unknown",
        }

    data["filename"] = filename
    data["processed_at"] = datetime.now().isoformat()

    # 볼륨 메타데이터 병합
    vol_info = get_manifest().get(filename, {})
    if vol_info:
        data["volume_id"] = vol_info.get("volume_id")
        data["volume_title"] = vol_info.get("volume_title")
        data["year"] = vol_info.get("year")
        data["page_label"] = vol_info.get("label")

    return data


# ── 실시간 처리 모드 ──────────────────────────────────────────────────────────

def process_live(client: anthropic.Anthropic):
    """이미지를 하나씩 실시간으로 처리합니다."""
    files = get_image_files()
    if not files:
        print(f"[오류] {INPUT_DIR} 에 이미지 파일이 없습니다.")
        return

    state = load_state()
    processed_set = set(state["processed"])
    pending = [f for f in files if f.name not in processed_set]

    print(f"총 {len(files)}장 중 {len(pending)}장 처리 예정 "
          f"({len(processed_set)}장 이미 완료)")

    if not pending:
        print("모든 이미지가 이미 처리되었습니다. '--mode build' 로 뷰어를 생성하세요.")
        return

    JSON_DIR.mkdir(parents=True, exist_ok=True)

    for path in tqdm(pending, desc="처리 중", unit="장"):
        try:
            image_data, media_type = encode_image(path)

            response = client.messages.create(
                model=MODEL,
                max_tokens=2048,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {"type": "text", "text": EXTRACTION_PROMPT},
                    ],
                }],
            )

            # 텍스트 블록 추출
            text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            result = parse_response(text, path.name)

            # 개별 JSON 저장
            out_path = JSON_DIR / f"{path.stem}.json"
            out_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            state["processed"].append(path.name)
            save_state(state)

        except anthropic.RateLimitError:
            print(f"\n[율 제한] {path.name} - 60초 대기 후 재시도...")
            time.sleep(60)
            # 재시도
            try:
                image_data, media_type = encode_image(path)
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=2048,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                            {"type": "text", "text": EXTRACTION_PROMPT},
                        ],
                    }],
                )
                text = next((b.text for b in response.content if b.type == "text"), "")
                result = parse_response(text, path.name)
                out_path = JSON_DIR / f"{path.stem}.json"
                out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                state["processed"].append(path.name)
                save_state(state)
            except Exception as e:
                print(f"\n[오류] {path.name}: {e}")
                state["errors"].append({"file": path.name, "error": str(e)})
                save_state(state)

        except Exception as e:
            print(f"\n[오류] {path.name}: {e}")
            state["errors"].append({"file": path.name, "error": str(e)})
            save_state(state)

    print(f"\n완료: {len(state['processed'])}장 처리, {len(state['errors'])}장 오류")
    build_archive()


# ── 배치 처리 모드 ────────────────────────────────────────────────────────────

def submit_batch(client: anthropic.Anthropic):
    """Batches API로 모든 이미지를 제출합니다 (50% 비용 절감)."""
    files = get_image_files()
    if not files:
        print(f"[오류] {INPUT_DIR} 에 이미지 파일이 없습니다.")
        return

    state = load_state()
    processed_set = set(state["processed"])
    pending = [f for f in files if f.name not in processed_set]

    if not pending:
        print("모든 이미지가 이미 처리되었습니다.")
        return

    print(f"배치 제출 준비: {len(pending)}장")
    print("이미지 인코딩 중...")

    # 배치 요청 구성 (이미지 크기 고려해 20장씩 분할)
    CHUNK_SIZE = 20
    batch_ids = []

    for chunk_start in range(0, len(pending), CHUNK_SIZE):
        chunk = pending[chunk_start:chunk_start + CHUNK_SIZE]
        requests = []

        for path in tqdm(chunk, desc=f"인코딩 ({chunk_start+1}-{chunk_start+len(chunk)})", unit="장"):
            try:
                image_data, media_type = encode_image(path)
                requests.append({
                    "custom_id": path.name,
                    "params": {
                        "model": MODEL,
                        "max_tokens": 2048,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_data,
                                    },
                                },
                                {"type": "text", "text": EXTRACTION_PROMPT},
                            ],
                        }],
                    },
                })
            except Exception as e:
                print(f"\n[인코딩 오류] {path.name}: {e}")

        if not requests:
            continue

        print(f"\n배치 {len(batch_ids)+1} 제출 중 ({len(requests)}장)...")
        batch = client.messages.batches.create(requests=requests)
        batch_ids.append(batch.id)
        print(f"  배치 ID: {batch.id} (상태: {batch.processing_status})")

    batch_state = {
        "batch_ids": batch_ids,
        "submitted_at": datetime.now().isoformat(),
        "total_images": len(pending),
    }
    save_batch_state(batch_state)

    print(f"\n배치 제출 완료!")
    print(f"배치 수: {len(batch_ids)}개")
    print(f"처리에 최대 24시간이 소요될 수 있습니다.")
    print(f"완료 확인: python process.py --mode check")


def check_batch(client: anthropic.Anthropic):
    """배치 작업 상태를 확인하고 완료된 결과를 수집합니다."""
    batch_state = load_batch_state()
    if not batch_state["batch_ids"]:
        print("[오류] 제출된 배치가 없습니다. '--mode batch' 로 먼저 제출하세요.")
        return

    state = load_state()
    JSON_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    all_done = True
    total_success = 0
    total_error = 0

    for batch_id in batch_state["batch_ids"]:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"\n배치 {batch_id}:")
        print(f"  상태: {batch.processing_status}")
        print(f"  처리 중: {counts.processing} | 완료: {counts.succeeded} | "
              f"오류: {counts.errored} | 취소: {counts.canceled}")

        if batch.processing_status != "ended":
            all_done = False
            continue

        # 결과 수집
        print("  결과 수집 중...")
        for result in client.messages.batches.results(batch_id):
            filename = result.custom_id

            if result.result.type == "succeeded":
                msg = result.result.message
                text = next(
                    (b.text for b in msg.content if b.type == "text"), ""
                )
                data = parse_response(text, filename)
                out_path = JSON_DIR / f"{Path(filename).stem}.json"
                out_path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                if filename not in state["processed"]:
                    state["processed"].append(filename)
                total_success += 1

            else:
                err_msg = str(result.result)
                print(f"  [오류] {filename}: {err_msg}")
                state["errors"].append({"file": filename, "error": err_msg})
                total_error += 1

    save_state(state)
    print(f"\n수집 완료: 성공 {total_success}장, 오류 {total_error}장")

    if all_done:
        print("\n모든 배치가 완료되었습니다!")
        print("아카이브 생성 중...")
        build_archive()
    else:
        print("\n아직 처리 중인 배치가 있습니다. 나중에 다시 확인하세요.")


# ── 아카이브 빌드 ─────────────────────────────────────────────────────────────

def build_archive():
    """JSON 결과들을 하나의 archive.json으로 통합하고 볼륨별로 구성합니다."""
    json_files = sorted(JSON_DIR.glob("*.json"))
    has_ocr = len(json_files) > 0
    if not has_ocr and not MANIFEST_FILE.exists():
        print("[오류] 처리된 JSON 파일도, manifest.json도 없습니다.")
        print("  먼저 'python scripts/1_preprocess.py' 를 실행하세요.")
        return
    if not has_ocr:
        print(f"[알림] OCR 결과 없음. manifest 기반으로 이미지 뷰어만 생성합니다.")

    # 파일별 OCR 결과 로드
    records_by_file: dict[str, dict] = {}
    for jf in json_files:
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            records_by_file[data["filename"]] = data
        except Exception as e:
            print(f"[경고] {jf.name} 읽기 실패: {e}")

    # manifest 기반 볼륨 구성
    volumes = []
    all_records = []

    if MANIFEST_FILE.exists():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        for vol in manifest:
            vol_records = []
            for page in vol.get("pages", []):
                fname = page["file"]
                rec = records_by_file.get(fname, {
                    "filename": fname,
                    "content": None,
                    "summary": None,
                    "author": None,
                    "date": None,
                    "tags": [],
                    "mood": "neutral",
                    "decade": f"{vol.get('year', 'unknown')}",
                    "image_quality": "unknown",
                })
                rec["page_label"] = page.get("label", "")
                rec["volume_id"] = vol["id"]
                rec["volume_title"] = vol["title"]
                rec["year"] = vol.get("year")
                vol_records.append(rec)
                all_records.append(rec)

            volumes.append({
                "id": vol["id"],
                "title": vol["title"],
                "short_title": vol.get("short_title", vol["title"]),
                "year": vol.get("year"),
                "cover": vol.get("cover"),
                "page_count": vol.get("page_count", len(vol_records)),
                "pages": vol_records,
            })
    else:
        # manifest 없을 경우 평탄한 목록 사용
        all_records = list(records_by_file.values())
        all_records.sort(key=lambda r: r.get("filename", ""))
        volumes = [{"id": "all", "title": "열린마음", "year": None,
                    "cover": None, "page_count": len(all_records), "pages": all_records}]

    archive = {
        "project": "열린마음 아카이브",
        "created_at": datetime.now().isoformat(),
        "total_records": len(all_records),
        "volumes": volumes,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ARCHIVE_FILE.write_text(
        json.dumps(archive, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"아카이브 저장: {ARCHIVE_FILE} ({len(all_records)}건, {len(volumes)}권)")

    # HTML 뷰어 생성
    build_viewer(volumes)


def build_viewer(volumes: list[dict]):
    """볼륨 타임라인 + 페이지 뷰어 + 검색이 포함된 SPA HTML을 생성합니다."""
    volumes_json = json.dumps(volumes, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>열린마음 아카이브</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:"Malgun Gothic","Apple Gothic",sans-serif;background:#f5f0e8;color:#333;min-height:100vh}}

/* ── 헤더 ── */
.site-header{{background:#3d2b1f;color:#f5efe4;padding:0 24px;display:flex;align-items:center;gap:16px;height:56px;position:sticky;top:0;z-index:100;box-shadow:0 2px 8px rgba(0,0,0,.3)}}
.site-header h1{{font-size:1.25rem;font-weight:700;letter-spacing:-.5px;cursor:pointer}}
.site-header h1 span{{font-size:.8rem;opacity:.6;margin-left:8px;font-weight:400}}
.header-search{{flex:1;max-width:360px;position:relative}}
.header-search input{{width:100%;padding:6px 12px 6px 32px;border:none;border-radius:20px;background:rgba(255,255,255,.15);color:#fff;font-size:.9rem;outline:none}}
.header-search input::placeholder{{color:rgba(255,255,255,.5)}}
.header-search input:focus{{background:rgba(255,255,255,.25)}}
.search-icon{{position:absolute;left:10px;top:50%;transform:translateY(-50%);opacity:.6;font-size:.85rem}}
.header-nav{{margin-left:auto;display:flex;gap:8px}}
.nav-btn{{background:rgba(255,255,255,.15);border:none;color:#fff;padding:6px 14px;border-radius:16px;cursor:pointer;font-size:.85rem}}
.nav-btn:hover{{background:rgba(255,255,255,.25)}}
.nav-btn.active{{background:rgba(255,255,255,.3);font-weight:600}}

/* ── 홈: 타임라인 ── */
#home-view{{max-width:1100px;margin:0 auto;padding:32px 16px}}
.timeline-year{{margin-bottom:40px}}
.year-label{{font-size:1.4rem;font-weight:700;color:#3d2b1f;margin-bottom:16px;padding-bottom:8px;border-bottom:2px solid #c8b89a;display:flex;align-items:center;gap:12px}}
.year-label::after{{content:"";flex:1;height:1px;background:#e0d0bc}}
.vol-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px}}
.vol-card{{background:#fffdf7;border:1px solid #e0d0bc;border-radius:12px;overflow:hidden;cursor:pointer;transition:all .2s}}
.vol-card:hover{{box-shadow:0 6px 20px rgba(0,0,0,.15);transform:translateY(-2px)}}
.vol-cover{{width:100%;aspect-ratio:3/4;object-fit:cover;background:#e8dfd0;display:flex;align-items:center;justify-content:center;color:#a09070;font-size:.85rem}}
.vol-cover img{{width:100%;height:100%;object-fit:cover}}
.vol-cover-placeholder{{width:100%;aspect-ratio:3/4;background:linear-gradient(135deg,#e8dfd0,#d4c4a8);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;color:#8b7355}}
.vol-info{{padding:12px}}
.vol-title{{font-size:.9rem;font-weight:600;color:#3d2b1f;margin-bottom:4px;line-height:1.3}}
.vol-meta{{font-size:.75rem;color:#8b7355}}

/* ── 볼륨 뷰어 ── */
#viewer-view{{display:none;height:calc(100vh - 56px);flex-direction:column}}
.viewer-toolbar{{background:#fff;border-bottom:1px solid #e0d0bc;padding:10px 20px;display:flex;align-items:center;gap:12px;flex-shrink:0}}
.back-btn{{background:#f0e8dc;border:1px solid #c8b89a;color:#3d2b1f;padding:5px 14px;border-radius:8px;cursor:pointer;font-size:.85rem;font-weight:600}}
.back-btn:hover{{background:#e8dccf}}
.viewer-title{{font-size:1rem;font-weight:700;color:#3d2b1f;flex:1}}
.page-nav{{display:flex;align-items:center;gap:8px}}
.page-btn{{background:#f0e8dc;border:1px solid #c8b89a;color:#3d2b1f;width:32px;height:32px;border-radius:8px;cursor:pointer;font-size:1rem;display:flex;align-items:center;justify-content:center}}
.page-btn:hover:not(:disabled){{background:#e8dccf}}
.page-btn:disabled{{opacity:.3;cursor:default}}
.page-indicator{{font-size:.85rem;color:#6b5a48;min-width:80px;text-align:center}}
.viewer-body{{flex:1;display:flex;overflow:hidden}}
.viewer-image-pane{{flex:1;overflow:auto;background:#2a2a2a;display:flex;align-items:center;justify-content:center;padding:16px}}
.viewer-image-pane img{{max-width:100%;max-height:100%;object-fit:contain;border-radius:4px;box-shadow:0 4px 24px rgba(0,0,0,.5)}}
.viewer-text-pane{{width:380px;flex-shrink:0;overflow-y:auto;background:#fffdf7;border-left:1px solid #e0d0bc;padding:20px}}
.text-pane-section{{margin-bottom:20px}}
.text-pane-label{{font-size:.7rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#9b8870;margin-bottom:6px}}
.text-author{{background:#f0e8dc;padding:4px 10px;border-radius:12px;font-size:.9rem;color:#3d2b1f;display:inline-block}}
.text-date{{color:#6b5a48;font-size:.9rem}}
.text-summary{{font-size:.9rem;line-height:1.7;color:#444}}
.text-content{{font-size:.85rem;line-height:1.8;color:#555;white-space:pre-wrap;word-break:break-all;background:#f9f5ef;border-radius:8px;padding:14px;border:1px solid #ebe0ce}}
.text-tags{{display:flex;flex-wrap:wrap;gap:6px}}
.tag{{font-size:.72rem;background:#ede3d3;color:#6b5344;padding:3px 10px;border-radius:12px}}
.mood-badge{{display:inline-flex;align-items:center;gap:4px;font-size:.8rem;padding:3px 10px;border-radius:12px}}
.mood-positive{{background:#e8f5e8;color:#2d6b2d}}
.mood-neutral{{background:#f5f0e0;color:#6b5a20}}
.mood-negative{{background:#f5e8e8;color:#6b2020}}
.no-text{{color:#aaa;font-style:italic;font-size:.85rem}}

/* ── 검색 결과 ── */
#search-view{{display:none;max-width:900px;margin:0 auto;padding:24px 16px}}
.search-results-header{{margin-bottom:20px;color:#6b5a48;font-size:.9rem}}
.result-card{{background:#fffdf7;border:1px solid #e0d0bc;border-radius:10px;padding:16px;margin-bottom:12px;cursor:pointer;transition:box-shadow .15s}}
.result-card:hover{{box-shadow:0 4px 12px rgba(0,0,0,.1)}}
.result-vol{{font-size:.75rem;color:#9b8870;margin-bottom:4px}}
.result-summary{{font-size:.9rem;color:#444;line-height:1.6;margin-top:6px}}
.result-content{{font-size:.8rem;color:#777;margin-top:6px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}}
mark{{background:#ffe066;border-radius:2px;padding:0 2px}}
</style>
</head>
<body>

<header class="site-header">
  <h1 onclick="showHome()">열린마음 <span>아카이브</span></h1>
  <div class="header-search">
    <span class="search-icon">🔍</span>
    <input type="text" id="global-search" placeholder="내용, 작성자 검색..." oninput="onSearch(this.value)">
  </div>
  <nav class="header-nav">
    <button class="nav-btn" id="nav-home" onclick="showHome()">연도별 보기</button>
  </nav>
</header>

<!-- 홈 뷰 -->
<div id="home-view"></div>

<!-- 볼륨 뷰어 -->
<div id="viewer-view">
  <div class="viewer-toolbar">
    <button class="back-btn" onclick="showHome()">← 목록</button>
    <span class="viewer-title" id="viewer-title"></span>
    <div class="page-nav">
      <button class="page-btn" id="btn-prev" onclick="changePage(-1)">‹</button>
      <span class="page-indicator" id="page-indicator"></span>
      <button class="page-btn" id="btn-next" onclick="changePage(1)">›</button>
    </div>
  </div>
  <div class="viewer-body">
    <div class="viewer-image-pane">
      <img id="viewer-img" src="" alt="">
    </div>
    <div class="viewer-text-pane" id="viewer-text"></div>
  </div>
</div>

<!-- 검색 결과 -->
<div id="search-view"></div>

<script>
const VOLUMES = {volumes_json};

// 볼륨 맵 (id → volume)
const volMap = {{}};
VOLUMES.forEach(v => volMap[v.id] = v);

// ── 홈 렌더링 ──────────────────────────────────────────────────────────────
function showHome() {{
  document.getElementById("global-search").value = "";
  document.getElementById("home-view").style.display = "block";
  document.getElementById("viewer-view").style.display = "none";
  document.getElementById("search-view").style.display = "none";
  renderHome();
}}

function renderHome() {{
  const years = {{}};
  VOLUMES.forEach(v => {{
    const y = v.year || "연도 미상";
    if (!years[y]) years[y] = [];
    years[y].push(v);
  }});

  const sortedYears = Object.keys(years).sort((a, b) => {{
    if (a === "연도 미상") return 1;
    if (b === "연도 미상") return -1;
    return Number(a) - Number(b);
  }});

  const totalVols = VOLUMES.length;
  const totalPages = VOLUMES.reduce((s, v) => s + (v.page_count || 0), 0);

  let html = `<div style="background:#3d2b1f;color:#f5efe4;padding:32px 24px;margin-bottom:32px">
    <h2 style="font-size:1.8rem;margin-bottom:8px">열린마음 아카이브</h2>
    <p style="opacity:.7;font-size:.95rem">1990년대 ~ 2000년대 학과 방명록 &amp; 일상 기록 디지털 보존</p>
    <div style="display:flex;gap:32px;margin-top:20px">
      <div><div style="font-size:2rem;font-weight:700">${{totalVols}}</div><div style="opacity:.6;font-size:.85rem">권</div></div>
      <div><div style="font-size:2rem;font-weight:700">${{totalPages}}</div><div style="opacity:.6;font-size:.85rem">페이지</div></div>
    </div>
  </div>`;

  sortedYears.forEach(year => {{
    html += `<div class="timeline-year">
      <div class="year-label">${{year}}년</div>
      <div class="vol-grid">`;
    years[year].forEach(vol => {{
      const cover = vol.cover ? `<img src="input/${{vol.cover}}" alt="표지" onerror="this.parentElement.innerHTML='<div style=\\"width:100%;aspect-ratio:3/4;background:linear-gradient(135deg,#e8dfd0,#d4c4a8);display:flex;align-items:center;justify-content:center;color:#8b7355;font-size:.8rem;text-align:center;padding:16px\\">${{vol.short_title || vol.title}}</div>'">` :
        `<div style="width:100%;aspect-ratio:3/4;background:linear-gradient(135deg,#e8dfd0,#d4c4a8);display:flex;align-items:center;justify-content:center;color:#8b7355;font-size:.8rem;text-align:center;padding:16px">${{vol.short_title || vol.title}}</div>`;
      html += `<div class="vol-card" onclick="openVolume('${{vol.id}}')">
        <div class="vol-cover">${{cover}}</div>
        <div class="vol-info">
          <div class="vol-title">${{vol.title}}</div>
          <div class="vol-meta">${{vol.page_count || 0}}페이지</div>
        </div>
      </div>`;
    }});
    html += `</div></div>`;
  }});

  document.getElementById("home-view").innerHTML = html;
}}

// ── 볼륨 뷰어 ──────────────────────────────────────────────────────────────
let currentVol = null;
let currentPageIdx = 0;

function openVolume(volId, pageIdx) {{
  currentVol = volMap[volId];
  currentPageIdx = pageIdx || 0;
  if (!currentVol) return;

  document.getElementById("home-view").style.display = "none";
  document.getElementById("search-view").style.display = "none";
  const vv = document.getElementById("viewer-view");
  vv.style.display = "flex";
  document.getElementById("viewer-title").textContent = currentVol.title;

  renderPage();
}}

function renderPage() {{
  const pages = currentVol.pages || [];
  const total = pages.length;
  if (total === 0) return;

  const page = pages[currentPageIdx];

  // 이미지
  const img = document.getElementById("viewer-img");
  img.src = `input/${{page.filename}}`;
  img.alt = page.page_label || "";

  // 페이지 표시
  const label = page.page_label === "cover_front" ? "앞표지" :
                page.page_label === "cover_back" ? "뒷표지" :
                `${{currentPageIdx + 1}} / ${{total}}`;
  document.getElementById("page-indicator").textContent = label;
  document.getElementById("btn-prev").disabled = currentPageIdx === 0;
  document.getElementById("btn-next").disabled = currentPageIdx >= total - 1;

  // 텍스트 패널
  const mood = page.mood || "neutral";
  const moodLabel = mood === "positive" ? "긍정적" : mood === "negative" ? "부정적" : "중립";
  const moodClass = `mood-${{mood}}`;

  let textHtml = "";

  if (page.author) {{
    textHtml += `<div class="text-pane-section">
      <div class="text-pane-label">작성자</div>
      <span class="text-author">✍ ${{page.author}}</span>
    </div>`;
  }}
  if (page.date) {{
    textHtml += `<div class="text-pane-section">
      <div class="text-pane-label">날짜</div>
      <span class="text-date">📅 ${{page.date}}</span>
    </div>`;
  }}
  if (page.summary) {{
    textHtml += `<div class="text-pane-section">
      <div class="text-pane-label">요약</div>
      <div class="text-summary">${{page.summary}}</div>
    </div>`;
  }}
  if (page.content && page.content !== "텍스트 없음") {{
    const escaped = (page.content || "").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
    textHtml += `<div class="text-pane-section">
      <div class="text-pane-label">원문 전사</div>
      <div class="text-content">${{escaped}}</div>
    </div>`;
  }} else if (!page.content) {{
    textHtml += `<div class="text-pane-section"><p class="no-text">OCR 처리 전</p></div>`;
  }}
  if ((page.tags || []).length > 0) {{
    textHtml += `<div class="text-pane-section">
      <div class="text-pane-label">태그</div>
      <div class="text-tags">${{page.tags.map(t=>`<span class="tag">${{t}}</span>`).join("")}}</div>
    </div>`;
  }}
  textHtml += `<div class="text-pane-section">
    <div class="text-pane-label">분위기</div>
    <span class="mood-badge ${{moodClass}}">${{moodLabel}}</span>
  </div>`;

  document.getElementById("viewer-text").innerHTML = textHtml;
}}

function changePage(delta) {{
  const pages = currentVol ? currentVol.pages : [];
  const newIdx = currentPageIdx + delta;
  if (newIdx < 0 || newIdx >= pages.length) return;
  currentPageIdx = newIdx;
  renderPage();
  document.querySelector(".viewer-image-pane").scrollTop = 0;
}}

// 키보드 네비게이션
document.addEventListener("keydown", e => {{
  if (document.getElementById("viewer-view").style.display === "flex") {{
    if (e.key === "ArrowLeft" || e.key === "ArrowUp") changePage(-1);
    if (e.key === "ArrowRight" || e.key === "ArrowDown") changePage(1);
    if (e.key === "Escape") showHome();
  }}
}});

// ── 검색 ──────────────────────────────────────────────────────────────────
function onSearch(query) {{
  if (!query.trim()) {{ showHome(); return; }}

  document.getElementById("home-view").style.display = "none";
  document.getElementById("viewer-view").style.display = "none";
  const sv = document.getElementById("search-view");
  sv.style.display = "block";

  const q = query.toLowerCase();
  const results = [];

  VOLUMES.forEach(vol => {{
    (vol.pages || []).forEach((page, idx) => {{
      const text = [page.content, page.author, page.date, page.summary, ...(page.tags||[])].join(" ").toLowerCase();
      if (text.includes(q)) results.push({{vol, page, idx}});
    }});
  }});

  function hl(str) {{
    if (!str) return "";
    return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")
      .replace(new RegExp(q.replace(/[.*+?^${{}}()|[\\]\\\\]/g,"\\\\$&"), "gi"), m => `<mark>${{m}}</mark>`);
  }}

  let html = `<h2 style="font-size:1rem;color:#6b5a48;margin-bottom:16px">"${{query}}" 검색 결과 — ${{results.length}}건</h2>`;
  if (results.length === 0) {{
    html += `<p style="color:#aaa;font-style:italic">검색 결과가 없습니다.</p>`;
  }} else {{
    results.forEach(({{vol, page, idx}}) => {{
      const label = page.page_label === "cover_front" ? "앞표지" :
                    page.page_label === "cover_back" ? "뒷표지" : `${{idx+1}}페이지`;
      html += `<div class="result-card" onclick="openVolume('${{vol.id}}', ${{idx}})">
        <div class="result-vol">${{vol.title}} · ${{label}}</div>
        ${{page.author ? `<span style="background:#f0e8dc;padding:2px 8px;border-radius:10px;font-size:.8rem">✍ ${{hl(page.author)}}</span>` : ""}}
        ${{page.summary ? `<div class="result-summary">${{hl(page.summary)}}</div>` : ""}}
        ${{page.content ? `<div class="result-content">${{hl(page.content)}}</div>` : ""}}
      </div>`;
    }});
  }}
  sv.innerHTML = html;
}}

// ── 초기화 ────────────────────────────────────────────────────────────────
showHome();
</script>
</body>
</html>"""

    viewer_path = BASE_DIR / "viewer.html"
    viewer_path.write_text(html, encoding="utf-8")
    print(f"HTML 뷰어 생성: {viewer_path}")
    print("브라우저에서 viewer.html 을 열어 아카이브를 확인하세요.")


# ── Markdown 내보내기 ─────────────────────────────────────────────────────────

DOCS_DIR = OUTPUT_DIR / "docs"

def export_markdown():
    """OCR 결과를 볼륨별 Markdown 파일 + 전체 통합본으로 내보냅니다."""
    if not ARCHIVE_FILE.exists():
        print("[오류] archive.json이 없습니다. 먼저 '--mode build'를 실행하세요.")
        return

    archive = json.loads(ARCHIVE_FILE.read_text(encoding="utf-8"))
    volumes = archive.get("volumes", [])
    if not volumes:
        print("[오류] 볼륨 데이터가 없습니다.")
        return

    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    all_lines = [
        "# 열린마음 아카이브 — 전체 기록",
        "",
        f"> 생성일: {datetime.now().strftime('%Y-%m-%d')}  ",
        f"> 총 {len(volumes)}권, {sum(len(v.get('pages', [])) for v in volumes)}페이지",
        "",
        "---",
        "",
    ]

    for vol in volumes:
        vol_title = vol.get("title", "제목 없음")
        vol_id = vol.get("id", "unknown")
        pages = vol.get("pages", [])
        content_pages = [p for p in pages if p.get("page_label") not in ("cover_front", "cover_back")]
        has_ocr = any(p.get("content") for p in pages)

        # 볼륨별 Markdown
        lines = [
            f"# {vol_title}",
            "",
        ]
        if not has_ocr:
            lines += ["> OCR 미처리 — `python process.py --mode build` 실행 후 내보내기 재시도", ""]

        for page in pages:
            label = page.get("page_label", "")
            filename = page.get("filename", "")

            if label == "cover_front":
                heading = "## 앞표지"
            elif label == "cover_back":
                heading = "## 뒷표지"
            else:
                # 페이지 번호: p001 → 1페이지
                num = label.lstrip("p").lstrip("0") or "0"
                heading = f"## {num}페이지"

            lines.append(heading)
            lines.append("")
            if filename:
                lines.append(f"> 이미지: `{filename}`")
                lines.append("")

            author = page.get("author")
            date = page.get("date")
            mood = page.get("mood")
            mood_str = {"positive": "긍정적", "negative": "부정적", "neutral": "중립"}.get(mood, "")

            if author:
                lines.append(f"**작성자:** {author}  ")
            if date:
                lines.append(f"**날짜:** {date}  ")
            if mood_str:
                lines.append(f"**분위기:** {mood_str}  ")
            if author or date or mood_str:
                lines.append("")

            content = page.get("content")
            if content and content != "텍스트 없음":
                lines.append(content)
                lines.append("")

            tags = page.get("tags") or []
            if tags:
                lines.append("  ".join(f"#{t}" for t in tags))
                lines.append("")

            lines.append("---")
            lines.append("")

        md_text = "\n".join(lines)
        out_path = DOCS_DIR / f"{vol_id}.md"
        out_path.write_text(md_text, encoding="utf-8")
        print(f"  저장: {out_path.name} ({len(content_pages)}페이지)")

        # 전체 통합본에 추가
        all_lines.append(f"# {vol_title}")
        all_lines.append("")
        all_lines.extend(lines[2:])  # 제목 중복 제거

    # 전체 통합 Markdown
    combined_path = DOCS_DIR / "전체_열린마음.md"
    combined_path.write_text("\n".join(all_lines), encoding="utf-8")

    total_pages = sum(len(v.get("pages", [])) for v in volumes)
    print(f"\nMarkdown 내보내기 완료!")
    print(f"  볼륨별 파일: {DOCS_DIR}/*.md ({len(volumes)}개)")
    print(f"  전체 통합본: {combined_path}")
    print(f"  총 {total_pages}페이지 기록")


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="열린마음 아카이빙 프로세서")
    parser.add_argument(
        "--mode",
        choices=["batch", "live", "check", "build", "export"],
        default="batch",
        help="처리 모드 (기본: batch)",
    )
    args = parser.parse_args()

    # API 키 확인
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and args.mode not in ("build", "export"):
        print("[오류] ANTHROPIC_API_KEY 환경변수를 설정하세요.")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    # input 디렉토리 확인
    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.mode == "build":
        build_archive()
        return

    if args.mode == "export":
        export_markdown()
        return

    client = anthropic.Anthropic(api_key=api_key)

    if args.mode == "live":
        process_live(client)
    elif args.mode == "batch":
        submit_batch(client)
    elif args.mode == "check":
        check_batch(client)


if __name__ == "__main__":
    main()
