# 열린마음 아카이브

1990~2000년대 학과 방명록 "열린마음" 디지털 아카이브 프로젝트.

## 프로젝트 구조

```
열린마음-archive/
├── scripts/
│   └── 1_preprocess.py   # 소스 사진 → input/ 정규화 (HEIC 변환)
├── process.py            # OCR 처리 + 뷰어 빌드 (메인 스크립트)
├── input/                # 정규화된 JPG 이미지 (744장)
├── data/
│   └── manifest.json     # 13권 볼륨 목록
├── output/
│   ├── json/             # 페이지별 OCR 결과
│   └── archive.json      # 통합 아카이브 데이터
├── viewer.html           # 웹 뷰어 (브라우저로 열기)
└── requirements.txt
```

## 실행 방법

### 환경 설정 (최초 1회)
```bash
cd ~/열린마음-archive
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 1단계: 이미지 전처리 (이미 완료)
```bash
.venv/bin/python scripts/1_preprocess.py
# → input/ 에 744장 JPG 복사/변환
# → data/manifest.json 생성
```

### 2단계: OCR 처리
```bash
# API 키 설정
export ANTHROPIC_API_KEY='sk-ant-...'

# 배치 모드 (권장 - 50% 비용 절감, 비동기)
.venv/bin/python process.py --mode batch

# 완료 확인 (배치 완료까지 최대 24시간)
.venv/bin/python process.py --mode check

# 실시간 모드 (즉시 확인 가능, 비용 2배)
.venv/bin/python process.py --mode live
```

### 3단계: 뷰어 빌드
```bash
.venv/bin/python process.py --mode build
```

### 4단계: 로컬 미리보기
```bash
.venv/bin/python -m http.server 8000
# 브라우저에서 http://localhost:8000/viewer.html 열기
```

## 비용 안내

| 모드 | 예상 비용 (744장) |
|------|------------------|
| batch (claude-haiku) | **약 $1~2** |
| live (claude-haiku) | **약 $2~4** |

## 현황

- 볼륨: 13권 (1991~2000년)
- 처리된 이미지: 744장 / 772장
  - 손상 파일 28장: 2000년 열린마음 폴더의 일부 HEIC 파일 (복구 불가)
- OCR: 미완료 (ANTHROPIC_API_KEY 설정 후 `--mode batch` 실행 필요)

## 웹 뷰어 기능

- **연도별 타임라인**: 권별 표지 이미지 + 클릭하면 페이지 뷰어 진입
- **페이지 뷰어**: 원본 이미지(좌) + OCR 전사 텍스트(우) 나란히 표시
- **키보드 네비게이션**: ← → 방향키로 페이지 이동, Esc로 목록 복귀
- **전체 검색**: 상단 검색창으로 내용, 작성자, 날짜 검색
