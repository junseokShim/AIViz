# AIViz

**로컬에서 구동되는 AI 기반 오픈소스 데스크톱 분석 플랫폼**

AIViz는 Tableau의 오픈소스 대안을 목표로 하는 **PyQt6 데스크톱 애플리케이션**입니다.  
Ollama 기반의 로컬 AI 에이전트와 결합하여, 데이터 분석·시각화·시계열·주파수 분석·이미지 분석을 하나의 앱에서 수행할 수 있습니다.  
인터넷 연결이나 클라우드 서비스 없이 **완전한 로컬 환경**에서 동작합니다.

---

## v0.2 변경사항 (최신)

### 버그 수정
| 항목 | 내용 |
|------|------|
| **이미지 표준편차 오류** | PIL `ImageStat.stddev`의 부동소수점 음수 분산 문제 → numpy 기반으로 교체, `ValueError` 완전 해결 |
| **FFT/플롯 오버플로우** | 수백만 포인트 렌더링 시 `OverflowError: Exceeded cell block limit` → 자동 다운샘플링 적용 |
| **QFont 경고** | `QFont::setPointSize <= 0` 경고 → 폰트 크기 항상 8-72pt 범위로 클램핑 |

### 한국어 폰트 안정화
- **Qt stylesheet**: `Apple SD Gothic Neo`, `Malgun Gothic` 등 한국어 폰트를 font-family 최우선으로 설정
- **matplotlib**: 플랫폼별 한국어 폰트 자동 감지 및 `font.sans-serif` 선행 주입
- `axes.unicode_minus = False` 전역 적용 (□ 문자 방지)
- `utils/font_utils.py`: 안전한 QFont 생성 + 한국어 폰트 탐색 유틸리티

### 대용량 데이터 처리 (빅데이터 전략)
Tableau처럼 메모리와 스토리지를 혼합 활용하는 구조로 개선:

| 전략 | 내용 |
|------|------|
| **미리보기 모드** | 50MB 초과 파일은 기본적으로 10만 행만 로드 |
| **Parquet 스트리밍** | pyarrow 행 그룹 단위 순차 로드로 메모리 절약 |
| **안전한 시각화** | 10,000포인트 초과 시 자동 다운샘플링, 플롯에 샘플링 경고 표시 |
| **원본 데이터 보존** | 분석 객체는 전체 데이터 유지, 시각화만 샘플링 적용 |

### 이미지 분석 확장
| 기능 | 모듈 |
|------|------|
| **엣지 검출** | Canny (cv2 또는 scipy 폴백), Sobel, Laplacian |
| **이미지 분할** | 전역 임계값, 적응형 임계값, K-Means 클러스터링 |
| **상호작용 캔버스** | 클릭 → 픽셀 값 확인, 드래그 → ROI 선택 및 통계 |
| **전처리 파이프라인** | 밝기·대비·채도·클리핑·정규화 슬라이더 |
| **XML 저장/불러오기** | 전처리 파라미터, 엣지/분할 설정, 채널 통계를 XML로 저장 및 복원 |

---

## 주요 기능 전체

| 기능 | 설명 |
|------|------|
| 데이터 미리보기 | 테이블 뷰, 스키마 검사, 결측값 히트맵 |
| **폴더 일괄 로드** | 폴더 내 CSV/Excel/JSON 파일 일괄 파싱, 스키마 불일치 처리 |
| **파생 컬럼 생성** | 수식 엔진으로 새 컬럼 생성, 모든 탭에서 즉시 사용 가능 |
| 차트 빌더 | 선형·산점도·막대·히스토그램·박스플롯·상관관계 히트맵 |
| 시계열 분석 | 롤링 통계, 추세 감지, 이상치 탐지, 스무딩 |
| 주파수 분석 | FFT, 진폭/전력 스펙트럼, 피크 감지, 밴드 에너지, 스펙트로그램 |
| **AC 성분 분석** | DC 오프셋 제거 후 AC 신호 FFT, RMS, 크레스트 팩터 |
| **이미지 전처리** | 밝기·대비·채도·클리핑 슬라이더, 결과 즉시 미리보기 |
| **엣지 검출** | Canny / Sobel / Laplacian, 오버레이 표시 |
| **이미지 분할** | 임계값 / 적응형 / K-Means, 영역 오버레이 표시 |
| **인터랙티브 캔버스** | 픽셀 클릭 검사, ROI 드래그 선택, 통계 출력 |
| **XML 저장/불러오기** | 분석 세션 전체를 XML로 직렬화 및 복원 |
| 예측 모듈 | Holt-Winters / ARIMA / Simple ES |
| **클러스터링** | KMeans / DBSCAN, 실루엣 점수 |
| **신경망 (MLP)** | 다층 퍼셉트론 회귀/분류 |
| AI 어시스턴트 | Ollama 기반 데이터 요약, 차트 제안, 패턴 설명 |
| 멀티모달 분석 | LLaVA를 통한 이미지 픽셀 기반 AI 설명 |
| 보고서 내보내기 | PDF (fpdf2) 및 HTML 자동 생성 |
| 오프라인 폴백 | Ollama 비활성 시에도 모든 분석 기능 정상 동작 |

---

## 전체 아키텍처

```
AIViz/
├── main.py                          # 데스크톱 앱 진입점 (PyQt6)
├── config.py                        # 전역 설정
├── requirements.txt
│
├── aiviz/
│   ├── app/
│   │   ├── main_window.py           # QMainWindow – 탭/독/메뉴바/상태바
│   │   ├── controller.py            # AppController – 데이터 상태 및 신호 허브
│   │   └── style.py                 # Catppuccin Mocha 다크 테마 + 한국어 폰트 설정
│   │
│   ├── ui/
│   │   ├── panel_image.py           # 이미지 분석 탭 (v0.2: 전처리/엣지/분할/XML)
│   │   ├── panel_frequency.py       # 주파수 분석 탭
│   │   └── widgets/
│   │       ├── image_canvas.py      # NEW: 인터랙티브 이미지 캔버스
│   │       ├── plot_widget.py       # Matplotlib 캔버스 래퍼
│   │       └── data_table.py        # DataFrame → QTableView
│   │
│   ├── analytics/
│   │   ├── image_analysis.py        # 픽셀 통계 (numpy 기반, 수치 안정)
│   │   ├── image_preprocess.py      # NEW: 전처리 파이프라인
│   │   ├── image_edges.py           # NEW: 엣지 검출 (Canny/Sobel/Laplacian)
│   │   ├── image_segmentation.py    # NEW: 이미지 분할
│   │   ├── timeseries.py
│   │   ├── frequency.py
│   │   └── forecast.py
│   │
│   ├── services/
│   │   └── image_xml_service.py     # NEW: XML 저장/불러오기
│   │
│   ├── ingestion/
│   │   └── loader.py                # 빅데이터 지원: 미리보기 모드, Parquet 스트리밍
│   │
│   ├── visualization/
│   │   └── mpl_charts.py            # 안전한 렌더링: 자동 다운샘플링, 오버플로 방지
│   │
│   └── utils/
│       ├── font_utils.py            # NEW: 한국어 폰트 탐색 및 matplotlib 설정
│       └── safe_font.py             # 기존: QFont 안전 생성
│
├── tests/                           # pytest 테스트 (80개 통과)
└── docs/
```

**핵심 아키텍처 원칙:**
- `analytics/`, `visualization/`, `ai/`는 PyQt 임포트 없음
- 모든 패널은 `AppController` 신호를 통해서만 데이터를 수신
- Ollama 연결 실패 시 모든 기능은 폴백 모드로 정상 동작
- 시각화는 디스플레이용 다운샘플링, 분석은 원본 데이터 사용

---

## 설치 방법

### 사전 요구사항

- Python 3.10 이상
- [Ollama](https://ollama.com/download) (AI 기능 사용 시 필요)

### 저장소 클론 및 패키지 설치

```bash
git clone https://github.com/junseokShim/AIViz.git
cd AIViz
pip install -r requirements.txt
```

### 선택적 패키지

| 패키지 | 용도 | 없을 때 동작 |
|--------|------|-------------|
| `opencv-python-headless` | Canny 엣지 검출 최적화 | scipy 기반 근사 구현 자동 사용 |
| `scikit-learn` | K-Means 이미지 분할, 클러스터링 | K-Means 탭에서 ImportError 안내 |

```bash
pip install opencv-python-headless  # Canny 최적화
pip install scikit-learn             # K-Means 분할 + 클러스터링
```

---

## Ollama 설정

```bash
ollama serve
ollama pull llama3.2   # 텍스트 분석
ollama pull llava      # 이미지 멀티모달 분석 (선택)
```

---

## 실행 방법

```bash
python main.py
```

---

## 이미지 분석 기능 사용법 (v0.2)

### 전처리 (전처리 탭)

1. 이미지 파일 로드 → 🖼 Image 탭
2. **전처리** 탭 클릭
3. 슬라이더로 밝기·대비·채도·클리핑 조정
4. **전처리 적용** 버튼 → 캔버스에 결과 즉시 반영
5. **전처리 초기화** 버튼 → 원본 복원

### 엣지 검출 (엣지 검출 탭)

1. **엣지 검출** 탭 클릭
2. 방법 선택: `canny` / `sobel` / `laplacian`
3. Canny 임계값·σ 파라미터 설정
4. **엣지 검출 실행** → 엣지 맵 표시
5. **오버레이 표시** → 원본 이미지 위에 엣지 오버레이

### 이미지 분할 (분할 탭)

1. **분할** 탭 클릭
2. 방법 선택: `threshold` / `adaptive` / `kmeans`
3. 해당 파라미터 설정 후 **분할 실행**
4. **오버레이 표시** → 색상별 영역 시각화

### 상호작용 캔버스

- **클릭**: 픽셀 좌표 및 RGB 값 표시
- **드래그**: ROI 직사각형 선택 → 영역 평균 RGB 통계 출력

### XML 저장/불러오기

```
💾 XML 저장  → 전처리 파라미터, 엣지/분할 설정, 채널 통계를 XML 파일로 저장
📂 XML 불러오기 → XML 파일 로드 시 UI 파라미터 자동 복원
```

**XML 스키마 예시:**
```xml
<AIVizImageAnalysis version="0.2.0">
  <Metadata>
    <ImagePath>/path/to/image.jpg</ImagePath>
    <ImageSize>1920x1080</ImageSize>
    <ImageMode>RGB</ImageMode>
    <CreatedAt>2026-04-22T10:30:00+00:00</CreatedAt>
  </Metadata>
  <Preprocessing>
    <brightness>1.2</brightness>
    <contrast>1.0</contrast>
    <saturation>0.9</saturation>
    <clip_min>10</clip_min>
    <clip_max>245</clip_max>
  </Preprocessing>
  <EdgeDetection method="canny">
    <low_threshold>50</low_threshold>
    <high_threshold>150</high_threshold>
    <sigma>1.5</sigma>
  </EdgeDetection>
  <Segmentation method="threshold">
    <threshold>128</threshold>
  </Segmentation>
  <ChannelStats>
    <Channel channel="R" mean="127.5" std="45.2" min="0" max="255"/>
  </ChannelStats>
</AIVizImageAnalysis>
```

---

## 대용량 데이터 처리 방식

| 파일 크기 | 동작 |
|-----------|------|
| < 50 MB | 전체 로드 |
| ≥ 50 MB (CSV) | 미리보기 모드: 첫 100,000행 로드, 상태바에 경고 표시 |
| ≥ 50 MB (Parquet) | pyarrow 행 그룹 스트리밍으로 100,000행 적재 |
| 500,000행 이상 (전체 로드 시) | 시각화 자동 다운샘플링 경고 표시 |

**시각화 안전 한도:**
- 선형/주파수 차트: 최대 10,000 포인트
- 산점도: 최대 5,000 포인트
- 피크 어노테이션: 최대 20개
- 초과 시 플롯 위에 `⚠ 표시: N / M 포인트 (다운샘플링)` 경고 출력

---

## 테스트 실행

```bash
python -m pytest tests/ -v
# 현재 80개 테스트 통과 (Ollama 없이 실행 가능)
```

---

## 한계점

- **Ollama 필요**: AI 기능은 `ollama serve` 없이 폴백 모드만 동작
- **멀티모달 AI**: LLaVA 등 비전 모델 별도 다운로드 필요
- **K-Means 분할**: scikit-learn 설치 필요 (`pip install scikit-learn`)
- **Canny 최적화**: opencv 미설치 시 scipy 근사 구현 사용 (결과 유사하나 속도 차이)
- **엑셀 빅데이터**: openpyxl은 스트리밍 미지원 → 전체 로드 (RAM 주의)
- **분할 영역 수**: K-Means의 n_clusters를 낮게 설정할수록 빠름

---

## 향후 확장 방향

- [ ] 실시간 데이터 스트리밍 지원 (소켓/시리얼 포트)
- [ ] CLIP 임베딩 기반 이미지 유사도 검색
- [ ] 이미지 배치 처리 (폴더 단위 자동 분석)
- [ ] 고급 분할: Watershed, GrabCut
- [ ] Prophet 기반 고급 예측 모듈
- [ ] 밝은 테마(라이트 모드) 지원
- [ ] 파생 컬럼 세션 저장/불러오기
- [ ] DuckDB 백엔드로 진정한 빅데이터 지원
- [ ] PyTorch 기반 딥러닝 모듈

---

## 라이선스

MIT License
