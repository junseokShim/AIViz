# AIViz

**로컬에서 구동되는 AI 기반 오픈소스 데스크톱 분석 플랫폼**

AIViz는 Tableau의 오픈소스 대안을 목표로 하는 **PyQt6 데스크톱 애플리케이션**입니다.  
Ollama 기반의 로컬 AI 에이전트와 결합하여, 데이터 분석·시각화·시계열·주파수 분석·이미지 분석을 하나의 앱에서 수행할 수 있습니다.  
인터넷 연결이나 클라우드 서비스 없이 **완전한 로컬 환경**에서 동작합니다.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 데이터 미리보기 | 테이블 뷰, 스키마 검사, 결측값 히트맵 |
| 차트 빌더 | 선형·산점도·막대·히스토그램·박스플롯·상관관계 히트맵 |
| 시계열 분석 | 롤링 통계, 추세 감지, 이상치 탐지, 스무딩 |
| 주파수 분석 | FFT, 진폭/전력 스펙트럼, 피크 감지, 밴드 에너지, 스펙트로그램 |
| 이미지 분석 | 픽셀 통계, 채널 히스토그램, 주요 색상, AI 설명 |
| 예측 모듈 | Holt-Winters / ARIMA / Simple ES (statsmodels 기반) |
| AI 어시스턴트 | Ollama 기반 데이터 요약, 차트 제안, 패턴 설명, 자유 대화 |
| 멀티모달 분석 | LLaVA를 통한 실제 이미지 픽셀 기반 AI 설명 |
| 보고서 내보내기 | PDF (fpdf2) 및 HTML 자동 생성 |
| 오프라인 폴백 | Ollama 비활성 시에도 모든 분석 기능 정상 동작 |

---

## 전체 아키텍처

```
AIViz/
├── main.py                      # 데스크톱 앱 진입점 (PyQt6)
├── config.py                    # 전역 설정 (Ollama URL, 모델명, 앱 설정)
├── requirements.txt
│
├── aiviz/
│   ├── app/                     # 데스크톱 앱 인프라
│   │   ├── main_window.py       # QMainWindow – 탭/독/메뉴바/상태바
│   │   ├── controller.py        # AppController – 데이터 상태 및 신호 허브
│   │   └── style.py             # Catppuccin Mocha 다크 테마 (QSS + Matplotlib)
│   │
│   ├── ui/                      # PyQt6 패널 (UI 전용, 비즈니스 로직 없음)
│   │   ├── panel_file.py        # 파일 불러오기 사이드바
│   │   ├── panel_data.py        # 데이터 개요 탭
│   │   ├── panel_charts.py      # 차트 빌더 탭
│   │   ├── panel_timeseries.py  # 시계열 분석 탭
│   │   ├── panel_frequency.py   # 주파수 분석 탭
│   │   ├── panel_image.py       # 이미지 분석 탭
│   │   ├── panel_forecast.py    # 예측 탭
│   │   ├── panel_assistant.py   # AI 어시스턴트 탭
│   │   ├── panel_export.py      # 보고서 내보내기 탭
│   │   └── widgets/
│   │       ├── plot_widget.py   # Matplotlib 캔버스 래퍼 (내비게이션 툴바 포함)
│   │       └── data_table.py    # Pandas DataFrame → QTableView
│   │
│   ├── ingestion/               # 파일 파싱 계층
│   │   ├── loader.py            # CSV/Excel/JSON/Parquet/이미지 → DataLoadResult
│   │   └── schema.py            # 스키마 타입 추론 및 SchemaReport
│   │
│   ├── analytics/               # 순수 분석 로직 (UI 의존성 없음)
│   │   ├── summary.py           # 기술 통계, 상관관계
│   │   ├── timeseries.py        # 롤링 통계, 이상치 탐지
│   │   ├── frequency.py         # FFT, 밴드 통계, STFT
│   │   ├── image_analysis.py    # 픽셀 통계, 히스토그램
│   │   └── forecast.py          # Holt-Winters / ARIMA / Simple ES
│   │
│   ├── visualization/           # Matplotlib 차트 팩토리 (axes에 직접 렌더링)
│   │   └── mpl_charts.py        # plot_line, plot_fft_amplitude 등 모든 차트
│   │
│   ├── ai/                      # AI 에이전트 계층
│   │   ├── ollama_client.py     # Ollama REST API 클라이언트 (텍스트 + 멀티모달)
│   │   ├── agent.py             # AnalysisAgent – 프롬프트 → LLM → 결과
│   │   └── prompts.py           # 모든 프롬프트 템플릿 (분리 관리)
│   │
│   ├── export/                  # 보고서 생성 계층
│   │   ├── html_exporter.py     # 자체 완결 HTML (Base64 이미지 포함)
│   │   └── pdf_exporter.py      # PDF 보고서 (fpdf2 사용)
│   │
│   └── utils/
│       └── helpers.py           # 공통 유틸 (컬럼 타입 감지, 포맷팅 등)
│
├── examples/                    # 샘플 데이터
│   ├── sample_timeseries.csv    (500행, 멀티 신호, 이상치 포함)
│   ├── sample_sales.csv         (300행, 범주형 포함)
│   └── sample_frequency.csv     (2000행, 50+120+200 Hz 복합 신호)
│
├── tests/                       # pytest 테스트 (59개 통과)
└── docs/
    ├── MAINTENANCE.md
    └── DEVELOPER_GUIDE.md
```

**핵심 아키텍처 원칙:**
- `analytics/`, `visualization/`, `ai/`는 Streamlit/PyQt 임포트 없음
- 모든 패널은 `AppController` 신호를 통해서만 데이터를 수신
- Ollama 연결 실패 시 모든 기능은 폴백 모드로 정상 동작

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

### 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `PyQt6` | 데스크톱 UI 프레임워크 |
| `matplotlib` | Qt 내 차트 렌더링 |
| `pandas`, `numpy` | 데이터 처리 |
| `scipy` | FFT, 스펙트로그램, 스무딩 |
| `statsmodels` | 시계열 예측 |
| `Pillow` | 이미지 분석 |
| `httpx` | Ollama API 통신 |
| `fpdf2` | PDF 보고서 생성 |

---

## Ollama 설정 방법

### 1. Ollama 설치

[https://ollama.com/download](https://ollama.com/download)에서 운영체제에 맞는 버전을 설치합니다.

### 2. 서버 실행 및 모델 다운로드

```bash
# Ollama 서버 시작 (별도 터미널에서)
ollama serve

# 텍스트 분석 모델 (필수)
ollama pull llama3.2

# 이미지 멀티모달 분석 모델 (선택, 이미지 AI 설명에 필요)
ollama pull llava
```

### 3. 환경 변수 설정 (선택)

```bash
export OLLAMA_BASE_URL=http://localhost:11434   # 기본값
export OLLAMA_MODEL=llama3.2                   # 텍스트 모델
export OLLAMA_VISION_MODEL=llava               # 비전 모델
export OLLAMA_TIMEOUT=60                       # 요청 타임아웃(초)
```

---

## 실행 방법

```bash
python main.py
```

앱이 실행되면 다음과 같은 레이아웃의 데스크톱 창이 열립니다:

```
┌─────────────────────────────────────────────────────────┐
│ File  View  Help                                        │
├──────────┬──────────────────────────────────────────────┤
│ 파일     │  📊 Data  📈 Charts  📉 TimeSeries           │
│ 사이드바 │  🔊 Freq  🖼 Image  🔮 Forecast              │
│          │  🤖 AI Assistant  📄 Export                  │
│ [파일열기]│                                             │
│ [초기화] │  <선택된 탭의 분석 내용>                     │
├──────────┴──────────────────────────────────────────────┤
│ 상태바: Ready  |  🟢 Ollama  |  model: llama3.2        │
├─────────────────────────────────────────────────────────┤
│ 로그 (접을 수 있음)                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 사용 예시

### 1. 시계열 데이터 분석

```
1. 사이드바의 [파일 열기] 클릭
2. examples/sample_timeseries.csv 선택
3. 📊 Data 탭 → 스키마, 통계, 상관관계 확인
4. 📉 Time-Series 탭 → signal_a 선택, window=10으로 분석 실행
5. 이상치 탭에서 감지된 이상값 확인
6. 🤖 AI 버튼으로 추세 설명 요청
```

### 2. 주파수 분석

```
1. examples/sample_frequency.csv 로드
2. 🔊 Frequency 탭 선택
3. signal 컬럼 선택, Sample Rate = 1000 Hz 설정
4. Run FFT 클릭 → 50, 120, 200 Hz 피크 확인
5. Spectrogram 탭에서 시간-주파수 분포 시각화
```

### 3. 이미지 분석 (멀티모달 AI)

```
1. PNG/JPG 이미지 파일 로드
2. 🖼 Image 탭 → 픽셀 통계, 히스토그램 확인
3. Multimodal AI 탭 → 질문 입력 후 [LLaVA로 이미지 설명] 클릭
   (ollama pull llava 필요)
```

### 4. 시계열 예측

```
1. 시계열 데이터 로드
2. 🔮 Forecast 탭 선택
3. 신호 컬럼, Holt-Winters 방법, horizon=30 설정
4. Run Forecast 클릭 → 예측 차트 및 지표 확인
```

---

## 보고서 생성 방법

1. 데이터 파일 로드
2. **📄 Export** 탭으로 이동
3. 포함할 섹션 체크박스 선택
4. [AI 요약 생성] 버튼으로 AI 텍스트 자동 생성 (선택)
5. **[HTML로 내보내기]** 또는 **[PDF로 내보내기]** 클릭
6. 저장 경로 지정 후 완료

보고서에 포함 가능한 내용:
- 데이터셋 개요 (행/열 수, 결측값, 중복)
- 스키마 테이블 (컬럼 타입, 역할, null 비율)
- 기술 통계 (평균, 표준편차, 사분위수 등)
- AI 생성 요약 텍스트

---

## 지원 파일 형식

| 유형 | 확장자 |
|------|--------|
| 표 형식 데이터 | CSV, Excel (.xlsx/.xls), JSON, Parquet |
| 이미지 | PNG, JPG/JPEG, BMP, TIF/TIFF |

---

## 테스트 실행

```bash
python -m pytest tests/ -v
# 현재 59개 테스트 통과 (Ollama 없이 실행 가능)
```

---

## 한계점

- **Ollama 필요**: AI 기능은 `ollama serve` 실행 없이는 폴백 모드로만 동작
- **멀티모달 AI**: LLaVA 등 비전 모델이 별도로 다운로드되어 있어야 함
- **대용량 파일**: 50만 행 이상의 CSV는 미리보기 성능이 저하될 수 있음 (분석은 전체 데이터 대상)
- **스트리밍 UI**: AI 어시스턴트에서 토큰 스트리밍이 구현되어 있으나, 매우 큰 응답에서는 버벅임 발생 가능
- **PDF 수식**: fpdf2는 수식/특수문자 렌더링이 제한적 (reportlab으로 교체 가능)

---

## 향후 확장 방향

- [ ] 실시간 데이터 스트리밍 지원 (소켓/시리얼 포트)
- [ ] 플러그인 시스템으로 사용자 정의 분석 모듈 추가
- [ ] Docker 컨테이너 패키징
- [ ] 다중 파일 동시 비교 분석
- [ ] Prophet 기반 고급 예측 모듈
- [ ] 보고서 템플릿 시스템 (사용자 정의 레이아웃)
- [ ] 데이터 필터링/정렬 UI 강화
- [ ] 밝은 테마(라이트 모드) 지원

---

## 라이선스

MIT License
