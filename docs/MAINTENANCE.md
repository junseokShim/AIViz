# AIViz 유지보수 가이드

AIViz 프로젝트를 유지·확장·디버깅하는 담당자를 위한 핵심 참조 문서입니다.

---

## 모듈 역할 및 의존 관계

| 모듈 | 역할 | 의존 방향 |
|------|------|-----------|
| `config.py` | 전역 설정 (Ollama URL, 모델명, 앱 설정) | 없음 (루트) |
| `aiviz/ingestion/loader.py` | 파일 파싱 → DataLoadResult | config |
| `aiviz/ingestion/schema.py` | 스키마·타입 추론 | pandas |
| `aiviz/analytics/summary.py` | 기술 통계, 상관관계 | schema |
| `aiviz/analytics/timeseries.py` | 롤링 통계, 추세, 이상치 | numpy, scipy |
| `aiviz/analytics/frequency.py` | FFT, 스펙트로그램, 밴드 통계 | numpy, scipy |
| `aiviz/analytics/image_analysis.py` | 픽셀 통계, 히스토그램 | Pillow |
| `aiviz/analytics/forecast.py` | Holt-Winters / ARIMA / Simple ES | statsmodels |
| `aiviz/visualization/mpl_charts.py` | Matplotlib 차트 팩토리 | matplotlib |
| `aiviz/ai/ollama_client.py` | Ollama REST API 클라이언트 | httpx, config |
| `aiviz/ai/prompts.py` | 프롬프트 템플릿 | 없음 |
| `aiviz/ai/agent.py` | LLM 호출 오케스트레이션 + 폴백 | client, prompts, utils |
| `aiviz/export/html_exporter.py` | HTML 보고서 생성 | ingestion, analytics |
| `aiviz/export/pdf_exporter.py` | PDF 보고서 생성 (fpdf2) | ingestion, analytics |
| `aiviz/app/controller.py` | AppController – 앱 상태 + Qt 신호 | ingestion |
| `aiviz/app/main_window.py` | QMainWindow – 탭/독/메뉴 | controller, panels |
| `aiviz/app/style.py` | QSS 스타일 + Matplotlib 테마 | 없음 |
| `aiviz/ui/panel_*.py` | 각 분석 탭 (Qt 렌더링만 담당) | analytics, visualization, ai |
| `aiviz/ui/widgets/*.py` | 재사용 가능한 Qt 위젯 | PyQt6, matplotlib |

**핵심 불변 규칙**: `analytics/`, `visualization/`, `ai/` 모듈은 절대 `PyQt6`, `streamlit` 등 UI 라이브러리를 임포트해선 안 됩니다.

---

## 확장 포인트

### 새 차트 추가

1. `aiviz/visualization/mpl_charts.py`에 `plot_<이름>(ax, ...)` 함수 추가
2. `aiviz/ui/panel_charts.py`의 `CHART_TYPES` 리스트에 이름 추가
3. `_generate_chart()` 메서드에 렌더링 분기 추가

```python
# mpl_charts.py 예시
def plot_violin(ax: Axes, df: pd.DataFrame, columns: list[str]) -> None:
    data = [df[c].dropna().values for c in columns]
    ax.violinplot(data, positions=range(len(columns)))
    ax.set_title("Violin Plot")
```

### 새 분석 함수 추가

1. `aiviz/analytics/` 내 적절한 파일 또는 새 파일에 함수 추가
2. `@dataclass` 결과 타입 정의 (복잡한 출력의 경우)
3. 해당 패널 `aiviz/ui/panel_*.py`에서 호출
4. `tests/test_analytics.py`에 단위 테스트 추가

### 새 AI 도구/프롬프트 추가

1. `aiviz/ai/prompts.py`에 프롬프트 빌더 함수 추가

```python
# prompts.py
def my_analysis_prompt(context: str, param: str) -> str:
    return f"You are AIViz...\n\nContext:\n{context}\n\nParameter: {param}"
```

2. `aiviz/ai/agent.py`의 `AnalysisAgent`에 메서드 추가

```python
def explain_my_analysis(self, context: str, param: str) -> AgentResult:
    prompt = prompts.my_analysis_prompt(context, param)
    return self._call(prompt, fallback="AI unavailable")
```

3. 해당 패널에서 `WorkerThread`로 비동기 호출

### 새 파일 형식 지원

1. `config.py`의 `APP.supported_tabular` 또는 `supported_image`에 확장자 추가
2. `aiviz/ingestion/loader.py`의 `_load_tabular()` 또는 `_load_image()`에 파싱 분기 추가
3. `tests/test_ingestion.py`에 테스트 케이스 추가

### 새 보고서 섹션 추가

1. `aiviz/export/html_exporter.py`의 `HTMLExporter`에 `add_<섹션명>()` 메서드 추가
2. `aiviz/export/pdf_exporter.py`의 `PDFExporter`에 동일한 메서드 추가
3. `aiviz/ui/panel_export.py`의 `_export_html()` / `_export_pdf()`에서 호출

---

## Ollama 모델 교체 방법

모든 모델 설정은 `config.py`에서 관리됩니다.

```python
# config.py
@dataclass
class OllamaConfig:
    default_model: str = "llama3.2"      # 변경 대상
    vision_model: str = "llava"          # 멀티모달 모델
    timeout: int = 60
    vision_timeout: int = 120
```

또는 환경 변수로 오버라이드:

```bash
OLLAMA_MODEL=mistral python main.py
OLLAMA_MODEL=deepseek-r1:7b python main.py
OLLAMA_VISION_MODEL=llava:13b python main.py
```

앱 실행 중에는 **AI 어시스턴트** 탭의 모델 드롭다운에서 실시간으로 변경 가능합니다.

**LLM 백엔드 교체 방법**: `aiviz/ai/ollama_client.py`만 교체하면 됩니다.
동일한 인터페이스(`generate`, `generate_with_image`, `is_healthy`, `list_models`, `stream`)를 구현하면 나머지 코드는 변경 불필요.

---

## 에러 처리 전략

### 데이터 로딩 오류
- 모든 파싱 오류는 `DataLoadResult(error=...)` 형태로 반환
- 절대 예외가 UI 레이어까지 전파되지 않도록 설계
- 패널은 `result.ok`를 확인한 후 데이터 접근

### AI/Ollama 오류
- `OllamaClient.generate()`는 절대 예외를 raise하지 않음
- `OllamaResponse(error=...)` 반환
- `AnalysisAgent._call()`은 `is_healthy()` 체크 후 폴백 분기 처리
- 모든 AI 버튼은 **Ollama 없이도 폴백 결과 표시**

### 분석 오류
- `analytics/` 함수는 잘못된 입력에 대해 명확한 `ValueError` 발생
- 패널은 분석 호출을 `try/except`로 감싸고 오류를 로그 독에 표시

### 스레딩 패턴
- 모든 장시간 작업(AI 호출, 예측, FFT)은 `WorkerThread` 사용
- UI 스레드 블로킹 금지
- 결과는 `result_ready(object)` 신호로 메인 스레드에 전달

---

## 로깅 전략

```python
import logging
logger = logging.getLogger("aiviz.<모듈명>")
```

- `logger.warning()`: 복구 가능한 문제 (모델 없음, 타임아웃)
- `logger.exception()`: 백그라운드 스레드의 예상치 못한 예외
- UI 패널에서는 `self._ctrl.log_message.emit(...)` 신호로 로그 독에 표시
- Streamlit처럼 콘솔 혼용 금지

---

## 테스트 전략

**프레임워크**: pytest  
**원칙**: Ollama 없이 모두 실행 가능해야 함

| 계층 | 테스트 유형 | 파일 |
|------|-------------|------|
| Ingestion | 인메모리 바이트 단위 테스트 | `tests/test_ingestion.py` |
| Analytics | 알려진 입력·출력 단위 테스트 | `tests/test_analytics.py` |
| Forecast | 통계 모델 결과 검증 | `tests/test_forecast.py` |
| Ollama Client | HTTP 모킹 테스트 | `tests/test_ollama_client.py` |
| Export | 파일 생성 및 내용 검증 | `tests/test_export.py` |
| UI 패널 | 미구현 (Streamlit 한계와 동일한 이유) | 수동 테스트 |

```bash
python -m pytest tests/ -v        # 전체 실행 (~1초, Ollama 불필요)
python -m pytest tests/ -k fft    # 특정 테스트만 실행
```

---

## 리팩토링 우선순위

현재 확장 시 우선적으로 처리해야 할 항목:

1. **세션 상태 관리**: `AppController`가 성장하면 `DataStore` / `AnalysisStore`로 분리 권장
2. **분석 캐싱**: 동일 컬럼·파라미터에 대한 반복 분석을 캐싱하면 성능 대폭 향상 가능 (예: `functools.lru_cache` 또는 `diskcache`)
3. **차트 레지스트리**: `CHART_TYPES` 딕셔너리 → `ChartPlugin` 데이터클래스 목록으로 전환 (10개 이상 시)
4. **Export 파이프라인**: 현재 패널별로 PNG를 수동 등록하는 방식 → 자동 캡처 파이프라인으로 개선 가능
5. **단일 mpl_charts.py 파일**: 함수가 15개를 넘으면 `mpl_charts_tabular.py`, `mpl_charts_frequency.py` 등으로 분리 권장

---

## 기술 부채 관리 포인트

| 항목 | 리스크 | 비고 |
|------|--------|------|
| `pd.to_datetime()` 포맷 미지정 | 낮음 | 대용량 컬럼에서 느릴 수 있음 |
| `select_dtypes(include=["object","string"])` | 낮음 | pandas 3.x 대응 확인 필요 |
| `fpdf2` 특수문자 처리 | 중간 | 비 Latin-1 문자는 `?`로 치환됨 |
| `WorkerThread` GC 방지 | 중간 | 패널은 `self._worker`로 참조 유지 필수 |
| AI 스트리밍 중 탭 전환 | 낮음 | StreamWorker 강제 종료 로직 개선 가능 |
| 비전 모델 JPEG 강제 변환 | 낮음 | PNG 직접 전송으로 개선 가능 |

---

## 성능 참고

| 작업 | 예상 시간 | 비고 |
|------|-----------|------|
| FFT (2000 samples) | < 10ms | 즉각 반응 |
| Holt-Winters (500pts, h=30) | 50–200ms | 백그라운드 처리 권장 |
| ARIMA (500pts) | 1–5초 | 반드시 백그라운드 처리 |
| Ollama (llama3.2:3b, M-chip) | 2–8초 | 모델/하드웨어 따라 다름 |
| LLaVA 이미지 설명 | 10–30초 | 이미지 크기, GPU 여부 따라 다름 |
