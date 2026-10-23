# agent-elapsed

<p align="center">
  <img src="https://img.shields.io/github/stars/Vinger-lee/agent-elapsed?style=social" alt="Stars">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License: MIT">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/tests-237%20passing-22bb33" alt="237개 테스트 통과">
  <img src="https://img.shields.io/badge/runtime-numpy%20only-013243" alt="런타임: numpy만">
  <img src="https://img.shields.io/badge/model-compound%20Poisson%20%C3%97%20lognormal-8957e5" alt="모델">
  <br>
  <b>⏱️ 오래 걸리는 에이전트 작업에 정직한 예상 완료 시간(ETA) — 로컬에서 밀리초 단위로, LLM 루프 없이.</b>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <a href="../../README_CN.md">中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.ko.md">한국어</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.ru.md">Русский</a> ·
  <a href="README.ar.md">العربية</a>
</p>

`agent-elapsed`는 긴 작업을 도는 에이전트용 드롭인 경과 시간 위젯입니다. 작업이 시작되면 작업과 **병렬로**
"6–14분" 구간을 내고, 끝나면 예측과 실제가 같은 축에 그려집니다. 내부는 작업 훅, 상태 프로토콜, TUI 텍스트
렌더러, 줄 단위 JSON 브리지뿐이며 웹 페이지도 HTTP 서버도 없습니다. 런타임에는 **numpy 외에 아무것도**
필요하지 않습니다.

> **중요.** 모든 검증은 **합성 텔레메트리와 독립적 수치 실험**에 근거합니다. 저장소에 실제 로그가 없으므로
> 실제 로그에서 측정된 정확도는 주장하지 않습니다. `constants.json`은 콜드 스타트 사전(prior)이며 학습된
> 모델이 아닙니다. 피팅된 테이블이 보정 게이트를 통과하기 전까지 컴포넌트는 구간을 숨깁니다 — 사전은
> 검증된 예측으로 포장되지 않습니다.

---

## 전체 모델

작업은 `llm`, `tool`, `build`, `net` 네 연산 유형으로 묶인 단계들의 **직렬** 체인입니다. 특징 벡터 `g`와 잠재
클래스 `C = c`가 주어졌을 때:

```
N_j | C=c, g  ~  Poisson(λ_jc),     λ_jc = exp(clip(b_j + w_jᵀ g_c, -20, 20))
X_ji          ~  LogNormal(μ_j^eff, s_j²),   s_j² = σ_j² · max(φ, 1)
μ_j^eff       =  μ_j − (s_j² − σ_j²) / 2
T             =  Σ_j Σ_{i=1..N_j} X_ji
```

가정: 유형 내 단계 시간은 i.i.d.이고 개수와 독립이며, 모든 단계는 직렬·비중첩입니다.

**φ는 순수한 산포 파라미터입니다.** σ를 키우며 로그평균을 증가분의 절반만큼 낮추므로 평균은 φ와 무관하고
2차 모멘트만 커집니다:

```
E[X_j]   =  exp(μ_j + σ_j²/2)          (independent of φ)
E[X_j²]  =  E[X_j]² · exp(s_j²)        (grows with φ)
```

개수가 포아송이고 단계 시간과 독립이므로:

```
E[Σ_i X_ji]    = λ_jc · E[X_j]
Var(Σ_i X_ji)  = E[N_j]·Var(X_j) + Var(N_j)·E[X_j]²  =  λ_jc · E[X_j²]
```

분산이 아니라 2차 모멘트입니다: 복합 포아송의 분산은 `λ·E[X²]`. 클래스 불확실성은 전분산 법칙으로
다루고, **경로마다 잠재 클래스를 한 번만 추출**해 모든 유형이 공유합니다:

```
M_c  = Σ_j λ_jc · E[X_j]
V_c  = Σ_j λ_jc · E[X_j²]

E[T]   = Σ_c p_c M_c
Var(T) = Σ_c p_c V_c            (within-class)
       + Σ_c p_c (M_c − E[T])²  (between-class)
```

평균 레이트의 단일 포아송으로 접으면 클래스 간 항이 사라집니다(`tests/test_math_audit.py` 픽스처: 참 분산
**276**, 평균 레이트 모델은 20). 유형들은 **무조건부로는** 독립이 아닙니다.

**φ의 값.** φ에 의존하는 것은 2차 모멘트뿐이고 `exp(φ·σ_j²)`를 통해서만 들어가므로 평균 상대 분산은 단조
지수합입니다:

```
mean_t[ Var(T_t)/E[T_t]² ]  =  mean_t[ between-class relative variance ]     ← φ-free
                             + Σ_j B_j · exp(φ · σ_j²)
```

`B_j = mean_t[ λ̄_tj · E[X_j]² / E[T_t]² ]`, 목표 `r = mean_t[ ((y_t − E[T_t])/E[T_t])² ]`는 홀드아웃
잔차에서 오며 `[1, 25]`에서 이분법으로 풉니다: `f(1) ≥ 0` → φ = 1(이미 충분히 넓고, 단계 노이즈 아래로
좁힐 수는 없음), `f(25) ≤ 0` → φ = 25(클립, 경계로 보고), 그 외에는 CV²가 잔차 평균제곱과 **정확히** 같은
내부 해입니다. 등 σ에서는 `φ = 1 + ln(r)/σ²`로 교차 검증합니다.

---

## 분위수와 그 불확실성

분위수는 몬테카를로(기본 5000 경로)이고 어디에서도 정규 근사를 쓰지 않습니다. 0의 원자는 명시적으로
처리합니다:

```
P(T = 0) = Σ_c p_c · exp(−Σ_j λ_jc)
```

따라서 분포는 연속이 아닙니다. `predict()`는 `p50`, `p90` 옆에 정확 이항 순위 검정을 역산한 분포-무관
구간을 내는데, 분위수 아래 표본 수가 `Binomial(m, τ)`이므로

```
k = m·τ ,   h = z₀.₉₇₅ · √(m·τ·(1−τ))
interval = ( S_(k−h) , S_(k+h) )        # order statistics, clipped to 0..m−1
```

밀도 추정이 필요 없고, 원자에서는 좁은 띠 대신 구간이 한 점으로 축소됩니다.

측정된 동작 (`tests/test_math_upgrade.py`, m = 2000, 300회 독립 반복):

| τ | 명목 | 측정 커버리지 |
|---|---|---|
| 0.5 | 95 % | 0.960 |
| 0.9 | 95 % | 0.960 |

m = 5000에서 P50은 약 ±22초(SE ≈ 11초), P90은 약 ±89초(SE ≈ 46초)입니다 — 5000 경로의 해상도이며,
더 좁은 P90은 경로를 더 쓰는 대가입니다. `estimate_task`가 돌려주는 `p50_se`, `p90_se`가 이 표준오차입니다.

---

## 보정과 수용 게이트

후보는 **시간 분할** 홀드아웃에서 온라인 테이블을 이겨야 교체됩니다.

### 무엇을 측정하는가

* **무작위화 예측 순위(PIT).** `R = (#{Sᵢ < y} + u·(#{Sᵢ = y} + 1)) / (m + 1)`, `u ~ U(0,1)` 독립.
  교환가능성 아래 `R`은 **동점이 있어도, 분포가 완전히 퇴화해도** 균등합니다 — 0 원자가 안전한 이유입니다.
* **분위수 커버리지 브래킷.** 이산 분위수의 조건은 등식이 아니라 `P(Y < q_τ) ≤ τ ≤ P(Y ≤ q_τ)`이고,
  허용오차는 두 표준오차 `2√(τ(1−τ)/n)`입니다. 고정 ±0.05는 n = 20에서 겨우 1σ라 정당한 모델을 기각합니다.
* **PIT 표본의 KS**, 점근 p-값을 동반한 진단: 근사이지 증명이 아닙니다.
* **페어드 pinball 손실.** 같은 작업·같은 시드로 두 모델이 예측하므로 비교가 페어드입니다. `gate.evaluate`는
  `paired_diff`, `paired_diff_se`, `paired_z`도 내어 기각에 불확실성을 붙입니다.

### 임계값과 상태 전이

| 규칙 | 값 |
|---|---|
| 저장소의 완전한 작업 | ≥ 80 |
| 피팅 측 작업 | ≥ 60 |
| 홀드아웃 | `max(20, ceil(20 %))`, 항상 최근 구간 |
| 연산당 신규 단계 관측 | ≥ 50 |
| 산포 φ | `[1, 25]`로 클립, 이분법 |
| 수용 | 보정 통과 **그리고** 페어드 pinball이 더 나쁘지 않음 (`require_strict`는 엄격히 더 나음을 요구) |

후보는 임시 파일 + `os.replace`로 쓰고 이전 테이블은 `constants.prev.json`으로 남기며 `rollback`이
복원합니다. 기각된 후보는 라이브 테이블과 롤백 사본을 건드리지 않습니다. 텔레메트리가 부족하면 CLI는
사전을 쓰고 후보를 승격하지 않으며, "데이터 부족"은 종료 코드 2이지 조용한 통과가 아닙니다.

드리프트는 채점에 쓰는 같은 pinball 시계열에서 봅니다: `log1p`, 최근 100개 점의 EWMA, 고정 안정 기준선의
`2σ` 밴드, 연속 두 번의 이탈. 기준선은 검사 중인 데이터로 갱신하지 않습니다 — 그러지 않으면 영구적 이동이
기준선으로 옮겨가 스스로를 숨깁니다.

---

## 호스트에 내장

세 가지 접점, 하나의 생애주기 코어:

| 호스트 종류 | 접점 | 진입점 |
|---|---|---|
| Python 에이전트 | 훅 + 텍스트 렌더러 | `ElapsedHooks`, `render_tui(state)` |
| 네이티브 UI (Qt, SwiftUI, 웹 뷰, …) | 상태 구독 | `WidgetState` → 호스트 자체 위젯 |
| 그 밖의 언어 | 상주 프로세스, 한 줄에 JSON 하나 | `python -m agent_elapsed.bridge` |

표시 게이팅은 보수적입니다. 틀린 구간이 숨긴 구간보다 나쁘기 때문입니다.

* 기본 장기 작업 임계값은 **60초**이며 설정 가능합니다;
* 호스트가 `long_task=True`를 주면 즉시 표시합니다;
* 그렇지 않으면 **보정된** P50이, 또는 실제 경과 시간이 임계값을 넘을 때 나타납니다;
* `calibrating`인 동안에는 실제 경과 시간과 "calibrating"만 보이고 사전 구간과 과대 추정 경고는 숨겨집니다;
* 일반 채팅은 타이머도 예측도 시작하지 않습니다.

프로토콜과 스레드 경계는 [docs/integration.md](../../docs/integration.md)를 보세요.

---

## 오프라인 명령

```bash
python3 -m offline.cli refit    --store data --constants constants.json --env 'my-model|my-host'
python3 -m offline.cli report   --constants constants.json --env 'my-model|my-host' --out out/report.html
python3 -m offline.cli rollback --constants constants.json
```

| 종료 코드 | 의미 |
|---|---|
| `0` | 성공 |
| `1` | 수용 게이트에서 기각 |
| `2` | 데이터 부족 (JSON에 부족분과 서술적 검사) |
| `3` | 해당 환경의 텔레메트리 없음 |
| `4` | 롤백 사본 없음 |

---

## 텔레메트리 계약

두 테이블, 둘 다 JSONL, 둘 다 추가 전용이며 파생량(`X`, `n_j`)은 저장하지 않습니다.

* 단계 이벤트: `task_id`, `op`, `t_start`, `t_end`, `ok`, `tool`, `v`
* 작업 레코드: `task_id`, `g_pred`, `g_pred_probs`, `g_final`, `total_wall`, `censored`, `model_id`,
  `host`, `estimator_wall`, `g_schema_version`, `v`

`t_end > t_start`, 유한한 타임스탬프, 알려진 `op`, `(task_id, op, t_start)` 중복 금지는 코드가 강제합니다.
그 위의 규칙:

* 위반 행은 `rejects.jsonl`로 갑니다 — 조용히 버리지 않습니다. 버려지는 행은 대개 아주 길거나 이상한
  것이라, 버리면 평균이 검열과 같은 방향으로 낮아집니다;
* 피팅은 `g_pred`만 읽습니다; `g_final`은 레이블 누출이며, `g_final`을 조작해 두 번 피팅하고 계수가 같기를
  요구하는 테스트가 있습니다;
* 취소된 작업의 완료 단계는 단계 길이 피팅에 들어가지만(일어난 단계는 완전한 관측) 개수·커버리지·수용에서는
  빠집니다(합계가 하한일 뿐);
* 모든 task_id는 고유하며 단계와 작업은 그것으로 연결됩니다.

---

## 검증

```bash
python3 -B -m unittest discover -s tests -t .     # 237개
```

합성 텔레메트리 단대단(작업 400개, seed 61, 환경 `m-a|h-a`, 단계 이벤트 15 047개, 완전한 작업 351개,
홀드아웃 32개):

| 항목 | 결과 |
|---|---|
| 수용 게이트 | accepted |
| P50 / P80 / P90 상한 커버리지 | 0.594 / 0.844 / 0.969 |
| PIT KS D / p | 0.129 / 0.627 |
| 후보 페어드 pinball | 242.8 |
| 사전 페어드 pinball | 481.9 |
| 페어드 차 / z | −239.0 / −2.77 |
| 피팅된 φ | 1.0 (하한 클립: 테이블 자체 산포가 이미 홀드아웃을 덮음) |

`offline/synth.py`의 숨은 참값에 대한 단계 길이 복원:

| op | μ 피팅 / 참값 | σ 피팅 / 참값 | 단계 관측 수 |
|---|---|---|---|
| llm | 2.234 / 2.197 | 0.924 / 0.90 | 1789 |
| tool | 0.765 / 0.788 | 0.722 / 0.72 | 2613 |
| build | 5.305 / 5.347 | 1.097 / 1.10 | 339 |
| net | 1.760 / 1.792 | 0.860 / 0.88 | 721 |

5000 경로 예측 한 번은 약 **5 ms**입니다. 이 숫자들은 내부 일관성과 참값 복원을 보여줄 뿐 실제 에이전트
로그에 대한 증거가 아닙니다. 전체 기록: [docs/validation.md](../../docs/validation.md).

---

## 주장하지 않는 것

* **실제 작업의 참값은 없습니다.** 실제 로그는 포함되어 있지 않고 `constants.json`은 사전입니다.
* **직렬 합산은 wall time이 아닙니다.** 겹치는 호출, 중첩 타이머, 병렬 분기, 기록되지 않은 대기가 모델을
  깨뜨립니다. 병렬 작업에는 실행 그래프가 필요하고 이 버전은 그것을 만들어내지 않습니다.
* **조건부 독립은 가정이지 발견이 아닙니다.** 재시도 상관은 모델링하지 않습니다; φ는 산포 손잡이입니다.
* **파라미터는 점 추정입니다.** NIG 사후분포가 단계 파라미터를 축소하지만 예측은 점 추정을 쓰고, 개수 회귀와
  분류기 불확실성은 전파되지 않습니다.
* **취소는 피팅에 편향을 줄 수 있습니다** (소요 시간과 상관될 때). 생존 모델은 없고 위의 하한 인식 분할만
  있습니다.
* **몬테카를로 분위수는 표집 오차를 가집니다** — 이제 보고하지만 여전히 오차입니다.
* **KS p-값은 점근적입니다.** p > 0.05 근처의 진단이지 증명이 아닙니다.
* **분류기는 호스트의 것입니다.** 클래스 확률은 주어진 것으로 취급되며, 틀어지면 혼합도 틀어집니다.

---

## 문서

| 문서 | 내용 |
|---|---|
| [docs/design.md](../../docs/design.md) | 데이터 경계, 통계적 정의, 수용 판정 기준 |
| [docs/embedded-widget.md](../../docs/embedded-widget.md) | 호스트 TUI / 네이티브 UI 내장 |
| [docs/integration.md](../../docs/integration.md) | 통합 코드, 이벤트 프로토콜, 스레드 경계 |
| [docs/accuracy-audit.md](../../docs/accuracy-audit.md) | 유도, 보정 수정, 정확도 주장의 경계 |
| [docs/math-upgrade.md](../../docs/math-upgrade.md) | v1.1 수식 업그레이드: 유도, 검증, 비목표 |
| [docs/implementation-notes.md](../../docs/implementation-notes.md) | 구현 세부와 초기 초안 수정 |
| [docs/validation.md](../../docs/validation.md) | 재현 가능한 수용 기록 |
| [CHANGELOG.md](../../CHANGELOG.md) | 릴리스 이력 |
| [CONTRIBUTING.md](../../CONTRIBUTING.md) | 개발 워크플로, 테스트와 번역 규칙 |
| [SECURITY.md](../../SECURITY.md) | 위협 모델과 취약점 보고 방법 |

코드 식별자, 텔레메트리 필드 이름, JSON 키는 번역하지 않습니다 — 번역하면 호스트 통합이 깨집니다.

---

## 라이선스

[MIT](../../LICENSE) © agent-elapsed contributors