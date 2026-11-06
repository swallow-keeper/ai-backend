# 삼킴지킴이 AI 백엔드

비임상 연구용 프로토타입입니다. 흡인을 진단하지 않으며, 임상 성능 수치를 제시하지 않습니다.

식사 전 "아~" 발성 3회를 기준으로 삼고, 삼킴 직후 2초 발성이 그 기준에서 얼마나 달라졌는지를 점수로 냅니다. 결과는 변화 없음, 재확인, 주의, 분석 불가 중 하나입니다. 점수는 흡인 확률이 아닙니다.

영상은 서버로 받지 않습니다. 서버는 삼킴 후보 이벤트(시각, 신뢰도, 트리거)와 음성 클립만 받습니다. 원본 음성은 분석 직후 메모리에서 폐기하고 디스크와 로그에 남기지 않습니다.

## 구성

```text
shared/     전처리, 품질 검사, 임베딩, 변화 점수
server/     FastAPI, 세션, 상태 기록
ml/         학습 실험 자리, 1차에서는 카메라 학습 코드를 두지 않음
tools/      Mock 서버, 재생 클라이언트
contracts/  OpenAPI, WebSocket 스키마
docs/       녹음 사양, 상태 코드, 기기 내 영상 처리 전제
registry/   모델 버전, 임계값
```

학습과 서버는 `swallow_shared`만 사용해 전처리를 맞춥니다.

성능 표기가 필요하면 "개발 내부 검증, 임상 성능 아님"을 붙입니다. 변화 점수는 흡인 확률이 아니고, 분류 head는 임상 라벨이 생기기 전까지 꺼져 있습니다. `registry/model_registry.yaml`의 임계값은 가설값입니다.

## 실행

```bash
uv sync
uv run uvicorn swallow_server.main:app --host 127.0.0.1 --port 8000
```

기본 토큰은 `dev-token`입니다. 식사 세션의 기준 음성과 삼킴 후 발성을 분석하려면 먼저 `POST /me/consents`로 `voice_analysis` 동의를 남깁니다.

프론트 계약은 `contracts/openapi.yaml`과 `contracts/ws`입니다. 녹음 형식은 `docs/recording.md`, 결과 문구 초안과 사유 코드는 `docs/states.md`입니다. 영상은 올리지 않습니다. 서버 영상 폴백은 구현하지 않았고, 조건만 `docs/g0-fallback.md`에 적혀 있습니다.

```bash
uv run swallow-mock
uv run swallow-replay --base-url http://127.0.0.1:8000
```

`swallow-mock`은 8001 포트에서 고정 응답만 돌려줍니다. 토큰은 `mock-token`입니다. 분석 결과가 아닙니다. `swallow-replay`는 떠 있는 분석 서버에 합성 발성을 넣어 변화 없음, 재확인, 주의, 분석 불가를 확인합니다. 이 확인은 개발 내부 검증이며 임상 성능이 아닙니다.

스키마를 바꿀 때는 `server`에서 Alembic을 사용합니다.

```bash
uv run alembic -c server/alembic.ini upgrade head
```
