# API 계약

프론트 타입 생성용입니다. REST 계약은 `uv run python tools/export_contract.py`로 구현에서 재생성합니다. 스키마 변경 후 계약과 테스트를 함께 갱신합니다.

- `openapi.yaml`: REST
- `ws/*.json`: 서버가 앱으로 보내는 WebSocket 메시지

WebSocket 경로: `/ws/sessions/{session_id}`

인증은 `Authorization: Bearer` 헤더 또는 쿼리 `token`입니다. 토큰이 없거나 세션이 아니면 연결을 거절합니다.

1차에서 앱이 서버로 보내는 삼킴 후보는 REST `POST /sessions/{id}/swallow-events`입니다. 카메라 특징 시계열과 영상 프레임은 이 계약에 없습니다.

연결 직후 서버는 `RESYNC`를 보냅니다. 발성이 필요하면 `PROMPT_VOICE`, 분석이 끝나면 `RESULT`, 주의로 올라가면 `ALERT`를 추가로 보냅니다. `STATE`는 세션 단계가 바뀔 때 보냅니다.

`RESULT.result_id`는 같은 분석의 HTTP 응답과 WS 푸시에서 동일합니다. 발성을 다시 녹음하면 상태와 사유가 같아도 새로운 ID가 생깁니다.

목 미검출 안내는 기기 안에서 처리합니다. 서버 `STATE`는 목이 보이는지 판단하지 않습니다.
