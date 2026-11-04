# API 계약

프론트 타입 생성용입니다. 구현은 이 파일을 기준으로 맞춥니다.

- `openapi.yaml`: REST
- `ws/*.json`: 서버가 앱으로 보내는 WebSocket 메시지

WebSocket 경로: `/ws/sessions/{session_id}`

1차에서 앱이 서버로 보내는 삼킴 후보는 REST `POST /sessions/{id}/swallow-events`입니다. 카메라 특징 시계열과 영상 프레임은 이 계약에 없습니다.

연결 직후 서버는 `RESYNC`를 보냅니다. 발성이 필요하면 `PROMPT_VOICE`, 분석이 끝나면 `RESULT`, 주의로 올라가면 `ALERT`를 추가로 보냅니다. `STATE`는 세션 단계가 바뀔 때 보냅니다.

목 미검출 안내는 기기 안에서 처리합니다. 서버 `STATE`는 목이 보이는지 판단하지 않습니다.
