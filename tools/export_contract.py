"""Regenerate the HTTP contract from the implemented FastAPI routes."""
from pathlib import Path

import yaml

from swallow_server.main import create_app

ROOT = Path(__file__).resolve().parents[1]
schema = create_app().openapi()
schema["info"]["description"] = (
    "연구용 음성 변화 분석 API. 흡인을 진단하지 않습니다. 기본 앱은 이 API 대신 "
    "온디바이스 분석을 사용하며, 서버 모드는 개발 환경에서 명시적으로 선택합니다."
)
schema.setdefault("components", {}).setdefault("securitySchemes", {})["bearerAuth"] = {
    "type": "http", "scheme": "bearer"
}
schema["security"] = [{"bearerAuth": []}]
(ROOT / "contracts/openapi.yaml").write_text(
    yaml.safe_dump(schema, allow_unicode=True, sort_keys=False), encoding="utf-8"
)
