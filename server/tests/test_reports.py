from datetime import datetime, timezone

from swallow_server.tables import MealSession, User
from test_api import _app, _baseline, _session, _sine


def test_meal_reports_count_latest_result_and_honor_voice_setting(tmp_path):
    with _app(tmp_path) as client:
        headers, _ = _session(client)
        meal = client.post("/sessions", headers=headers, json={"meal": "breakfast"}).json()
        assert meal["meal"] == "breakfast"
        sid = meal["id"]
        _baseline(client, headers, sid)
        counted = client.post(f"/sessions/{sid}/swallow-events", headers=headers,
                              json={"trigger": "auto", "request_voice": False}).json()
        assert counted["prompt"] is False
        assert counted["phase"] == "TRACKING"
        event = client.post(f"/sessions/{sid}/swallow-events", headers=headers,
                            json={"trigger": "manual"}).json()
        for frequency in (480, 460):
            result = client.post(f"/events/{event['id']}/voice-clip", headers=headers,
                                 files={"clip": ("v.wav", _sine(freq=frequency), "audio/wav")})
            assert result.status_code == 200
        assert result.json()["state"] == "caution"
        report = client.get(f"/sessions/{sid}/report", headers=headers).json()
        assert report["session"]["swallow_count"] == 2
        assert report["session"]["caution_count"] == 1
        assert len(report["events"]) == 2
        assert report["events"][-1]["state"] == "caution"
        assert report["baseline_quality_flag"] == "ok"
        assert client.get("/sessions", headers=headers).json()[0]["meal"] == "breakfast"
        assert client.get("/home/summary", headers=headers).json()["caution_count"] == 1
        point = client.get("/reports/trend?period=day", headers=headers).json()["points"][0]
        assert point["caution_count"] == 1
        assert point["no_change_ratio"] == 0


def test_reports_are_private_and_validate_queries(tmp_path):
    with _app(tmp_path) as client:
        headers, sid = _session(client)
        with client.app.state.session_factory() as db:
            db.add(User(id="other", api_token="other-token", display_name="Other", role="patient",
                        created_at=datetime.now(timezone.utc)))
            db.commit()
        other = {"Authorization": "Bearer other-token"}
        assert client.get(f"/sessions/{sid}/report", headers=other).status_code == 404
        assert client.get("/sessions", headers=other).json() == []
        for path in ("/sessions", "/home/summary", "/reports/trend", f"/sessions/{sid}/report"):
            assert client.get(path).status_code == 401
        assert client.get("/reports/trend?period=invalid", headers=headers).status_code == 422
        assert client.get("/reports/trend?tz=invalid", headers=headers).status_code == 422
        assert client.post("/sessions", headers=headers, json={"meal": "invalid"}).status_code == 422


def test_calendar_grouping_and_periodic_check_is_not_a_swallow(tmp_path):
    with _app(tmp_path) as client:
        headers, sid = _session(client)
        with client.app.state.session_factory() as db:
            meal = db.get(MealSession, sid)
            meal.started_at = datetime(2026, 1, 4, 16, 0, tzinfo=timezone.utc)
            db.commit()
        event = client.post(f"/sessions/{sid}/swallow-events", headers=headers,
                            json={"trigger": "periodic"}).json()
        assert client.post(f"/sessions/{sid}/end", headers=headers).status_code == 200
        report = client.get(f"/sessions/{sid}/report", headers=headers).json()
        assert report["session"]["swallow_count"] == 0
        assert report["skipped_count"] == 1
        assert report["events"][0]["id"] == event["id"]
        day = client.get("/reports/trend?period=day", headers=headers).json()["points"][0]
        week = client.get("/reports/trend?period=week", headers=headers).json()["points"][0]
        utc = client.get("/reports/trend?period=day&tz=UTC", headers=headers).json()["points"][0]
        assert day["label"] == "2026-01-05"
        assert week["label"] == "2026-W02"
        assert utc["label"] == "2026-01-04"
        assert day["no_change_ratio"] is None
