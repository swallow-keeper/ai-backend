from __future__ import annotations

import argparse

from swallow_tools.audio_samples import sine_wav

SCENARIOS = ("no_change", "recheck", "caution", "unanalyzable")


def run_scenario(client, token: str, name: str) -> str:
    if name not in SCENARIOS:
        raise ValueError(name)
    headers = {"Authorization": f"Bearer {token}"}
    consent = client.post("/me/consents", headers=headers, json={"item": "voice_analysis", "granted": True})
    _expect(consent, 201)
    started = client.post("/sessions", headers=headers)
    _expect(started, 201)
    session_id = started.json()["id"]
    clip = sine_wav(freq=200.0)
    baseline = client.post(
        f"/sessions/{session_id}/baseline",
        headers=headers,
        files={
            "clip_1": ("a.wav", clip, "audio/wav"),
            "clip_2": ("b.wav", clip, "audio/wav"),
            "clip_3": ("c.wav", clip, "audio/wav"),
        },
    )
    _expect(baseline, 200)
    if baseline.json()["stable"] is not True:
        raise RuntimeError("baseline unstable")
    event = client.post(
        f"/sessions/{session_id}/swallow-events",
        headers=headers,
        json={"trigger": "manual"},
    )
    _expect(event, 201)
    event_id = event.json()["id"]
    if name == "unanalyzable":
        voice = _upload(client, headers, event_id, sine_wav(amp=0.001))
        if voice["state"] != "unanalyzable":
            raise RuntimeError(voice["state"])
        return voice["state"]
    if name == "no_change":
        voice = _upload(client, headers, event_id, sine_wav(freq=200.0))
    else:
        voice = _upload(client, headers, event_id, sine_wav(freq=480.0))
    if name == "no_change" and voice["state"] != "no_change":
        raise RuntimeError(voice["state"])
    if name == "recheck" and voice["state"] != "recheck":
        raise RuntimeError(voice["state"])
    if name == "caution":
        if voice["state"] != "recheck":
            raise RuntimeError(voice["state"])
        voice = _upload(client, headers, event_id, sine_wav(freq=470.0))
        if voice["state"] != "caution":
            raise RuntimeError(voice["state"])
        resolved = client.post(f"/events/{event_id}/resolve", headers=headers)
        _expect(resolved, 200)
    return voice["state"]


def run_all(client, token: str, scenario: str = "all") -> list[tuple[str, str]]:
    names = SCENARIOS if scenario == "all" else (scenario,)
    return [(name, run_scenario(client, token, name)) for name in names]


def main() -> None:
    import httpx

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default="dev-token")
    parser.add_argument("--scenario", default="all", choices=("all", *SCENARIOS))
    args = parser.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=30) as client:
        for name, state in run_all(client, args.token, args.scenario):
            print(f"{name}: {state}")


def _upload(client, headers: dict[str, str], event_id: str, wav: bytes) -> dict:
    response = client.post(
        f"/events/{event_id}/voice-clip",
        headers=headers,
        files={"clip": ("v.wav", wav, "audio/wav")},
    )
    _expect(response, 200)
    return response.json()


def _expect(response, status: int) -> None:
    if response.status_code != status:
        raise RuntimeError(f"{response.status_code} {response.text}")
