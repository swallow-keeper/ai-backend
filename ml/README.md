# 로컬 삼킴 후보 모델 학습

이 패키지는 영상에서 삼킴 후보를 학습합니다. 흡인 여부를 학습하거나 진단하는 모델이 아닙니다. 실제 데이터와 배포 가중치는 제공되지 않습니다.

## 설치

Python 3.12 가상환경에서 실행하세요. 영상 추출 의존성은 서버 기본 설치와 분리되어 있습니다.

```sh
uv venv --python 3.12 .vision-venv
uv pip install --python .vision-venv/bin/python -e shared -e 'ml[vision]'
```

Windows에서는 `.vision-venv/bin/python` 대신 `.vision-venv/Scripts/python.exe`를 사용합니다. 이후 명령도 이 가상환경을 활성화한 뒤 실행합니다.

## 데이터 준비

사용 권한을 확보한 로컬 영상에 삼킴 **동작 완료 시각**을 밀리초로 주석합니다. 얼굴과 목이 모두 보여야 하며 원본 fps가 8 이상이어야 합니다. 피험자 ID는 익명 ID로 사용하세요. 씹기·말하기·고개 움직임·기침·휴식도 음성 후보가 아닌 예제로 포함해야 합니다.

`data/annotations.json` 예시 형식:

```json
[
  {"subject_id": "subject-001", "video": "subject-001-meal.mp4", "swallows_ms": [5420, 12300]},
  {"subject_id": "subject-002", "video": "subject-002-meal.mp4", "swallows_ms": [7200, 16800]}
]
```

위 숫자는 형식 예시입니다. 실제 영상 관찰 결과로 대체해야 합니다. 개인 내 음성 비교가 흡인을 반영하는지 평가하려면 별도로 VFSS/FEES 등 실제 흡인 여부와 대응된 데이터가 필요합니다.

## 특징 추출

앱에서 `python scripts/prepare_vision.py`를 실행하면 공식 얼굴 모델이 준비됩니다. 다음 명령은 백엔드 저장소에서 실행합니다.

```sh
python -m swallow_ml.motion data/annotations.json \
  --face-model ../app/.vision-cache/face_landmarker.task \
  --output data/motion-windows.jsonl
```

MediaPipe 랜드마크로 얼굴·목 ROI를 정하고 OpenCV LK 광류를 구합니다. 모바일과 같은 `landmark-lk6-v1` 특징을 8 Hz·16프레임 창으로 출력합니다. 얼굴 소실·큰 고개 회전·프레임 공백은 창을 폐기합니다. 삼킴 직후 250ms 이내의 창을 양성으로 삼고, 삼킴에서 1초 이내인 애매한 음성 창은 제외합니다.

입력 영상과 주석은 로컬에서만 읽습니다. 원본 영상이나 음성을 업로드하지 않습니다. `data/`와 원본 미디어를 Git에 넣지 마세요.

## 학습 및 앱 내보내기

```sh
python -m swallow_ml.temporal data/motion-windows.jsonl \
  --epochs 100 --seed 42 \
  --output ../app/assets/models/swallow-tcn.json
```

- 피험자 기준으로 학습·검증·테스트를 나눕니다. 같은 사람의 겹치는 시간 창을 여러 분할에 섞지 않습니다.
- 최소 6명의 피험자와 각 분할의 양성·음성 예제가 필요합니다. 이는 코드의 분할 최소 조건일 뿐, 충분한 연구 표본 수나 임상 검증 기준이 아닙니다.
- 특징 정규화는 학습 분할에서만 계산합니다. 모델 선택과 임계값 선택은 검증 분할만 사용합니다. 테스트 분할로 튜닝하지 않습니다.
- 모델은 causal convolution 두 층(dilation 1·2), ReLU, 평균 pooling, sigmoid입니다. NumPy로 학습하며 앱은 동일 그래프의 JSON 가중치를 실행합니다.
- 내보낸 파일에는 데이터 SHA256, 분할별 피험자 ID, 검증·테스트 지표, 학습 설정이 들어갑니다.
- 결과의 precision/recall/F1은 창 단위 연구 지표입니다. 임상 성능, 흡인 성능, 실제 식사에서의 시간당 오탐률로 해석하면 안 됩니다. 식사 단위 이벤트 매칭·오탐률·기기별 성능은 추가 평가해야 합니다.
- 충분히 평가한 가중치를 내보낸 뒤 앱을 다시 빌드합니다. 모델이 없거나 형식이 틀리면 앱은 자동 감지를 시작하지 않습니다.

## 식사 영상 전체 재생 평가

검증 분할에서 모델과 임계값 선택을 마친 뒤, 학습·검증에 사용하지 않은 피험자의 전체 식사 영상으로 평가합니다. 이 단계에서는 삼킴 근처의 애매한 창이나 추적 실패 구간을 지우지 않습니다.

```sh
python -m swallow_ml.evaluate data/heldout-meals.json \
  --model ../app/assets/models/swallow-tcn.json \
  --face-model ../app/.vision-cache/face_landmarker.task \
  --output data/heldout-event-report.json
```

입력 manifest는 특징 추출 때와 같은 형식입니다. 학습·검증 피험자를 포함하거나 같은 영상을 중복 지정하면 거절합니다. 앱의 16프레임 시간 창 선택·프레임 공백 처리·2500ms cooldown을 재현하고, 검출을 실제 주석에 일대일로 대응시킵니다. 놓친 삼킴(FN), 중복 검출·오탐(FP), precision/recall/F1, 시간당 오탐, 검출 지연, 추적 가능한 샘플 수를 기록합니다. 얼굴·목을 추적하지 못한 구간의 실제 삼킴도 FN에 포함합니다.

기본 매칭 범위는 주석보다 250ms 이전부터 1000ms 이후까지입니다. `--early-ms`, `--late-ms`로 변경할 수 있으며 평가 전에 정하고 결과의 설정과 함께 보고해야 합니다. 정답이나 예측이 없는 분모의 지표는 100%로 채우지 않고 `null`로 남깁니다.

이 도구는 연속 영상에서 카메라 분류기와 시간 정책을 평가합니다. 앱의 발성 팝업 동안 카메라를 쉬는 동작, 휴대폰 프레임 처리 속도·발열·배터리를 재현하지 않으며 실기기 측정을 대체하지 않습니다. 평가 결과에는 모델·manifest SHA256을 남기며 흡인/임상 검증 결과로 표시하지 않습니다.

## 개발 검증

```sh
uv run pytest ml/tests
```

테스트는 수치 미분으로 학습 gradient를 확인하고, 피험자 분리와 합성 데이터에서의 학습 동작을 검증합니다. 합성 데이터로 생성한 가중치는 앱에 배포하지 않습니다.

`ml/tests/write_parity_fixture.py`는 TypeScript 추론과 비교할 **테스트 전용** 수치 벡터를 만듭니다. 앱의 `tests/temporal.test.ts`가 Python 계산과의 일치를 검증합니다. 이 파일은 학습 완료 모델이 아닙니다.
