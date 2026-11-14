# PC → USB 삼킴 리모컨

블루투스 리모컨의 키를 Windows PC가 받아 USB로 연결된 Android 앱에 수동 삼킴을 전달합니다. 휴대폰에 연결 버튼이나 새로운 안내를 추가하지 않습니다.

## 실행

Python 3.11 이상(Tk 포함)과 Android SDK Platform Tools가 필요합니다. `adb`를 PATH에 넣거나 `SWALLOW_ADB`에 실행 파일 경로를 지정합니다. 외부 Python 패키지는 필요하지 않습니다.

```powershell
# 여러 Android 장치가 연결된 경우에만 대상 기기의 adb 시리얼을 지정합니다.
$env:SWALLOW_PHONE_SERIAL = 'YOUR_DEVICE_SERIAL'
python remote.py
```

휴대폰에서 USB 디버깅을 허용하고, 이 저장소의 네이티브 앱을 설치한 뒤 기존 흐름대로 기준 발성과 모니터링을 시작합니다. PC 프로그램은 시작 시 USB 연결을 확인합니다. F8은 다른 창에서도, Space는 리모컨 창에서 사용할 수 있습니다.

단일 요청 전달 시험:

```powershell
python remote.py --send
python -m unittest -v test_usb
```

앱이 실제 이벤트를 생성하여 `RECORDED`를 응답한 경우에만 PC의 확인 횟수가 증가합니다. 녹음·분석·기준 발성·팝업 중이거나 앱이 배경에 있으면 입력을 무시합니다. 실패·재연결 시 명령을 자동 재전송하지 않습니다. 원격 이벤트는 `manual`로 저장하며 자동 감지 모델과 음성 분석 기준은 변경하지 않습니다.

## 현재 검증 범위

- Android 릴리스 빌드, S22 설치와 승인된 ADB 셸의 명령·응답을 확인했습니다.
- 앱 모의 입력 테스트는 수동 삼킴 1회, 녹음 창, 보고서 저장 및 중복 방지를 확인합니다.
- 실제 8BitDo Micro 버튼 매핑과 버튼부터 녹음 창까지의 실기기 시험은 진행 중입니다. F8 경로와 구분합니다.
- `--learn-key`는 일반 가상 키 등록을 위한 초안입니다. Micro의 기본 문자 키 매핑을 모두 처리하지 않으므로 완성된 Micro 연결 절차로 보지 않습니다.

수신기는 Android의 `DUMP` 권한으로 보호합니다. 일반 앱에서 임의로 브로드캐스트를 보내 삼킴을 만들 수 없으며, 승인된 ADB 연결을 전제로 합니다.
