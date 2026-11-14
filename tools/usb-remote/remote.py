"""Windows USB remote for Swallow Keeper. Sends explicit manual swallow events."""
import asyncio
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

ADB = os.environ.get("SWALLOW_ADB") or shutil.which("adb") or str(Path(os.environ.get("ANDROID_HOME", "")) / "platform-tools" / "adb.exe")
SERIAL = os.environ.get("SWALLOW_PHONE_SERIAL", "")
ACTION = "com.swallowkeeper.app.REMOTE_SWALLOW"
RECEIVER = "com.swallowkeeper.app/expo.modules.swallowvision.RemoteSwallowReceiver"

class RemoteLink:
    def __init__(self, publish):
        self.publish = publish
        self.connected = False
        self.sending = False
        self.connecting = False

    async def broadcast(self, *extras):
        process = await asyncio.create_subprocess_exec(ADB, *(["-s", SERIAL] if SERIAL else []), "shell", "am", "broadcast",
            "--receiver-foreground", "-n", RECEIVER, "-a", ACTION, *extras,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            creationflags=0x08000000)
        out, err = await asyncio.wait_for(process.communicate(), timeout=12)
        text = out.decode("utf-8", errors="replace")
        if process.returncode or "SecurityException" in text:
            raise RuntimeError(err.decode("utf-8", errors="replace") or text)
        return text

    async def connect(self):
        if self.connecting: return
        self.connecting = True
        try:
            result = await self.broadcast("--ez", "probe", "true")
            if "result=1" not in result: raise RuntimeError("USB 입력 기능이 포함된 앱 업데이트가 필요합니다.")
            self.connected = True
            self.publish("connected", "Android · USB")
        except Exception as error:
            self.connected = False
            self.publish("error", str(error))
        finally: self.connecting = False

    async def trigger(self):
        if self.sending: return
        self.sending = True
        try:
            request_id = secrets.token_hex(6)
            result = await self.broadcast("--es", "request_id", request_id)
            if "result=1" in result and "RECORDED" in result:
                self.publish("accepted", "삼킴 1회 기록 · 녹음 창 표시")
            elif "BUSY" in result:
                self.publish("ignored", "기준 발성·녹음·결과 창 중에는 새 입력을 받지 않습니다.")
            else: self.publish("error", "입력 처리를 확인하지 못했어요. 폰의 모니터링 상태를 확인해 주세요.")
        except Exception as error: self.publish("error", str(error))
        finally:
            self.sending = False
            self.publish("done", "")

    async def close(self): self.connected = False


class GlobalHotkey:
    def __init__(self, publish):
        self.publish = publish
        self.thread = None
        self.thread_id = None
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        self.user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
        self.user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        def listen():
            self.thread_id = self.kernel32.GetCurrentThreadId()
            key_file = Path(__file__).with_name("remote-key.json")
            selected = json.loads(key_file.read_text(encoding="utf-8"))["vk"] if key_file.exists() else 0x77
            keys = list(dict.fromkeys([0x77, selected]))
            registered = [0x534B+i for i, vk in enumerate(keys) if self.user32.RegisterHotKey(None, 0x534B+i, 0x4000, vk)]
            if not registered:  # MOD_NOREPEAT + F8
                self.publish("hotkey", "F8이 다른 프로그램에서 사용 중이에요. 이 창에서 Space를 누르세요.")
                return
            self.publish("hotkey", "F8 사용 가능 · 웹캠 창에서도 누를 수 있어요")
            msg = wintypes.MSG()
            try:
                while self.user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                    if msg.message == 0x0312 and msg.wParam in registered:
                        self.publish("key", "")
            finally:
                for registered_id in registered: self.user32.UnregisterHotKey(None, registered_id)
                self.thread_id = None
        self.thread = threading.Thread(target=listen, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread_id:
            self.user32.PostThreadMessageW(self.thread_id, 0x0012, 0, 0)


def run_window(auto_connect=False):
    root = tk.Tk()
    root.title("삼킴지킴이 · PC 리모컨")
    root.geometry("480x510")
    root.minsize(460, 500)
    root.configure(bg="#F5F9F8")
    events = queue.Queue()
    publish = lambda kind, message: events.put((kind, message))
    loop = asyncio.new_event_loop()
    worker = threading.Thread(target=loop.run_forever, daemon=True)
    worker.start()
    remote = RemoteLink(publish)
    hotkey = GlobalHotkey(publish)
    state = {"connected": False, "pending": False, "last": 0., "space": False, "count": 0}
    font = "맑은 고딕"
    tk.Label(root, text="삼킴지킴이", font=(font, 23, "bold"), fg="#278F9C", bg="#F5F9F8").pack(pady=(26, 4))
    tk.Label(root, text="블루투스 리모컨 → PC → USB → Android", font=(font, 12), fg="#48605F", bg="#F5F9F8").pack()
    instructions = "① Android를 USB로 연결\n② 기존 앱에서 모니터링 시작\n③ 리모컨 버튼 또는 F8 누르기"
    tk.Label(root, text=instructions, font=(font, 11), justify="left", bg="#F5F9F8", fg="#425C59").pack(pady=20)
    status = tk.StringVar(value="연결 대기 중")
    count = tk.StringVar(value="이 리모컨에서 확인된 입력 0회")
    key_status = tk.StringVar(value="연결 후 F8을 사용할 수 있습니다.")
    def connect():
        connect_button.config(state="disabled")
        future = asyncio.run_coroutine_threadsafe(remote.connect(), loop)
        future.add_done_callback(lambda _: publish("connect_done", ""))
    def send():
        now = time.monotonic()
        if not state["connected"] or state["pending"] or now - state["last"] < 1.5:
            return
        state["last"] = now
        state["pending"] = True
        send_button.config(state="disabled")
        asyncio.run_coroutine_threadsafe(remote.trigger(), loop)
    connect_button = ttk.Button(root, text="휴대폰 연결", command=connect)
    connect_button.pack(ipadx=20, ipady=6)
    send_button = tk.Button(root, text="삼킴 보내기  ·  Space / F8", command=send, state="disabled",
                            font=(font, 13, "bold"), bg="#278F9C", fg="white", activebackground="#217B86",
                            activeforeground="white", disabledforeground="#BEDBD9", relief="flat", padx=20, pady=14)
    send_button.pack(padx=30, pady=(14, 12), fill="x")
    tk.Label(root, textvariable=status, font=(font, 10), fg="#425C59", bg="#F5F9F8", wraplength=420).pack()
    tk.Label(root, textvariable=count, font=(font, 11, "bold"), fg="#278F9C", bg="#F5F9F8").pack(pady=(12, 6))
    tk.Label(root, textvariable=key_status, font=(font, 9), fg="#718580", bg="#F5F9F8", wraplength=420).pack()
    tk.Label(root, text="Space: 이 창에서 입력  ·  F8: 다른 창에서도 입력\n자동 감지 결과와 구분되는 수동 입력입니다.",
             font=(font, 9), fg="#718580", bg="#F5F9F8").pack(pady=12)
    def press_space(_):
        if not state["space"]:
            state["space"] = True
            send()
        return "break"
    def release_space(_):
        state["space"] = False
        return "break"
    root.bind("<KeyPress-space>", press_space)
    root.bind("<KeyRelease-space>", release_space)
    root.bind("<FocusOut>", lambda _: state.update(space=False))
    def pump():
        changed = False
        while True:
            try:
                kind, message = events.get_nowait()
            except queue.Empty:
                break
            changed = True
            if kind == "key":
                send()
            elif kind == "hotkey":
                key_status.set(message)
            elif kind == "connected":
                state["connected"] = True
                status.set(f"{message} 연결됨")
                hotkey.start()
            elif kind == "disconnected":
                state["connected"] = False
                hotkey.stop()
                status.set(message)
            elif kind == "accepted":
                state["count"] += 1
                count.set(f"이 리모컨에서 확인된 입력 {state['count']}회")
                status.set(message)
            elif kind == "done":
                state["pending"] = False
            elif kind != "connect_done":
                status.set(message[:250])
            if kind in ("connected", "disconnected", "done", "connect_done"):
                connect_button.config(state="disabled" if state["connected"] else "normal")
                send_button.config(state="normal" if state["connected"] and not state["pending"] else "disabled")
        if changed:
            snapshot = {"pid": os.getpid(), "connected": state["connected"], "pending": state["pending"],
                        "accepted_count": state["count"], "status": status.get(), "hotkey": key_status.get()}
            target = Path(__file__).with_name("connection-status.json")
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
            temporary.replace(target)
        root.after(80, pump)
    def close():
        hotkey.stop()
        try:
            asyncio.run_coroutine_threadsafe(remote.close(), loop).result(timeout=2)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(80, pump)
    publish("status", "연결 대기 중")
    if auto_connect:
        root.after(400, connect)
    root.mainloop()


if __name__ == "__main__":
    if "--learn-key" in sys.argv:
        user32 = ctypes.WinDLL("user32")
        candidates = [0x0D,0x20,0x21,0x22,0x25,0x26,0x27,0x28,0xAE,0xAF,0xB0,0xB1,0xB3,*range(0x70,0x88)]
        previous = {key: bool(user32.GetAsyncKeyState(key)&0x8000) for key in candidates}
        until = time.monotonic()+90
        while time.monotonic()<until:
            for key in candidates:
                down = bool(user32.GetAsyncKeyState(key)&0x8000)
                if down and not previous[key]:
                    Path(__file__).with_name("remote-key.json").write_text(json.dumps({"vk":key}),encoding="utf-8")
                    print(json.dumps({"learned_vk":key}),flush=True)
                    sys.exit(0)
                previous[key]=down
            time.sleep(.01)
        print("NO_REMOTE_KEY_RECEIVED")
    elif "--send" in sys.argv:
        async def send_once():
            remote=RemoteLink(lambda *event: print(json.dumps(event,ensure_ascii=False),flush=True))
            await remote.trigger()
        asyncio.run(send_once())
    else:
        run_window(auto_connect=True)
