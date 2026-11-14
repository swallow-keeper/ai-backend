"""Windowless USB remote. No startup registration or visible application window."""
import asyncio
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import queue
import time
from micro_input import MicroInput
from remote import GlobalHotkey, RemoteLink


async def run_background():
    folder = Path(__file__).parent
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes = [wintypes.LPVOID,wintypes.BOOL,wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    mutex = kernel.CreateMutexW(None,False,'Local\\SwallowKeeperUsbRemoteBackground')
    if not mutex: raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:
        kernel.CloseHandle(mutex)
        return
    stop = folder/'stop-background.request'
    stop.unlink(missing_ok=True)
    events = queue.Queue()
    def publish(kind,message): events.put((kind,message,time.monotonic()))
    remote = RemoteLink(publish)
    hotkey = GlobalHotkey(lambda kind,message:publish(kind,'F8' if kind=='key' else message))
    micro = MicroInput(publish)
    state = {'pid':os.getpid(),'background':True,'connected':False,'pending':False,'accepted_count':0,'input_received':0,'status':'USB 연결 확인 중'}
    target = folder/'connection-status.json'
    def save():
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(state,ensure_ascii=False),encoding='utf-8')
        temporary.replace(target)
    async def send():
        try: await remote.trigger()
        finally: state['pending'] = False
    last = -10.
    next_connect = 0.
    tasks = set()
    def task(coroutine):
        future = asyncio.create_task(coroutine)
        tasks.add(future)
        future.add_done_callback(tasks.discard)
    try:
        save(); hotkey.start(); micro.start()
        while not stop.exists():
            now = time.monotonic()
            if not remote.connected and not remote.connecting and now >= next_connect:
                next_connect = now+5
                task(remote.connect())
            changed = False
            while True:
                try: kind,message,at = events.get_nowait()
                except queue.Empty: break
                changed = True
                if kind == 'key':
                    state['input_received'] += 1
                    state['last_input_source'] = message
                    if now-at <= .5 and not state['pending'] and now-last >= 1.5:
                        last = now; state['pending'] = True; task(send())
                elif kind == 'accepted': state['accepted_count'] += 1; state['status'] = message
                elif kind == 'connected': state['connected'] = True; state['status'] = message+' 연결됨'
                elif kind == 'disconnected': state['connected'] = False; state['status'] = message
                elif kind == 'done': state['pending'] = False
                elif kind.startswith('micro'): state['micro'] = message; state['micro_ready'] = kind=='micro_ready'
                elif kind == 'hotkey': state['hotkey'] = message
                else: state['status'] = message
            if changed: save()
            await asyncio.sleep(.05)
    finally:
        hotkey.stop(); micro.stop()
        for pending in tasks: pending.cancel()
        if tasks: await asyncio.gather(*tasks,return_exceptions=True)
        await remote.close()
        state.update(connected=False,pending=False,status='종료됨')
        save(); stop.unlink(missing_ok=True); kernel.CloseHandle(mutex)


if __name__ == '__main__':
    asyncio.run(run_background())
