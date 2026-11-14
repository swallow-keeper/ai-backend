"""Windows background input restricted to the Micro's keyboard interface."""
import ctypes as c
from ctypes import wintypes as w
import os
import threading


class MicroKeys:
    def __init__(self):
        self.down = set()

    def press(self, device, key, released):
        name = device.lower()
        if '2dc8' not in name or '9021' not in name:
            return False
        identity = (device, key)
        if released:
            self.down.discard(identity)
            return False
        if key in (0, 255, 16, 17, 18, 91, 92) or identity in self.down:
            return False
        self.down.add(identity)
        return True


class MicroInput:
    def __init__(self, publish):
        self.publish = publish
        self.hwnd = None
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.thread.start()

    def listen(self):
        u = c.WinDLL('user32', use_last_error=True)
        k = c.WinDLL('kernel32', use_last_error=True)
        proc_type = c.WINFUNCTYPE(c.c_ssize_t, w.HWND, w.UINT, w.WPARAM, w.LPARAM)
        class WC(c.Structure):
            _fields_ = [('style', w.UINT), ('proc', proc_type), ('ce', c.c_int), ('we', c.c_int),
                        ('instance', w.HINSTANCE), ('icon', w.HANDLE), ('cursor', w.HANDLE),
                        ('brush', w.HANDLE), ('menu', w.LPCWSTR), ('name', w.LPCWSTR)]
        class RID(c.Structure):
            _fields_ = [('page', w.USHORT), ('usage', w.USHORT), ('flags', w.DWORD), ('target', w.HWND)]
        class Header(c.Structure):
            _fields_ = [('kind', w.DWORD), ('size', w.DWORD), ('device', w.HANDLE), ('param', w.WPARAM)]
        class Keyboard(c.Structure):
            _fields_ = [('make', w.USHORT), ('flags', w.USHORT), ('reserved', w.USHORT),
                        ('key', w.USHORT), ('message', w.UINT), ('extra', w.ULONG)]
        u.DefWindowProcW.argtypes = [w.HWND,w.UINT,w.WPARAM,w.LPARAM]
        u.DefWindowProcW.restype = c.c_ssize_t
        u.CreateWindowExW.argtypes = [w.DWORD,w.LPCWSTR,w.LPCWSTR,w.DWORD,c.c_int,c.c_int,c.c_int,c.c_int,w.HWND,w.HMENU,w.HINSTANCE,w.LPVOID]
        u.CreateWindowExW.restype = w.HWND
        u.RegisterClassW.argtypes = [c.POINTER(WC)]
        u.GetRawInputData.argtypes = [w.HANDLE,w.UINT,w.LPVOID,c.POINTER(w.UINT),w.UINT]
        u.GetRawInputDeviceInfoW.argtypes = [w.HANDLE,w.UINT,w.LPVOID,c.POINTER(w.UINT)]
        u.RegisterRawInputDevices.argtypes = [c.POINTER(RID),w.UINT,w.UINT]
        u.GetMessageW.argtypes = [c.POINTER(w.MSG),w.HWND,w.UINT,w.UINT]
        u.DispatchMessageW.argtypes = [c.POINTER(w.MSG)]
        u.DestroyWindow.argtypes = [w.HWND]
        k.GetModuleHandleW.restype = w.HMODULE
        names, keys = {}, MicroKeys()
        def device_name(handle):
            if handle not in names:
                size = w.UINT()
                u.GetRawInputDeviceInfoW(handle,0x20000007,None,c.byref(size))
                buf = c.create_unicode_buffer(size.value+1)
                u.GetRawInputDeviceInfoW(handle,0x20000007,buf,c.byref(size))
                names[handle] = buf.value
            return names[handle]
        @proc_type
        def procedure(hwnd, message, wp, lp):
            try:
                if message == 0x00FF:
                    size = w.UINT()
                    u.GetRawInputData(lp,0x10000003,None,c.byref(size),c.sizeof(Header))
                    data = c.create_string_buffer(size.value)
                    result = u.GetRawInputData(lp,0x10000003,data,c.byref(size),c.sizeof(Header))
                    if result != 0xFFFFFFFF and size.value >= c.sizeof(Header)+c.sizeof(Keyboard):
                        head = Header.from_buffer_copy(data)
                        if head.kind == 1:
                            key = Keyboard.from_buffer_copy(data,c.sizeof(Header))
                            if keys.press(device_name(head.device),key.key,bool(key.flags & 1)):
                                self.publish('key','8BitDo Micro')
                elif message == 0x0010:
                    u.DestroyWindow(hwnd)
                    return 0
                elif message == 0x0002:
                    u.PostQuitMessage(0)
                    return 0
            except Exception as error:
                self.publish('micro_error',str(error))
            return u.DefWindowProcW(hwnd,message,wp,lp)
        try:
            name = f'SwallowMicroInput_{os.getpid()}'
            instance = k.GetModuleHandleW(None)
            wc = WC(0,procedure,0,0,instance,None,None,None,None,name)
            if not u.RegisterClassW(c.byref(wc)):
                raise c.WinError(c.get_last_error())
            self.hwnd = u.CreateWindowExW(0,name,'',0,0,0,0,0,w.HWND(-3),None,instance,None)
            if not self.hwnd:
                raise c.WinError(c.get_last_error())
            device = RID(1,6,0x100,self.hwnd)
            if not u.RegisterRawInputDevices(c.byref(device),1,c.sizeof(RID)):
                raise c.WinError(c.get_last_error())
            self.publish('micro_ready','8BitDo Micro 백그라운드 입력 대기')
            msg = w.MSG()
            while u.GetMessageW(c.byref(msg),None,0,0)>0:
                u.TranslateMessage(c.byref(msg))
                u.DispatchMessageW(c.byref(msg))
        except Exception as error:
            self.publish('micro_error',str(error))
        finally:
            self.hwnd = None

    def stop(self):
        if self.hwnd:
            u = c.WinDLL('user32')
            u.PostMessageW.argtypes = [w.HWND,w.UINT,w.WPARAM,w.LPARAM]
            u.PostMessageW(self.hwnd,0x10,0,0)
        if self.thread:
            self.thread.join(timeout=2)
