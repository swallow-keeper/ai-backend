import ctypes
from ctypes import wintypes
import os
import queue
import unittest
from micro_input import MicroInput

@unittest.skipUnless(os.name=='nt','Windows input API')
class WindowsInputTest(unittest.TestCase):
    def test_background_input_registers_without_visible_window(self):
        events=queue.Queue()
        listener=MicroInput(lambda *event:events.put(event))
        try:
            listener.start()
            kind,message=events.get(timeout=5)
            self.assertEqual(kind,'micro_ready',message)
            user32=ctypes.WinDLL('user32')
            user32.IsWindowVisible.argtypes=[wintypes.HWND]
            self.assertEqual(user32.IsWindowVisible(listener.hwnd),0)
        finally: listener.stop()
        self.assertFalse(listener.thread.is_alive())
