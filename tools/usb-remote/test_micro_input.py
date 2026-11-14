import unittest
from micro_input import MicroKeys

class MicroTests(unittest.TestCase):
    def test_ordinary_keyboard_is_ignored(self):
        self.assertFalse(MicroKeys().press('HID#VID_1234&PID_5678',32,False))
    def test_only_micro_press_not_release_or_repeat(self):
        keys=MicroKeys(); device='HID#VID&00022DC8_PID&9021#KBD'
        self.assertTrue(keys.press(device,32,False))
        self.assertFalse(keys.press(device,32,False))
        self.assertFalse(keys.press(device,32,True))
        self.assertTrue(keys.press(device,32,False))
    def test_other_8bitdo_device_is_ignored(self):
        self.assertFalse(MicroKeys().press('HID#VID_2DC8&PID_6000',32,False))

if __name__=='__main__': unittest.main()
