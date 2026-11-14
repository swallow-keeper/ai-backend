import unittest
from unittest.mock import AsyncMock
from remote import RemoteLink

class UsbTests(unittest.IsolatedAsyncioTestCase):
    async def test_counts_only_confirmed_recording(self):
        events=[]
        remote=RemoteLink(lambda *e: events.append(e))
        remote.broadcast=AsyncMock(return_value='Broadcast completed: result=1, data="RECORDED"')
        await remote.trigger()
        self.assertEqual(sum(e[0]=='accepted' for e in events),1)
        remote.broadcast.assert_awaited_once()
    async def test_busy_does_not_count_or_retry(self):
        events=[]
        remote=RemoteLink(lambda *e: events.append(e))
        remote.broadcast=AsyncMock(return_value='Broadcast completed: result=0, data="BUSY"')
        await remote.trigger()
        self.assertFalse(any(e[0]=='accepted' for e in events))
        remote.broadcast.assert_awaited_once()
    async def test_failure_releases_pending_without_replay(self):
        events=[]
        remote=RemoteLink(lambda *e: events.append(e))
        remote.broadcast=AsyncMock(side_effect=RuntimeError('USB disconnected'))
        await remote.trigger()
        self.assertFalse(remote.sending)
        self.assertIn(('done',''),events)
        remote.broadcast.assert_awaited_once()

if __name__=='__main__': unittest.main()
