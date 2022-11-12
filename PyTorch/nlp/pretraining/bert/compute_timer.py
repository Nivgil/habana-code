import torch
import utils
import time
from collections import deque


class DeviceTimer(object):
    def _get_hpu_module(self):
        if self.habana_module is None:
            import habana_frameworks.torch as ht
            self.habana_module = ht

    def __init__(self, use_hpu=False, event_sync_delay=0):
        self.start_time = None
        self.events = deque(maxlen=event_sync_delay+1)  # sync on the previous event according to parameter
        self.habana_module = None
        self.use_hpu = use_hpu
        if (use_hpu):
            self._get_hpu_module()

    def _sync(self):
        if self.use_hpu:
            sync_event = self.habana_module.hpu.Event(enable_timing=True)
            sync_event.record()
            self.habana_module.hpu.current_stream().wait_event(sync_event)
        else:
            sync_event = torch.cuda.Event(enable_timing=True)
            sync_event.record()
            torch.cuda.current_stream().wait_event(sync_event)

        self.events.append(sync_event)
        wait_event = self.events[0]
        wait_event.synchronize()
        #self.habana_module.hpu.current_stream().synchronize()

    def start(self):
        self._sync()
        self.start_time = time.time()

    def elapsed(self):
        if self.start_time is None:
            return 0
        self._sync()
        return time.time() - self.start_time

    def reset(self):
        self.start_time = None

    def is_started(self):
        return self.start_time is not None
