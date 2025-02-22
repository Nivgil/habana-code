import torch
import utils
import time
from collections import deque


class ComputeTimeout(Exception):
    pass


class DeviceTimer(object):
    def _get_hpu_module(self):
        if self.habana_module is None:
            import habana_frameworks.torch as ht
            self.habana_module = ht

    def __init__(self, use_hpu=False, debug=False, event_sync_delay=9):
        self.start_time = None
        self.debug = debug
        self.events = deque(maxlen=event_sync_delay+1)  # sync on the previous event according to parameter
        self.habana_module = None
        self.use_hpu = use_hpu
        self.enable_drop_compute = False
        self.dropped = False
        self.drop_threshold = 0
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

    def check_drop_compute_throw(self, name=""):
        if not self.is_started():
            return False

        current_time = self.elapsed()
        #if utils.is_main_process():
        #    print(f"current_time: {current_time}, global_drop_threshold: {self.drop_threshold}, global_enable_drop_compute {self.enable_drop_compute}")
        if self.enable_drop_compute and (current_time > self.drop_threshold):
            self.dropped = True
            if self.debug:
                print(f"reached timeout with module: {name}. current_time: {current_time}, global_drop_threshold: {self.drop_threshold}")
            raise ComputeTimeout("")


    def reset(self):
        self.dropped = False
        self.start_time = None

    def is_started(self):
        return self.start_time is not None
