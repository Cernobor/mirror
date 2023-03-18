import collections
import depthai as dai


class HostSync:
    def __init__(self, device: dai.Device, *queues: str, print_add=False):
        self.device = device
        self._data = {}
        self._seqs = collections.deque()
        self._queues = [(q, device.getOutputQueue(q)) for q in queues]
        self._queue_names = queues
        self._n_queues = len(queues)
        self._last_sync_seq = -float('inf')
        self._last_rec_seq = -float('inf')
        self._print_add = print_add

    def get(self):
        for n, q in self._queues:
            if q.has():
                self.add_msg(n, q.get())
        return self._get_last_msg()

    def add_msg(self, name, msg):
        if self._print_add:
            print(name, msg.getSequenceNum(), msg.getTimestamp())
        seq = msg.getSequenceNum()
        if self._last_rec_seq < seq:
            self._last_rec_seq = seq
        if seq not in self._data:
            self._data[seq] = {name: msg}
            self._seqs.append(seq)
        else:
            self._data[seq][name] = msg
    
    def get_lag(self) -> int:
        return self._last_rec_seq - self._last_sync_seq

    def _get_last_msg(self):
        seq = None
        res = None
        drop = None
        found = False
        for i, s in enumerate(self._seqs):
            d = self._data.get(s, None)
            if d is not None and len(d) == self._n_queues:
                seq = s
                res = d
                found = True
            elif found:
                drop = i
                break
        if drop is not None:
            for _ in range(drop):
                self._data.pop(self._seqs.popleft())
        if seq is not None:
            self._last_sync_seq = seq
        return res, seq
