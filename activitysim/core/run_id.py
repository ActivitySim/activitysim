import struct
import time


class RunId(str):
    def __new__(cls, x=None):
        if x is None:
            return cls(
                hex(struct.unpack("<Q", struct.pack("<d", time.time()))[0])[-6:].lower()
            )
        return super().__new__(cls, x)
