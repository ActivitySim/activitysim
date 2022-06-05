import numba as nb
import numpy as np
import pandas as pd


@nb.njit
def _fast_map(fm, target, dtype=np.int32):
    out = np.zeros(len(target), dtype=dtype)
    for n in range(target.size):
        out[n] = fm[target[n]]
    return out


class FastMapping:
    def __init__(self, source, to_range=np.int64):
        if isinstance(source, pd.Series):
            m = nb.typed.Dict.empty(
                key_type=nb.from_dtype(source.index.dtype),
                value_type=nb.from_dtype(source.dtype),
            )
            for k, v in source.items():
                m[k] = v
            self._in_dtype = source.index.dtype
            self._out_dtype = source.dtype
            self._mapper = m
        elif to_range:
            m = nb.typed.Dict.empty(
                key_type=nb.from_dtype(source.dtype),
                value_type=nb.from_dtype(to_range),
            )
            for v, k in enumerate(source):
                m[k] = v
            self._in_dtype = source.dtype
            self._out_dtype = to_range
            self._mapper = m
        else:
            raise ValueError("invalid input")

    def __len__(self):
        return len(self._mapper)

    def __contains__(self, item):
        return item in self._mapper

    def __getitem__(self, item):
        return self._mapper[item]

    def apply_to(self, target):
        if isinstance(target, pd.Series):
            return pd.Series(
                _fast_map(
                    self._mapper,
                    target.astype(self._in_dtype).to_numpy(),
                    dtype=self._out_dtype,
                ),
                index=target.index,
            )
        return _fast_map(
            self._mapper,
            np.asarray(target, dtype=self._in_dtype),
            dtype=self._out_dtype,
        )
