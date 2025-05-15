from __future__ import annotations

import warnings

import psutil

from .progression import reset_progress_step
from .wrapping import workstep


@workstep(updates_context=True)
def chunk_sizing(
    chunk_size=None,
    chunk_size_pct_of_available=None,
    chunk_size_pct_of_total=None,
    chunk_size_minimum_gb=0,
):
    reset_progress_step(description="Figuring chunk size")

    # by default, if neither pct_of_available or pct_of_total is given,
    # and chunk_size is not set explicitly, then use 85% of available RAM
    if chunk_size_pct_of_available is None and chunk_size_pct_of_total is None:
        chunk_size_pct_of_available = 0.85

    vm = psutil.virtual_memory()
    available_ram = vm.available
    total_ram = vm.total

    # if chunk size is set explicitly, use it without regard to other settings
    if chunk_size is None:
        if chunk_size_pct_of_available is not None:
            if chunk_size_pct_of_available > 1:
                chunk_size_pct_of_available /= 100
            chunk_size = int(available_ram * chunk_size_pct_of_available)
        elif chunk_size_pct_of_total is not None:
            if chunk_size_pct_of_total > 1:
                chunk_size_pct_of_total /= 100
            chunk_size = int(total_ram * chunk_size_pct_of_total)

        min_chunk_size = int(chunk_size_minimum_gb * 1e9)
        if chunk_size < min_chunk_size:
            chunk_size = min_chunk_size

    if chunk_size > total_ram:
        warnings.warn(
            f"chunk size of {chunk_size/ 2**30:.2f}GB exceeds "
            f"total RAM of {total_ram/ 2**30:.2f}",
            stacklevel=2,
        )

    out = dict(chunk_size=chunk_size)
    return out
