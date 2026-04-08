# -*- coding: utf-8 -*-
"""
_component_helpers.py – Shared utility helpers for pyfofem component modules.

This module collects small, cross-cutting helper functions that are used by
two or more component sub-modules (e.g. ``tree_flame_calcs``,
``consumption_calcs``, ``mortality_calcs``).  Keeping them here avoids
circular imports and prevents any single domain module from being the
accidental "owner" of generic utilities.

Functions:
    _is_scalar     – Test whether an input is a scalar (not an array).
    _maybe_scalar  – Collapse a length-1 array to a Python float when the
                     original input was scalar.
    _to_str_arr    – Convert a categorical parameter (str, int, or ndarray)
                     to a 1-D numpy string array via an integer→string LUT.
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import numpy as np
from typing import Dict, Union


def _is_scalar(x) -> bool:
    """Return ``True`` if *x* is a Python scalar or 0-d array.

    A "scalar" in pyfofem's convention means a single numeric value, not a
    sequence or array.  Both plain Python numbers and 0-dimensional
    ``np.ndarray`` objects qualify.

    :param x: Any value to test.
    :returns: ``True`` when *x* is scalar; ``False`` otherwise.
    """
    if isinstance(x, np.ndarray):
        return x.ndim == 0
    return np.ndim(x) == 0


def _maybe_scalar(arr, scalar_input: bool):
    """Return ``float(arr[0])`` when *scalar_input* is ``True``.

    Used as the final return step in vectorised functions to preserve the
    scalar-in / scalar-out contract when the caller passed a single value.
    ``None`` is passed through unchanged regardless of *scalar_input*.

    :param arr: A numpy array (or ``None``).
    :param scalar_input: ``True`` when the original caller inputs were all
        scalar; ``False`` when they were array-like.
    :returns: ``float(arr[0])`` if *scalar_input* is ``True`` and *arr* is
        not ``None``; otherwise *arr* unchanged.
    """
    if arr is None:
        return None
    return float(arr[0]) if scalar_input else arr


def _to_str_arr(
    val: Union[str, int, np.ndarray],
    lut: Dict[int, str],
) -> np.ndarray:
    """Convert a categorical parameter to a 1-D numpy string array.

    Accepts a plain string, an integer code (looked up in *lut*), or a
    numpy array of strings or integer codes.  Scalar inputs return a
    length-1 array; array inputs are returned as a 1-D ``object`` array of
    strings.

    :param val: Input value — ``str``, ``int``, or ``np.ndarray``.
    :param lut: Integer → string lookup table (e.g. ``REGION_CODES``).
    :returns: 1-D ``np.ndarray`` of ``dtype=object`` (strings).
    :raises KeyError: If an integer code is not present in *lut*.
    """
    if isinstance(val, np.ndarray):
        flat = val.ravel()
        out = np.empty(flat.shape, dtype=object)
        for i, v in enumerate(flat):
            if isinstance(v, (int, np.integer)):
                out[i] = lut[int(v)]
            else:
                out[i] = str(v)
        return out
    if isinstance(val, (int, np.integer)):
        return np.array([lut[int(val)]], dtype=object)
    return np.array([str(val)], dtype=object)

