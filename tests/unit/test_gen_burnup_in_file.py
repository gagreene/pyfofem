#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_gen_burnup_in_file.py - Phase 7 item D: focused content and
validation coverage for
:func:`~pyfofem.components.burnup_calcs.gen_burnup_in_file`, beyond the
fire-environment-bounds clipping already characterized in
``test_fire_environment_bounds.py`` (path 3). Every assertion here parses
and checks real field content - never a snapshot of the opaque text blob.

Every test below uses the ``tmp_path`` fixture NAME for compatibility
with pytest's own ``tmp_path``-based idioms (``tmp_path / 'out.brn'``,
etc.), but this module OVERRIDES it (see :func:`tmp_path` below) with a
repository-local, collision-safe directory under
``tests/cpp_parity_live/_scratch.py``'s scratch root, never the system/
user temporary directory pytest's own built-in ``tmp_path`` resolves to
- required by the Phase 7 correction pass's explicit filesystem
boundary (2026-09-06). The override is scoped to this module only (a
module-level fixture shadows the builtin only for tests collected from
this file), so other test modules' own use of the real ``tmp_path``
fixture is unaffected.

Function order: private helpers first, then public test functions, each
group alphabetized, per AGENTS.md.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from pyfofem.components import burnup_calcs as bc
from tests.cpp_parity_live._scratch import scratch_tempdir

#: Exact field order gen_burnup_in_file() writes, taken directly from its
#: own ``params`` list construction.
_EXPECTED_FIELD_ORDER = [
    'MAX_TIMES', 'INTENSITY', 'IG_TIME', 'WINDSPEED', 'DEPTH',
    'AMBIENT_TEMP', 'R0', 'DR', 'TIMESTEP',
    'SURat_Lit', 'SURat_DW1', 'SURat_DW10', 'SURat_DW100',
    'SURat_DWk_3_6', 'SURat_DWk_6_9', 'SURat_DWk_9_20', 'SURat_DWk_20',
]


def _parse_lines(content: str):
    """Parse a generated ``#NAME value`` .brn text blob into an ordered
    list of ``(name, raw_value_str)`` pairs - preserving both order and
    the exact serialized string, unlike a dict-based parse.

    :param content: Raw file text as produced by
        :func:`~pyfofem.components.burnup_calcs.gen_burnup_in_file`.
    :return: Ordered list of ``(name, raw_value_str)`` tuples.
    """
    pairs = []
    for line in content.splitlines():
        match = re.match(r'#(\S+) (.+)$', line)
        assert match, f'unparseable .brn line: {line!r}'
        pairs.append((match.group(1), match.group(2)))
    return pairs


def test_content_has_no_trailing_newline_and_uses_hash_prefixed_lines(tmp_path):
    """Serialization-format contract: lines are ``\\n``-joined with no
    trailing newline (``'\\n'.join(lines)``, not ``+= '\\n'``), and every
    line begins with ``#``."""
    out_path = tmp_path / 'out.brn'
    bc.gen_burnup_in_file(out_brn_path=str(out_path))
    content = out_path.read_text()
    assert not content.endswith('\n')
    lines = content.splitlines()
    assert len(lines) == len(_EXPECTED_FIELD_ORDER)
    assert all(line.startswith('#') for line in lines)


def test_deterministic_repeated_generation_produces_byte_identical_output(tmp_path):
    """Calling the function twice with identical arguments (default or
    otherwise) must produce byte-identical file content - no timestamp,
    randomness, or hidden state."""
    out_path_a = tmp_path / 'a.brn'
    out_path_b = tmp_path / 'b.brn'
    bc.gen_burnup_in_file(out_brn_path=str(out_path_a), intensity=123.4, max_times=777)
    bc.gen_burnup_in_file(out_brn_path=str(out_path_b), intensity=123.4, max_times=777)
    assert out_path_a.read_bytes() == out_path_b.read_bytes()

    # Overwriting the same path twice is also byte-identical and leaves no
    # residue from the first write.
    bc.gen_burnup_in_file(out_brn_path=str(out_path_a), intensity=999.0, max_times=42)
    first_content = out_path_a.read_bytes()
    bc.gen_burnup_in_file(out_brn_path=str(out_path_a), intensity=999.0, max_times=42)
    assert out_path_a.read_bytes() == first_content


def test_max_times_clipped_on_both_sides(tmp_path):
    """``max_times`` (not one of the ``_FIRE_BOUNDS``-linked fields
    covered in ``test_fire_environment_bounds.py``) is independently
    clipped to ``[1, 100000]``, both sides, with no rejection path."""
    below_path = tmp_path / 'below.brn'
    bc.gen_burnup_in_file(out_brn_path=str(below_path), max_times=0)
    pairs = dict(_parse_lines(below_path.read_text()))
    assert pairs['MAX_TIMES'] == '1'

    above_path = tmp_path / 'above.brn'
    bc.gen_burnup_in_file(out_brn_path=str(above_path), max_times=500000)
    pairs2 = dict(_parse_lines(above_path.read_text()))
    assert pairs2['MAX_TIMES'] == '100000'


def test_missing_directory_raises_and_leaves_no_partial_file(tmp_path):
    """File-write failure behavior: a parent directory that does not
    exist causes ``open(path, 'w')`` itself to fail before any bytes are
    written - no partial/stale output file is created."""
    out_path = tmp_path / 'nonexistent_subdir' / 'out.brn'
    with pytest.raises(OSError):
        bc.gen_burnup_in_file(out_brn_path=str(out_path))
    assert not out_path.exists()
    assert not out_path.parent.exists()


def test_missing_out_brn_path_raises_bare_exception():
    """The documented ``out_brn_path is None`` guard raises a bare
    ``Exception`` (not a domain-specific subclass) with the exact
    documented message - characterizing current behavior, not endorsing
    it as good API design."""
    with pytest.raises(Exception, match='No output path specified'):
        bc.gen_burnup_in_file(out_brn_path=None)


def test_nominal_content_field_order_matches_the_documented_params_list(tmp_path):
    """Nominal generated content: every one of the 17 documented fields
    is present, in the exact order the function's own ``params`` list
    constructs them, using entirely default keyword values."""
    out_path = tmp_path / 'nominal.brn'
    bc.gen_burnup_in_file(out_brn_path=str(out_path))
    pairs = _parse_lines(out_path.read_text())
    names = [name for name, _value in pairs]
    assert names == _EXPECTED_FIELD_ORDER


def test_numeric_serialization_preserves_int_vs_float_representation(tmp_path):
    """Units/numeric-serialization contract: integer-default SAV fields
    (``surat_lit=8200`` etc.) serialize WITHOUT a decimal point, while
    float-valued fields (``surat_dwk_3_6=39.4`` etc., and any clipped
    fire-environment float) serialize WITH one - since the function does
    no explicit numeric formatting, this is a direct consequence of
    Python's own ``str()``/f-string conversion of whatever type the
    caller passed in, not a deliberate schema."""
    out_path = tmp_path / 'types.brn'
    bc.gen_burnup_in_file(out_brn_path=str(out_path))
    pairs = dict(_parse_lines(out_path.read_text()))
    # int-default fields: no decimal point.
    for name in ('MAX_TIMES', 'SURat_Lit', 'SURat_DW1', 'SURat_DW10', 'SURat_DW100'):
        assert '.' not in pairs[name], f'{name}={pairs[name]!r} unexpectedly has a decimal point'
    # float-default fields: a decimal point is present.
    for name in ('INTENSITY', 'IG_TIME', 'WINDSPEED', 'DEPTH', 'AMBIENT_TEMP',
                 'R0', 'DR', 'TIMESTEP', 'SURat_DWk_3_6', 'SURat_DWk_6_9',
                 'SURat_DWk_9_20', 'SURat_DWk_20'):
        assert '.' in pairs[name], f'{name}={pairs[name]!r} unexpectedly lacks a decimal point'

    # Passing an int explicitly for a normally-float field preserves that
    # int's own string form (no forced float coercion anywhere in the
    # function) - e.g. r0=2 (int) stays "2", not "2.0".
    out_path2 = tmp_path / 'int_override.brn'
    bc.gen_burnup_in_file(out_brn_path=str(out_path2), r0=2, timestep=10)
    pairs2 = dict(_parse_lines(out_path2.read_text()))
    assert pairs2['R0'] == '2'
    assert pairs2['TIMESTEP'] == '10'


def test_output_path_that_is_an_existing_directory_raises_and_is_unmodified(tmp_path):
    """A second file-write failure mode: ``out_brn_path`` pointing at an
    existing DIRECTORY (not a writable file target) raises ``OSError``
    (``IsADirectoryError`` on this platform's ``open()``), and the
    directory itself is left untouched (still a directory, still
    empty)."""
    out_dir = tmp_path / 'a_directory'
    out_dir.mkdir()
    with pytest.raises(OSError):
        bc.gen_burnup_in_file(out_brn_path=str(out_dir))
    assert out_dir.is_dir()
    assert list(out_dir.iterdir()) == []


def test_surat_values_pass_through_unclipped_and_unvalidated(tmp_path):
    """The 8 ``surat_*`` (SAV) keyword parameters have NO documented
    bound and are written through completely unchanged, including
    physically nonsensical values (e.g. negative) - characterizing that
    ``gen_burnup_in_file()``'s validation is limited to the 5
    fire-environment-bounds fields plus ``max_times``."""
    out_path = tmp_path / 'surat.brn'
    bc.gen_burnup_in_file(
        out_brn_path=str(out_path),
        surat_lit=-1, surat_dw1=0, surat_dw10=999999.5,
    )
    pairs = dict(_parse_lines(out_path.read_text()))
    assert pairs['SURat_Lit'] == '-1'
    assert pairs['SURat_DW1'] == '0'
    assert pairs['SURat_DW10'] == '999999.5'


@pytest.fixture
def tmp_path(request):
    """
    Override pytest's built-in ``tmp_path`` fixture for every test in
    this module: a repository-local, collision-safe directory under
    ``tests/cpp_parity_live/_scratch.py``'s scratch root, never the
    system/user temporary directory, per the Phase 7 correction pass's
    explicit filesystem boundary (2026-09-06). Returns a real
    ``pathlib.Path`` so existing test bodies (``tmp_path / 'x.brn'``,
    ``.read_text()``, ``.is_dir()``, etc.) work completely unchanged.
    Removed on exit regardless of test outcome.

    :param request: The requesting test node (used only to namespace the
        directory name for readability under the shared scratch root).
    :return: Yields a ``pathlib.Path`` to the created directory.
    """
    with scratch_tempdir("gen_burnup_in_file", prefix=request.node.name) as path:
        yield Path(path)
