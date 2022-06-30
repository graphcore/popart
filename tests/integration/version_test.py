# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import json
import os
import popart


def test_version_number():
    ver = popart.versionNumber()
    assert isinstance(ver, tuple)
    assert len(ver) == 3

    ver_string = popart.versionString()
    assert ver_string
    assert isinstance(ver_string, str)

    pkg_hash = popart.packageHash()
    assert pkg_hash
    assert isinstance(pkg_hash, str)

    ver_path = os.path.abspath(
        os.path.join(__file__, '..', '..', '..', 'version.json'))

    with open(ver_path, 'r', encoding="utf-8") as f:
        version_obj = json.load(f)

    assert int(version_obj['major']) == ver[0]
    assert int(version_obj['minor']) == ver[1]
    assert int(version_obj['point']) == ver[2]
