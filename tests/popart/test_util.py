# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import fnmatch
import functools
import inspect
import os
import re
from typing import Dict

import pytest

import popart


def filter_dict(dict_to_filter, fun):
    sig = inspect.signature(fun)
    filter_keys = [
        param.name for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    ]
    filtered_dict = {
        filter_key: dict_to_filter[filter_key]
        for filter_key in filter_keys if filter_key in dict_to_filter.keys()
    }
    return filtered_dict


def create_test_device(
    numIpus: int = 1,
    tilesPerIPU: int = 0,
    opts: Dict = None,
    pattern: popart.SyncPattern = popart.SyncPattern.Full,
    connectionType: popart.DeviceConnectionType = popart.DeviceConnectionType.
    OnDemand,
    selectionCriterion: popart.DeviceSelectionCriterion = popart.
    DeviceSelectionCriterion.Random):
    testDeviceType = os.environ.get("TEST_TARGET")

    if opts:
        if tilesPerIPU != 0 and "tilesPerIPU" in opts.keys():
            raise RuntimeError("Cannot set tilesPerIPU in 2 ways")

    # NOTE: This function isn't symmetric with willow/include/popart/testdevice.hpp because it doesn't
    # pass on the number of tiles for simulated devices (perhaps it should).
    if tilesPerIPU == 0 and (testDeviceType == "IpuModel"
                             or testDeviceType == "Sim"):
        # We need a number of tiles for these device types.
        tilesPerIPU = 4

    if opts is None:
        opts = {}

    opts["numIPUs"] = numIpus
    opts["tilesPerIPU"] = tilesPerIPU

    if testDeviceType is None:
        testDeviceType = "Cpu"
    if testDeviceType == "Cpu":
        device = popart.DeviceManager().createCpuDevice()
    elif testDeviceType == "Sim":
        device = popart.DeviceManager().createSimDevice()
    elif testDeviceType == "Hw":
        dm = popart.DeviceManager()
        # Keep trying to attach for 15 minutes before aborting
        dm.setOnDemandAttachTimeout(900)
        device = popart.DeviceManager().acquireAvailableDevice(
            numIpus=numIpus,
            tilesPerIpu=tilesPerIPU,
            pattern=pattern,
            connectionType=connectionType,
            selectionCriterion=selectionCriterion)

        if device.tilesPerIpu != 1216:
            pytest.skip("T25924: intermittent hangs on IPUs with >1216 tiles")

    elif testDeviceType == "IpuModel":
        device = popart.DeviceManager().createIpuModelDevice(opts)
    else:
        raise RuntimeError(f"Unknown device type {testDeviceType}")

    if device is None:
        return pytest.fail(
            f"Tried to acquire device {testDeviceType} : {numIpus} IPUs, {tilesPerIPU} tiles,"
            f" {pattern} pattern, {connectionType} connection, but none were availaible"
        )
    return device


def get_compute_sets_from_report(report):

    lines = report.split('\n')
    cs = [x for x in lines if re.search(r' OnTileExecute:', x)]
    cs = [":".join(x.split(":")[1:]) for x in cs]
    cs = [x.strip() for x in cs]
    return set(cs)


def check_whitelist_entries_in_compute_sets(cs_list, whitelist):

    result = True
    fail_list = []
    wl = [x + '*' for x in whitelist]
    for cs in cs_list:
        if len([x for x in wl if fnmatch.fnmatch(cs, x)]) == 0:
            fail_list += [cs]
            result = False
    if not result:
        print("Failed to match " + str(fail_list))
    return result


def check_compute_sets_in_whitelist_entries(cs_list, whitelist):

    result = True
    fail_list = []
    wl = [x + '*' for x in whitelist]
    for x in wl:
        if len([cs for cs in cs_list if fnmatch.fnmatch(cs, x)]) == 0:
            fail_list += [x]
            result = False
    if not result:
        print("Failed to match " + str(fail_list))
    return result


def check_all_compute_sets_and_list(cs_list, whitelist):

    return (check_whitelist_entries_in_compute_sets(cs_list, whitelist)
            and check_compute_sets_in_whitelist_entries(cs_list, whitelist))


def get_compute_set_regex_count(regex, cs_list):

    return len([cs for cs in cs_list if re.search(regex, cs)])


def ipu_available(numIPUs=1):
    return len(popart.DeviceManager().enumerateDevices(numIpus=numIPUs)) > 0


def requires_ipu(func):
    @functools.wraps(func)
    def decorated_func(*args, **kwargs):
        curr_test_target = os.environ.get("TEST_TARGET")
        if "numIPUs" in kwargs:
            numIPUs = kwargs["numIPUs"]
        else:
            numIPUs = 1
        if not ipu_available(numIPUs):
            return pytest.fail(f"Test requires {numIPUs} IPUs")
        # If not already set, make the test target Hw
        if not curr_test_target:
            os.environ["TEST_TARGET"] = "Hw"

        return func(*args, **kwargs)

    return decorated_func


def requires_ipu_model(func):
    @functools.wraps(func)
    def decorated_func(*args, **kwargs):
        curr_test_target = os.environ.get("TEST_TARGET")

        # If not already set, make the test target IpuModel
        if not curr_test_target:
            os.environ["TEST_TARGET"] = "IpuModel"

        return func(*args, **kwargs)

    return decorated_func
