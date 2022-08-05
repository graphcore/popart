# Copyright (c) 2018 Graphcore Ltd. All rights reserved.
import fnmatch
import functools
import inspect
import os
import re
from typing import Dict, List, Union
from typing_extensions import Literal

import popart
import pytest

IPU_MODEL_TYPES: List[str] = ["IpuModel", "IpuModel21", "IpuModel2"]
SIM_TYPES: List[str] = ["Sim21", "Sim2", "Sim"]
HW_AND_MODEL_TYPES: List[str] = ["Hw"] + IPU_MODEL_TYPES + SIM_TYPES
ALL_TYPES: List[str] = HW_AND_MODEL_TYPES + ["Cpu"]


def filter_dict(dict_to_filter, fun):
    sig = inspect.signature(fun)
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    ]
    filtered_dict = {
        filter_key: dict_to_filter[filter_key]
        for filter_key in filter_keys
        if filter_key in dict_to_filter.keys()
    }
    return filtered_dict


USE_ALL_TILES = 0


def device_to_ipu_name(device_type: str) -> Literal["ipu2", "ipu21"]:
    """Get the ipu name from the device type.

    Args:
        device_type (str): The device type: Sim21, IpuModel

    Raises:
        ValueError: If an incorrect device type is passed.

    Returns:
        str: The ipu name, one of "ipu21" or "ipu2"
    """
    if device_type in ["Sim2", "IpuModel2"]:
        return "ipu2"
    elif device_type in ["Sim21", "IpuModel21"]:
        return "ipu21"
    elif device_type in ["Sim", "IpuModel"]:
        # default for backwards compatability
        return "ipu21"
    else:
        raise ValueError(f"Unknown device type: {device_type}")


class DeviceContext:
    """This is a wrapper around DeviceInfo which forces DeviceInfo to be used
    as a context manager."""

    def __init__(self, device):
        self.device = device

    def __enter__(self):
        return self.device.__enter__()

    def __exit__(self, *args, **kwargs):
        return self.device.__exit__(*args, **kwargs)


def create_test_device(
    numIpus: int = 1,
    opts: Union[Dict, None] = None,
    pattern: popart.SyncPattern = popart.SyncPattern.Full,
    connectionType: popart.DeviceConnectionType = popart.DeviceConnectionType.OnDemand,
    selectionCriterion: popart.DeviceSelectionCriterion = popart.DeviceSelectionCriterion.Random,
    tilesPerIPU: Union[int, None] = None,
) -> DeviceContext:
    """Create a PopART device to be used on a PopART test, given the provided
    arguments.

    Args:
        numIpus (int, optional):
            The number of IPUs to request. Must be a power of 2. Defaults to 1.
        opts (Union[Dict, None], optional):
            A dictionary of strings to pass to the device manager. Defaults to
            None.
        pattern (popart.SyncPattern, optional):
            The syncpattern to use. See the sync patterns section in the
            documentation. Defaults to popart.SyncPattern.Full.
        connectionType (popart.DeviceConnectionType, optional):
            The device connection type to use, see devicemanager.hpp for
            details. Defaults to popart.DeviceConnectionType.OnDemand.
        selectionCriterion (popart.DeviceSelectionCriterion, optional):
            The device selection to use, see devicemanager.hpp for details.
            Defaults to popart.DeviceSelectionCriterion.Random.
        tilesPerIPU (Union[int, None], optional):
            The number of tiles per IPU to use, use a small number e.g. 4 to
            speed up tests. Defaults to None, which will use the maximum
            available.

    Raises:
        RuntimeError: If there is a conflict between the options passed as
            arguments, and the options passed as `opts`
        RuntimeError: If an invalid device type is passed as the `TEST_TARGET`
            env variable.

    Returns:
        DeviceContext: A wrapper around the DeviceInfo object used to run a
            session. This is a wrapper around DeviceInfo which forces DeviceInfo
            to be used as a context manager.
    """
    testDeviceType = os.environ.get("TEST_TARGET")

    if opts:
        if tilesPerIPU is not None and "tilesPerIPU" in opts.keys():
            raise RuntimeError("Cannot set tilesPerIPU in 2 ways")

    # NOTE: This function isn't symmetric with testdevice.hpp because it doesn't
    # pass on the number of tiles for simulated devices (perhaps it should).
    if tilesPerIPU is None and testDeviceType in HW_AND_MODEL_TYPES:
        tilesPerIPU = 4

    if testDeviceType is None:
        testDeviceType = "Cpu"
    if testDeviceType == "Cpu":
        device = popart.DeviceManager().createCpuDevice()
    elif testDeviceType in SIM_TYPES:
        if opts is None:
            opts = {}

        opts["numIPUs"] = numIpus
        opts["tilesPerIPU"] = tilesPerIPU
        opts["ipuVersion"] = device_to_ipu_name(testDeviceType)
        device = popart.DeviceManager().createSimDevice(opts)
    elif testDeviceType == "Hw":
        dm = popart.DeviceManager()
        # Keep trying to attach for 15 minutes before aborting
        dm.setOnDemandAttachTimeout(900)
        device = popart.DeviceManager().acquireAvailableDevice(
            numIpus=numIpus,
            tilesPerIpu=tilesPerIPU,
            pattern=pattern,
            connectionType=connectionType,
            selectionCriterion=selectionCriterion,
        )

        assert device is not None

    elif testDeviceType in IPU_MODEL_TYPES:
        if opts is None:
            opts = {}
        opts["numIPUs"] = numIpus
        opts["tilesPerIPU"] = tilesPerIPU
        opts["ipuVersion"] = device_to_ipu_name(testDeviceType)

        device = popart.DeviceManager().createIpuModelDevice(opts)
    else:
        raise RuntimeError(f"Unknown device type {testDeviceType}")

    if device is None:
        return pytest.fail(
            f"Tried to acquire device {testDeviceType} : {numIpus} IPUs, {tilesPerIPU} tiles,"
            f" {pattern} pattern, {connectionType} connection, but none were availaible"
        )
    return DeviceContext(device)


def get_compute_sets_from_report(report):

    lines = report.split("\n")
    cs = [x for x in lines if re.search(r" OnTileExecute:", x)]
    cs = [":".join(x.split(":")[1:]) for x in cs]
    cs = [x.strip() for x in cs]
    return set(cs)


def check_whitelist_entries_in_compute_sets(cs_list, whitelist):

    result = True
    fail_list = []
    wl = [x + "*" for x in whitelist]
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
    wl = [x + "*" for x in whitelist]
    for x in wl:
        if len([cs for cs in cs_list if fnmatch.fnmatch(cs, x)]) == 0:
            fail_list += [x]
            result = False
    if not result:
        print("Failed to match " + str(fail_list))
    return result


def check_all_compute_sets_and_list(cs_list, whitelist):

    return check_whitelist_entries_in_compute_sets(
        cs_list, whitelist
    ) and check_compute_sets_in_whitelist_entries(cs_list, whitelist)


def get_compute_set_regex_count(regex, cs_list):

    return len([cs for cs in cs_list if re.search(regex, cs)])


def ipu_available(numIPUs=1):
    return len(popart.DeviceManager().enumerateDevices(numIpus=numIPUs)) > 0


def _run_test_on_target(func, target, args, kwargs):
    curr_test_target = os.environ.get("TEST_TARGET")

    # If not already set, make the test target `target`
    if not curr_test_target:
        os.environ["TEST_TARGET"] = target

    def reset_test_target():
        if curr_test_target is None:
            os.environ.pop("TEST_TARGET")
        else:
            os.environ["TEST_TARGET"] = curr_test_target

    try:
        result = func(*args, **kwargs)
        reset_test_target()
        return result
    except:
        reset_test_target()
        raise


def requires_ipu(func):
    @functools.wraps(func)
    def decorated_func(*args, **kwargs):
        if "numIPUs" in kwargs:
            numIPUs = kwargs["numIPUs"]
        else:
            numIPUs = 1
        if not ipu_available(numIPUs):
            return pytest.fail(f"Test requires {numIPUs} IPUs")

        _run_test_on_target(func, "Hw", args, kwargs)

    return decorated_func


def requires_ipu_model(func):
    @functools.wraps(func)
    def decorated_func(*args, **kwargs):
        _run_test_on_target(func, "IpuModel", args, kwargs)

    return decorated_func


def set_autoreport_options(
    options,
    directory,
    output_graph_profile=True,
    output_execution_profile=False,
    max_execution_reports=1000,
):
    """Sets autoReport engine options in the IPUConfig.

    Set outputExecutionProfile to True to allow execution reports to be
    generated.

    If execution reports are enabled, max_execution_reports controls the
    maximum number of executions included in a report.
    """
    engineOptions = {
        "autoReport.directory": str(directory),
        "autoReport.outputGraphProfile": str(output_graph_profile).lower(),
        "autoReport.outputExecutionProfile": str(output_execution_profile).lower(),
        "autoReport.executionProfileProgramRunCount": str(max_execution_reports),
    }
    for opt in engineOptions:
        options.engineOptions[opt] = engineOptions[opt]
