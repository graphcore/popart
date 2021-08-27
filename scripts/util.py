# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import operator
import subprocess
from itertools import accumulate
from pathlib import Path


def get_project_source_dir():
    return Path(__file__).resolve().parents[1].resolve()


def bor(*args: int):
    """Bitwise or.

    Example:
        bor(0x01, 0x10) == 0x01 | 0x10

    Returns:
        int: Inputs.
    """
    return list(accumulate(args, operator.or_))[-1]


def bash(args, cwd='.', log=True, ignore_return_code=False) -> str:
    """
    Run a bash subprocess.
    This is used primarily for executing cmake, ninja, the test script and gcovr.

    Arguments
    ---
        args: Sequence or str of args, the same as you would pass to subprocess.Popen
        cwd: Path-like to the directory in which to execute the bash subprocess
        log: Bool, if True the stdout of the process is sent to stdout and the 
            result is returned as a string. Otherwise, discard the output.
        ignore_return_code: Bool, if false an erroneous return code is discarded, 
            otherwise CalledProcessError is raised.

    Returns
    ---
        str: the collected stdout of the subprocess.
    """
    process = subprocess.Popen(args,
                               cwd=cwd,
                               stdout=subprocess.PIPE,
                               universal_newlines=True)
    result = ''
    for stdout_line in iter(process.stdout.readline, ""):
        result += stdout_line
        if log:
            print(stdout_line, end='')
    process.stdout.close()
    return_code = process.wait()
    if not ignore_return_code and return_code != 0:
        raise subprocess.CalledProcessError(return_code, args)
    else:
        return result
