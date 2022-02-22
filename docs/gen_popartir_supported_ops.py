#!/usr/bin/python
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import sys
import importlib
import inspect

if len(sys.argv) < 2:
    sys.exit("gen_popartir_supported_ops.py <output file>")


def get_docstring_summary(obj):
    """
    Get the summary of an object's docstring which is separated from the
    rest of the docstring by a blank line. See PEP 257.
    """
    if obj.__doc__ is None:
        return ""
    else:
        lines = obj.__doc__.splitlines()
        seen_text = False
        seen_blank_line = False
        summary = ""
        for line in lines:
            stripped_line = line.strip()
            if stripped_line == "":
                if seen_text:
                    # Blank line! We have a summary.
                    break
            else:
                summary += stripped_line + " "
                seen_text = True
        return summary


with open(sys.argv[1], "w") as f:

    for package_name in [
            'popart.ir.ops', 'popart.ir.ops.collectives',
            'popart.ir.ops.var_updates'
    ]:
        name = package_name.replace(".", "_")
        print(f".. list-table:: Available operations in ``{package_name}``",
              file=f)
        print(f"   :header-rows: 1", file=f)
        print(f"   :width: 100%", file=f)
        print(f"   :widths: 45, 55", file=f)
        print(f"   :name: {name}_available_ops", file=f)
        print(f"   :class: longtable", file=f)
        print(f"", file=f)
        print(f"   * - Operation", file=f)
        print(f"     - Description", file=f)
        print(f"   ", file=f)
        package = importlib.import_module(package_name)
        for name, obj in inspect.getmembers(package, inspect.isfunction):
            print(
                f"   * - :py:func:`{obj.__name__}<popart-python-api:{package_name}.{obj.__name__}>`",
                file=f)
            print(f"     - {get_docstring_summary(obj)}", file=f)
            print(f"   ", file=f)

        print(f"   ", file=f)
        print(f"   ", file=f)
