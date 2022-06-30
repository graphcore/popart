#!/usr/bin/python
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import sys
import importlib
import inspect

if len(sys.argv) < 2:
    sys.exit("gen_popartir_supported_ops.py <output file>")


def get_docstring_summary(obj):
    """
    Get the summary of an object's docstring.

    This is separated from the rest of the docstring by a blank line.
    See PEP 257 for details.
    """
    if obj.__doc__ is None:
        return ""
    else:
        lines = obj.__doc__.splitlines()
        seen_text = False
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


with open(sys.argv[1], "w", encoding="utf-8") as f:

    for package_name in [
            'popxl.ops', 'popxl.ops.collectives', 'popxl.ops.var_updates'
    ]:
        name = package_name.replace(".", "_")
        print(f".. list-table:: Available operations in ``{package_name}``",
              file=f)
        print("   :header-rows: 1", file=f)
        print("   :width: 100%", file=f)
        print("   :widths: 45, 55", file=f)
        print(f"   :name: {name}_available_ops", file=f)
        print("   :class: longtable", file=f)
        print("", file=f)
        print("   * - Operation", file=f)
        print("     - Description", file=f)
        print("   ", file=f)
        package = importlib.import_module(package_name)
        for name, obj in inspect.getmembers(package, inspect.isfunction):
            print(f"   * - :py:func:`~{package_name}.{obj.__name__}`", file=f)
            print(f"     - {get_docstring_summary(obj)}", file=f)
            print("   ", file=f)

        print("   ", file=f)
        print("   ", file=f)
