# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
from glob import glob
import itertools
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterator, List

from clang import cindex

# Code adapted from:
# https://github.com/pybind/pybind11_mkdoc/blob/93214d7882cafb92bd9bca0bd5c7ab5aae190286/pybind11_mkdoc/mkdoc_lib.py#L276-L289
# licensed MIT License (MIT):
# https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-on-github/licensing-a-repository
# Licence: https://opensource.org/licenses/MIT

# cython.util.find_library does not find `libclang` for all clang
# versions and distributions. LLVM switched to a monolithical setup
# that includes everything under /usr/lib/llvm{version_number}/
# We therefore glob for the library and select the highest version
if 'LIBCLANG_PATH' in os.environ:
    library_file = os.environ['LIBCLANG_PATH']
    print(f"Using {library_file} libclang file")
    cindex.Config.set_library_file(library_file)
else:
    library_file_dirs = glob("/usr/lib/llvm-*/lib/libclang.so.1")
    if len(library_file_dirs) > 0:
        library_file = sorted(library_file_dirs, reverse=True)[0]
        print(f"Using {library_file} libclang file")
        cindex.Config.set_library_file(library_file)
    else:
        raise FileNotFoundError(
            "Failed to find libclang.so shared object file! ")

# End 3rd party code.
"""
Example output:

        {
            "name": "AccumulateOp",
            "constructors": [
                {
                    "args": [
                        {
                            "arg_name": "type",
                            "arg_type": "popart::AccumulationType",
                            "const": false,
                            "ref": false,
                            "has_default": false,
                            "default": ""
                        },
                        {
                            "arg_name": "factor",
                            "arg_type": "popart::OptimizerValue",
                            "const": false,
                            "ref": false,
                            "has_default": false,
                            "default": ""
                        },
                        {
                            "arg_name": "",
                            "arg_type": "Op::Settings",
                            "const": true,
                            "ref": true,
                            "has_default": false,
                            "default": ""
                        }
                    ]
                }
            ],
            "display_name": "AccumulateOp(popart::AccumulationType, popart::OptimizerValue, const Op::Settings &)",
            "hash": 772877778
        },
"""

exceptions = [
    "MultiConvOptions", "FmodArg0GradOp", "LeakyReluOpBaseAttributes",
    "ShapeOrLikeOp"
]


class DepthCursor(cindex.Cursor):
    """Cindex cursor with concept of depth to 
    prevent over-recursion when using walk_preorder. 
    E.g. when processing a class if it has std::string as an argument, 
    it will call walk_preorder on std::string to inspect it, and 
    in turn on std::string's methods. We only want to recurse to a
    depth of 3 (popart->Class->method)

    Args:
        cindex (Cursor): Original cursor
        depth (int): Current depth
    """

    def __init__(self, cursor: cindex.Cursor, depth: int) -> None:
        self.depth = depth
        self.cursor = cursor
        super().__init__()

    def walk_preorder_depth(self, max_depth: int = 0) -> Iterator:
        self.depth += 1
        yield self.cursor

        # Only recurse if we are above the max depth
        if self.depth <= max_depth:
            for child in self.cursor.get_children():
                for descendant in DepthCursor(
                        child, self.depth).walk_preorder_depth(max_depth):
                    yield descendant


def find_constructors(node: cindex.Cursor, base: cindex.Cursor) -> Dict:
    """Find info on all the constructors for the given node.

    Args:
        node (cindex.Cursor): The op constructor node
        base (cindex.Cursor): The base class.
    Returns:
        Dict: A dict containing all the required info about the constructor.
            See function return for what it contains.
    """
    args = []

    for c in node.get_arguments():
        ref = False
        const = False
        name = c.spelling
        type = c.type.spelling
        fulltype = type
        has_default = False
        if re.match("const", type):
            type = re.sub("&", "", type)
            ref = True
        if re.match("const", type):
            type = re.sub("const ", "", type)
            const = True
        tokens = []
        for token in c.get_tokens():
            if token.spelling == "=":
                has_default = True
        default = ""
        if has_default:
            tokens += [t.spelling for t in list(c.get_tokens())]
            default = " ".join(tokens)
            default = default.split("=")[-1]

        args.append({
            "arg_name":
            name.strip(),
            "alt_arg_name":
            type.strip().lower().replace('popart::', '').replace('::', ''),
            "arg_type":
            type.strip(),
            "fulltype":
            fulltype,
            "const":
            const,
            "ref":
            ref,
            "has_default":
            has_default,
            "default":
            default.replace(" ", ""),
        })
    # Add the full string of args, to avoid having to re-create later.
    full_args = ""
    for i, arg in enumerate(args):
        full_args += f"{'const ' if arg['const'] else ''}{arg['arg_type']}{' &' if  arg['ref'] else ''}"
        if i + 1 != len(args):
            full_args += ", "

    # Get the relative include path
    file_ = node.location.file.name

    file_ = Path(file_).resolve().relative_to(Path(__file__).resolve().parents[1])

    return {
        "name":
        node.spelling,
        "constructors": [{
            "args": args,
            "full_args": full_args,
        }],
        "display_name":
        node.displayname,
        "hash":
        node.hash,
        "file":
        str(file_),
        "base_class":
        base.spelling.replace("class ", "")
        if hasattr(base, "spelling") else "popart::Op"
    }


def derives_from_op(node: cindex.Cursor) -> bool:
    pass


def is_template(node: cindex.Cursor) -> bool:
    return bool(re.match(r"[\w]+<[\w\s,]+>", node.spelling))


def process_op_header(path: Path, ops_path: str) -> List[Dict]:
    """Go through the file's definitions and process each child node in turn. 

    Args:
        path (Path): Path of the hpp file
        ops_path (str): Path where the ops.hpp files are stored.

    Returns:
        List[Dict]: List of dicts with info on the ops
    """
    ops = {}
    print(f"Processing {path}")
    idx = cindex.Index.create()
    tu = idx.parse(str(path),
                   args=['-xc++', '--std=c++11'],
                   options=cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES
                   | cindex.TranslationUnit.PARSE_INCOMPLETE)
    base_class = None
    has_pure_virtual = False
    for node in DepthCursor(cursor=tu.cursor,
                            depth=0).walk_preorder_depth(max_depth=3):
        # We only want definitions from the ops folder. Base classes might be
        # defined in other .../op/*.hpp files.
        if node.location.file:
            if not os.path.commonprefix(
                [ops_path, str(node.location.file)]).endswith("/op"):
                continue
        else:
            continue

        if node.spelling in exceptions:
            continue
        if hasattr(
                node.semantic_parent, "kind"
        ) and node.semantic_parent.kind == cindex.CursorKind.STRUCT_DECL:
            # Skip struct constructors
            continue
        if node.kind == cindex.CursorKind.CLASS_DECL:
            for c in node.get_children():
                if hasattr(c, "is_pure_virtual_method"
                           ) and c.is_pure_virtual_method():
                    has_pure_virtual = True
                    continue
        # Due to the depth first traversal, we will come to the base specifier first
        # so we can save the base node that our op derives from here.
        if node.kind == cindex.CursorKind.CXX_BASE_SPECIFIER:
            base_class = node
            continue
        # We only want public constructors.
        if not ((node.kind == cindex.CursorKind.CONSTRUCTOR) and
                (node.access_specifier == cindex.AccessSpecifier.PUBLIC)):
            continue

        # Don't want template classes, only specific derived classes.
        if is_template(node):
            continue

        d = find_constructors(node, base_class)
        if node.spelling in ops.keys():
            ops[node.spelling]["constructors"].append(d["constructors"][0])
            continue

        if d and not has_pure_virtual:
            ops[node.spelling] = d

    return [o for o in ops.values()]


def parse_ops(ops_path: str, out_path: str, jobs: int = 1) -> None:
    """Run over all the files in the given dir and process with process_file,
    one thread per file.

    Args:
        ops_path (str): Path where the <op>.hpp files are stored
        out_path (str): File to save the result as
        jobs (int, optional): Number of threads. Defaults to 1.
    """
    pool = Pool(jobs)
    ops = []
    files = list(Path(ops_path).glob('*.hpp'))
    assert len(files) > 0, "No files found. Please check your opdir argument."

    data = pool.starmap(process_op_header,
                        [(file, ops_path) for file in files])

    pool.close()
    pool.join()

    ops = list(itertools.chain.from_iterable(data))

    assert len(
        ops) > 0, "No ops were added to the list. Please check your arguments."

    print(
        f"Processed {len(ops)} ops, from { ops[0]['name']} to {ops[-1]['name']}"
    )

    my_json = {"Ops": ops}
    with open(out_path, "w") as outfile:
        json.dump(my_json, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Tool to get popART op info in JSON format", add_help=False)
    dir_path = Path(__file__).resolve().parents[1]
    parser.add_argument("-j",
                        "--jobs",
                        type=int,
                        help="Number of threads to use",
                        default=1)
    parser.add_argument("-d",
                        "--opdir",
                        type=str,
                        help="Directory to find <op>.hpp files",
                        default=os.path.join(dir_path, "willow", "include",
                                             "popart", "op"))
    parser.add_argument("-o",
                        "--outfile",
                        type=str,
                        help="File to save output into.",
                        default=os.path.join(dir_path, "op_metadata.json"))

    args = parser.parse_args()
    print(
        f"Processing ops in {args.opdir}\n to {args.outfile}\n with { args.jobs} jobs"
    )

    parse_ops(args.opdir, args.outfile, args.jobs)

    with open(args.outfile, 'r') as myfile:
        assert len(myfile.readlines()) > 1, \
            "The output file is too small, please check the script has run correctly."
