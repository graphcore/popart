# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
import argparse
import json
import os
import re
from multiprocessing import Pool
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterator, List, Text, Tuple

# clang is not a common install. Raise error and point to readme in case of missing package.
try:
    from clang import cindex
except:
    raise ImportError(
        "Failed to import python package `clang`, usually installed with package `libclang`."
        "Please make sure you install all dependencies. See the README.md for details."
    )

from util import bor, get_project_source_dir

# Monkey patch clang.cindex.TranslationUnit flags that are not available by
# default.
# See: https://clang.llvm.org/doxygen/group__CINDEX__TRANSLATION__UNIT.html#gab1e4965c1ebe8e41d71e90203a723fe9
cindex.TranslationUnit.PARSE_NONE = 0x0
cindex.TranslationUnit.PARSE_DETAILED_PREPROCESSING_RECORD = 0x01
cindex.TranslationUnit.PARSE_INCOMPLETE = 0x02
cindex.TranslationUnit.PARSE_PRECOMPILED_PREAMBLE = 0x04
cindex.TranslationUnit.PARSE_CACHE_COMPLETION_RESULTS = 0x08
cindex.TranslationUnit.PARSE_FOR_SERIALIZATION = 0x10
cindex.TranslationUnit.PARSE_CXX_CHAINED_PCH = 0x20
cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES = 0x40
cindex.TranslationUnit.PARSE_INCLUDE_BRIEF_COMMENTS_IN_CODE_COMPLETION = 0x80
cindex.TranslationUnit.PARSE_CREATE_PREAMBLE_ON_FIRST_PARSE = 0x100
cindex.TranslationUnit.PARSE_KEEP_GOING = 0x200
cindex.TranslationUnit.PARSE_SINGLE_FILE_PARSE = 0x400
cindex.TranslationUnit.PARSE_LIMIT_SKIP_FUNCTION_BODIES_TO_PREAMBLE = 0x800
cindex.TranslationUnit.PARSE_INCLUDE_ATTRIBUTED_TYPES = 0x1000
cindex.TranslationUnit.PARSE_VISIT_IMPLICIT_ATTRIBUTES = 0x2000
cindex.TranslationUnit.PARSE_IGNORE_NON_ERRORS_FROM_INCLUDED_FILES = 0x4000
cindex.TranslationUnit.PARSE_RETAIN_EXCLUDED_CONDITIONAL_BLOCKS = 0x8000


class DepthCursor(cindex.Cursor):
    """Cindex cursor with concept of depth to prevent over-recursion when using
    walk_preorder. E.g. when processing a class if it has std::string as an
    argument, it will call walk_preorder on std::string to inspect it, and in
    turn on std::string's methods. We only want to recurse to a depth of 3
    (popart->Class->method).

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


exceptions = [
    'DynamicBaseOp', 'DynamicBinaryBaseOp', 'DynamicSliceBaseOp',
    'DynamicTernaryBaseOp', 'ElementWiseBinaryBaseOp', 'MultiConvBaseOp',
    'MultiConvDataGradBaseOp', 'MultiConvWeightsGradBaseOp', 'RandomBaseOp',
    'RandomNormalBaseOp', 'RandomUniformBaseOp', 'ReshapeBaseOp',
    'TransposeBaseOp', 'ZerosBaseOp', "FmodArg0GradOp", 'ExchangeDescriptor',
    "LeakyReluOpBaseAttributes", "MultiConvOptions", "ShapeOrLikeOp",
    'ReverseBaseOp', 'DropoutBaseOp', 'CollectivesBaseOp', 'SubsampleBaseOp',
    'BaseSliceOp', 'LossOp', 'BasePadOp', 'BasePadOutplaceOp',
    'OneWayUnaryInPlaceOp', 'OneWayUnaryOp', 'BaseSortOp',
    'ElementWiseNonLinearUnaryGradOp', 'ElementWiseBinaryOp',
    'ElementWiseUnaryBooleanOp', 'ElementWiseUnaryOp', 'SGD0VarUpdateOpBase',
    'DynamicBinaryBaseInplaceOp', 'ElementWiseInplaceUnaryOp',
    'DynamicTernaryBaseInplaceOp', 'TiedGatherOp', 'TiedGatherGradOp',
    'ScanOp', 'ConcatOp', 'ConcatInplaceOp', 'ConcatGradOp', 'ExpandInplaceOp',
    'ExpandGradOp', 'ExpandOp', 'IdentityInplaceOp', 'TransposeInplaceOp',
    'UpsampleOp', 'PopartLSTMOp', 'PopartLSTMGradOp'
]


def find_constructors(node: cindex.Cursor, base: cindex.Cursor) -> Dict:
    """Find info on all the constructors for the given node.

    Args:
        node (cindex.Cursor): The op constructor node
        base (cindex.Cursor): The base class.
    Returns:
        Dict: A dict containing all the required info about the constructor.
            See function return for what it contains.
    """
    args_ = []

    for c in node.get_arguments():
        ref = False
        const = False
        name = c.spelling
        type_ = c.type.spelling
        fulltype = type_
        has_default = False
        if re.match("const", type_):
            type_ = re.sub("&", "", type_)
            ref = True
        if re.match("const", type_):
            type_ = re.sub("const ", "", type_)
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
        if type_.strip() == "Op::Settings":
            name = "settings"
        if type_.strip() == "popart::OperatorIdentifier":
            name = "opid"
        args_.append({
            "arg_name":
            name.strip(),
            "alt_arg_name":
            type_.strip().lower().replace('popart::', '').replace('::', ''),
            "arg_type":
            type_.strip(),
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
    for i, arg in enumerate(args_):
        full_args += f"{'const ' if arg['const'] else ''}{arg['arg_type']}{' &' if  arg['ref'] else ''}"
        if i + 1 != len(args_):
            full_args += ", "

    # Get the relative include path
    file_ = node.location.file.name

    file_ = Path(file_).resolve().relative_to(
        Path(__file__).resolve().parents[1])

    return {
        "name":
        node.spelling,
        "constructors": [{
            "args": args_,
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


def is_template(node: cindex.Cursor) -> bool:
    """Regex test if a Cursor is a template class.

    Args:
        node (cindex.Cursor): The Cursor to test

    Returns:
        bool: True if detected it's a template, false otherwise.
    """
    return bool(re.match(r"[\w]+<[\w\s,]+>", node.spelling))


def process_op_header(
        filename: Path, ops_path: str,
        include_directories: List[str]) -> Tuple[Path, List[Dict]]:
    """Go through the file's definitions and process each child node in turn.

    Args:
        path (Path): Path of the hpp file
        ops_path (str): Path where the ops.hpp files are stored.

    Returns:
        List[Dict]: List of dicts with info on the ops
    """
    path = ops_path / filename
    ops: Dict = {}
    print(f"Processing {path}")

    # `-ferror-limit=0` is required to prevent `clang` from stopping on too many
    # errors.
    args = ['--language=c++', '--std=c++11', '-ferror-limit=0']
    args += [f'-I{dir}' for dir in include_directories]
    opts = bor(
        cindex.TranslationUnit.PARSE_SKIP_FUNCTION_BODIES,
        cindex.TranslationUnit.PARSE_DETAILED_PREPROCESSING_RECORD,
        cindex.TranslationUnit.PARSE_RETAIN_EXCLUDED_CONDITIONAL_BLOCKS,
        cindex.TranslationUnit.PARSE_INCOMPLETE,
        cindex.TranslationUnit.PARSE_KEEP_GOING,
        cindex.TranslationUnit.PARSE_INCLUDE_ATTRIBUTED_TYPES,
    )
    idx = cindex.Index.create()

    tu = idx.parse(str(path), args=args, options=opts)

    # Skip some known errors
    known_errors = ['stddef.h', 'stdarg.h']
    print("Clang translation unit diagnostics:")
    for d in tu.diagnostics:
        if d.Fatal and d.category_number == 1 and not any(
                x in d.spelling for x in known_errors):
            raise cindex.LibclangError(
                f"{d.category_name} {d.category_number} {d.spelling} fatal error."
            )

    for d in include_directories:
        if not Path(str(d)).is_dir() or Path(str(d)).is_file():
            raise FileNotFoundError(
                f"Include path {str(d)} not found. {__file__} exiting.")

    base_class = None
    has_pure_virtual = False

    for node in DepthCursor(cursor=tu.cursor,
                            depth=0).walk_preorder_depth(max_depth=3):
        # Skip if not in current file.
        if not (node.location.file and str(node.location.file) == str(path)):
            continue

        # Skip selected ops
        if node.spelling in exceptions:
            continue

        # Skip struct constructors
        if hasattr(
                node.semantic_parent, "kind"
        ) and node.semantic_parent.kind == cindex.CursorKind.STRUCT_DECL:
            continue

        if node.kind == cindex.CursorKind.CLASS_DECL:
            for c in node.get_children():
                if hasattr(c, "is_pure_virtual_method"
                           ) and c.is_pure_virtual_method():
                    has_pure_virtual = True
                    exceptions.append(node.spelling)
                    continue

        # Due to the depth first traversal, we will come to the base specifier
        # first so we can save the base node that our op derives from here.
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

    return (path, [o for o in ops.values() if o['file'] in str(path)])


def parse_ops(ops_dir: Path,
              filenames: List[Path],
              include_directories: List[Path],
              jobs: int = 1) -> Tuple[List, List, List]:
    """Run over all the files in the given dir and process with process_file,
    one thread per file.

    Args:
        ops_path (str): Path where the <op>.hpp files are stored
        jobs (int, optional): Number of threads. Defaults to 1.
    """
    data = None

    with Pool(jobs) as p:
        data = p.starmap(process_op_header,
                         [(name, ops_dir, include_directories)
                          for name in filenames])

    assert data is not None, "Op parsing failed."

    filenames = [Path(i[0]) for i in data]
    namespaces = [namespaces_from_filename(ops_dir, n) for n in filenames]
    opss = [i[1] for i in data]

    n_ops = sum(len(ops) for ops in opss)
    assert n_ops > 0, "No ops were parsed. Please check your arguments."
    print(f'Extracted metadata for {n_ops} ops.')

    return filenames, namespaces, opss


def parse_op_filenames(ops_dir: Path, filenames: Any) -> List[Path]:
    """The filenames are provided as a string of colon-separated absolute paths.
    Break this into a list of string, where each item is a path relative to the
    ops directory.

    Args:
        ops_dir (Path): Path where the ops are declared
        filenames (str): The filenames to parse

    Returns:
        List[Path]: The parsed filenames
    """
    filenames = filenames.split(';')
    filenames = [Path(name).resolve() for name in filenames]
    filenames = [os.path.relpath(name, ops_dir) for name in filenames]
    filenames = [Path(name) for name in filenames]
    return filenames


def namespaces_from_filename(ops_dir: Path, filename: Path) -> Tuple[str, ...]:
    """Convert a filename to namespaces.

    Example:
        ops_dir = .../popart/op
        filename = .../popart/op/foo.cpp
        namespaces = ('op', )

    Example:
        ops_dir = .../popart/op
        filename = .../popart/op/bar/baz.cpp
        namespaces = ('op', 'bar')

    Args:
        ops_dir (Path): Path where op headers live
        filename (Path): The file in which ops are declared

    Returns:
        Tuple[str]: The namespaces.
    """
    relpath = Path(os.path.relpath(filename, ops_dir))
    parts = (ops_dir.parts[-1], ) + relpath.parts[:-1]
    return parts


def main(filenames: List[Path], out: Path, include_directories: Any):
    proj_root = get_project_source_dir()
    ops_dir = proj_root / 'willow/include/popart/op'
    rel_filenames = parse_op_filenames(ops_dir, filenames)

    filenames, namespaces, opss = parse_ops(ops_dir, rel_filenames,
                                            include_directories, 16)

    metadata = [{
        'filename': str(f),
        'namespaces': n,
        'ops': o
    } for f, n, o in zip(filenames, namespaces, opss)]

    with open(out, 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--headers',
        type=str,
        required=True,
        help='Colon-separated list of absolute paths for op headers.',
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help=
        'The absolute path of the directory where metadata will be written to.',
    )
    parser.add_argument(
        "--include-directories",
        type=str,
        help='A semicolon separated list of include directories.')
    args = parser.parse_args()

    include_directories_: Any = []
    if args.include_directories:
        include_directories_ = set(args.include_directories.split(';'))
        include_directories_.discard('')
        include_directories_ = list(include_directories_)
        include_directories_ = [
            Path(inc).resolve() for inc in include_directories_
        ]
    main(args.headers, Path(args.out).resolve(), include_directories_)
