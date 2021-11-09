# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""
This file generates the following:

- A .cpp file containing bindings for the op, taken from the op's hpp file,
    for each op, e.g.:
    ./build/popart/python/popart._internal.ir/bindings/op/zeros.gen.cpp
- A corresponding hpp file containing the bind<Op> declaration:
    ./build/popart/python/popart._internal.ir/bindings/op/zeros.gen.hpp
- A _all.gen.cpp file which runs all the bind<Op> functions
    (currently limited to a small subset of ops.).
    ./build/popart/python/popart._internal.ir/bindings/op/_all.gen.cpp
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path
from subprocess import CalledProcessError, check_output
from typing import Dict, Text

# jinja2 is not a common install. Raise error and point to readme in case of missing package.
try:
    import jinja2
except:
    raise ImportError(
        "Failed to import python package jinja2."
        "Please make sure you install all dependencies. See the README.md for details."
    )
from util import get_project_source_dir


def render_template(template_path: Path, **kwargs) -> Text:
    """Render the given template given **kwargs of data to use.

    Args:
        template_path (Path): Path to the template file.

    Returns:
        Text: The rendered template.
    """

    template = jinja2.Template(open(template_path).read())
    return template.render(**kwargs)


def create_rendered_files(metadata: Dict, template_path: Path,
                          out_dir: Path) -> None:
    """Create the generated files using the rendered templates. Will create a hpp and cpp
    file for each associated popart op header.

    Args:
        metadata (Dict): The metadata from the json file for the hpp in question.
        template_path (Path): Path to the template to render the cpp file.
        out_dir (Path): Path to output the files to.
    """
    proj_root = get_project_source_dir()

    filename = metadata['filename']
    op_header = os.path.relpath(filename, proj_root / 'willow/include')
    related_header = op_header.replace('popart/', 'bindings/')
    related_header = related_header.replace('.hpp', '.gen.hpp')
    namespaces = metadata['namespaces']
    fun_name = 'bind' + op_header.split('/')[-1].replace('.hpp',
                                                         '').capitalize()
    ops = metadata['ops']

    out = render_template(template_path,
                          op_header=op_header,
                          related_header=related_header,
                          namespaces=namespaces,
                          fun_name=fun_name,
                          ops=ops)

    filedir = out_dir / Path(*namespaces[1:])
    filepath = filedir / op_header.split('/')[-1].replace('.hpp', '.gen.cpp')
    Path(filedir).mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        f.write(out)

    out = render_template(Path(str(template_path).replace('.cpp', '.hpp')),
                          namespaces=namespaces,
                          fun_name=fun_name)
    filepath = Path(str(filepath).replace('.cpp', '.hpp'))
    with open(filepath, "w") as f:
        f.write(out)


def create__all_file(metadatas: tuple, template_path: Path,
                     out_dir: Path) -> None:
    """Create the _all.cpp.gen file that runs all the binding functions.

    Args:
        metadatas (tuple): All the metadata for all the ops which have had cpp/hpps generated.
        template_path (Path): Path to the _all.cpp.j2 template
        out_dir (Path): Path to save the rendered file to.
    """
    proj_root = get_project_source_dir()
    includes = []
    fun_names = []

    for data in metadatas:
        filename = data['filename']
        namespaces = data['namespaces']

        include = os.path.relpath(filename,
                                  proj_root / 'willow/include/popart')

        fun_name = '::'.join(namespaces[1:])
        fun_name += '::' if len(fun_name) else ''
        fun_name += 'bind'
        fun_name += include.split('/')[-1].replace('.hpp', '').capitalize()

        include = 'bindings/' + include
        include = include.replace('.hpp', '.gen.hpp')

        includes.append(include)
        fun_names.append(fun_name)

    filedir = out_dir
    filepath = filedir / '_all.gen.cpp'
    Path(filedir).mkdir(parents=True, exist_ok=True)

    out = render_template(template_path,
                          includes=includes,
                          fun_names=fun_names)
    with open(filepath, "w") as f:
        f.write(out)


def create_graph_file(metadatas: tuple, template_path: Path, out_dir: Path):
    """Create the graph.cpp.gen file that contains the createOp and createConnectedOp
    functions.

    Args:
        metadatas (tuple): The metadata for all the ops.
        template_path (Path): Path to the .j2 template.
        out_dir (Path): Directory to save to.
    """
    filedir = out_dir
    filepath = filedir / '../graph.gen.cpp'
    Path(filedir).mkdir(parents=True, exist_ok=True)

    out = render_template(template_path, metadatas=metadatas)
    with open(filepath, "w") as f:
        f.write(out)


def main(json_path: Path, out: Path, jobs: int) -> None:
    """Entry point for the script.
    Args:
        json_path (Path): Path to the op metadata file generated by `gen_op_metadata.py`
        out (Path): Path to the dir in which bindings will be written.
    """
    proj_root = get_project_source_dir()
    template_path = (proj_root /
                     'python/popart._internal.ir/templates/op/_op.cpp.j2')

    with open(json_path, 'r') as f:
        metadatas = json.load(f)

    with Pool(jobs) as p:
        args = [(metadata, template_path, out) for metadata in metadatas]
        p.starmap(create_rendered_files, args)

    template_path = (proj_root /
                     'python/popart._internal.ir/templates/op/_all.cpp.j2')
    create__all_file(metadatas, template_path, out)

    template_path = (proj_root /
                     'python/popart._internal.ir/templates/graph.cpp.j2')
    create_graph_file(metadatas, template_path, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json_path',
        type=str,
        required=True,
        help='Path to the op metadata file generated by `gen_op_metadata.py`.',
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Path to the dir in which bindings will be written.',
    )
    parser.add_argument(
        '--jobs',
        type=int,
        required=False,
        default=16,
        help='Number of jobs to use when multiprocessing hpp bindings.',
    )
    args_ = parser.parse_args()

    json_path_ = Path(args_.json_path).resolve()
    out_ = Path(args_.out).resolve()
    jobs_ = int(args_.jobs)

    main(json_path_, out_, jobs_)
