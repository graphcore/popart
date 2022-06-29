#!/usr/bin/env python3
# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# -*- coding: utf-8 -*-
#
#  Syntax: mkdoc.py [-I<path> ..] [.. a list of header files ..]
#
#  Extract documentation from C++ header files to use it in Python bindings
#

import os
import sys
import platform
import re

import ctypes.util

from clang import cindex
from clang.cindex import CursorKind
from collections import OrderedDict
from glob import glob
from threading import Thread, Semaphore
from multiprocessing import cpu_count

RECURSE_LIST = [
    CursorKind.TRANSLATION_UNIT, CursorKind.NAMESPACE, CursorKind.CLASS_DECL,
    CursorKind.STRUCT_DECL, CursorKind.ENUM_DECL, CursorKind.CLASS_TEMPLATE
]

PRINT_LIST = [
    CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL, CursorKind.ENUM_DECL,
    CursorKind.ENUM_CONSTANT_DECL, CursorKind.CLASS_TEMPLATE,
    CursorKind.FUNCTION_DECL, CursorKind.FUNCTION_TEMPLATE,
    CursorKind.CONVERSION_FUNCTION, CursorKind.CXX_METHOD,
    CursorKind.CONSTRUCTOR, CursorKind.FIELD_DECL
]

PREFIX_BLACKLIST = [CursorKind.TRANSLATION_UNIT]

CPP_OPERATORS = {
    '<=': 'le',
    '>=': 'ge',
    '==': 'eq',
    '!=': 'ne',
    '[]': 'array',
    '+=': 'iadd',
    '-=': 'isub',
    '*=': 'imul',
    '/=': 'idiv',
    '%=': 'imod',
    '&=': 'iand',
    '|=': 'ior',
    '^=': 'ixor',
    '<<=': 'ilshift',
    '>>=': 'irshift',
    '++': 'inc',
    '--': 'dec',
    '<<': 'lshift',
    '>>': 'rshift',
    '&&': 'land',
    '||': 'lor',
    '!': 'lnot',
    '~': 'bnot',
    '&': 'band',
    '|': 'bor',
    '+': 'add',
    '-': 'sub',
    '*': 'mul',
    '/': 'div',
    '%': 'mod',
    '<': 'lt',
    '>': 'gt',
    '=': 'assign',
    '()': 'call'
}

CPP_OPERATORS = OrderedDict(
    sorted(CPP_OPERATORS.items(), key=lambda t: -len(t[0])))

job_count = cpu_count()
job_semaphore = Semaphore(job_count)
errors_detected = False
docstring_width = int(70)


class NoFilenamesError(ValueError):
    pass


def d(s):
    return s if isinstance(s, str) else s.decode('utf8')


def sanitize_name(name):
    name = re.sub(r'type-parameter-0-([0-9]+)', r'T\1', name)
    for k, v in CPP_OPERATORS.items():
        name = name.replace('operator%s' % k, 'operator_%s' % v)
    name = re.sub('<.*>', '', name)
    name = ''.join([ch if ch.isalnum() else '_' for ch in name])
    name = re.sub('_$', '', re.sub('_+', '_', name))
    return '__doc_' + name


def process_comment(comment):
    result = ''

    # Remove C++ comment syntax
    leading_spaces = float('inf')
    for s in comment.expandtabs(tabsize=4).splitlines():
        s = s.strip()
        if s.startswith('/*'):
            s = s[2:].lstrip('*')
        elif s.endswith('*/'):
            s = s[:-2].rstrip('*')
        elif s.startswith('///'):
            s = s[3:]
        if s.startswith('*'):
            s = s[1:]
        if len(s) > 0:
            leading_spaces = min(leading_spaces, len(s) - len(s.lstrip()))
        result += s + '\n'

    # Remove any leading spaces present throughout all lines
    if leading_spaces != float('inf'):
        result2 = ""
        for s in result.splitlines():
            result2 += s[leading_spaces:] + '\n'
        result = result2

    # Doxygen tags
    cpp_group = r'([\w:\.<>\(\)]+)'
    param_group = r'([\[\w:,\]]+)'

    s = result
    s = re.sub(r'[\\@]c\s+%s' % cpp_group, r'``\1``', s)
    s = re.sub(r'[\\@]a\s+%s' % cpp_group, r'*\1*', s)
    s = re.sub(r'[\\@]e\s+%s' % cpp_group, r'*\1*', s)
    s = re.sub(r'[\\@]em\s+%s' % cpp_group, r'*\1*', s)
    s = re.sub(r'[\\@]b\s+%s' % cpp_group, r'**\1**', s)
    s = re.sub(r'[\\@]ingroup\s+%s' % cpp_group, r'', s)

    # Turns
    #
    # '''
    # Not parameter section here
    # \param A: description description description
    # \param B: description description description
    #     description description description
    # \param C:
    #     description description description
    # No longer parameter section here
    # '''
    #
    # into
    #
    # '''
    # Not parameter section here
    #
    # Args:
    #  A: description description description
    #  B: description description description
    #      description description description
    #  C:
    #      description description description
    # No longer parameter section here
    # '''
    def convert_parameters(comment):
        comment = comment.split('\n')
        pre_param_lines = []
        param_lines = []
        post_param_lines = []
        indent = -1
        for line in comment:
            space_len = len(line) - len(line.lstrip())
            if re.fullmatch(r'\s+', line):
                # empty line
                if post_param_lines != []:
                    post_param_lines.append(line)
                elif param_lines != []:
                    param_lines.append(line)
                else:
                    pre_param_lines.append(line)
            # this line contains a param
            elif re.match(
                    r'(\s)*[\\@]param%s?\s+%s' % (param_group, cpp_group),
                    line):
                param_lines.append(
                    re.sub(
                        r'(\s*)[\\@]param%s?\s+%s' % (param_group, cpp_group),
                        r'\1\3:', line))
                indent = space_len
            # line does not contain a param and we have finished with all the param lines
            elif post_param_lines != [] or (param_lines != []
                                            and space_len <= indent):
                post_param_lines.append(line)
            # we have not yet encountered a param line
            elif param_lines == []:
                pre_param_lines.append(line)
            # we're inside a param description
            else:
                param_lines.append(line)
        param_lines = [' ' + i for i in param_lines]  # indent param lines
        if indent == -1:
            return '\n'.join(pre_param_lines)
        else:
            return '\n'.join(pre_param_lines + ['', ' ' * indent + 'Args:'] +
                             param_lines + post_param_lines)

    # Turns
    #
    # '''
    # <pattern>: description description
    #     description description
    # <pattern>:
    #     description description description
    # '''
    #
    # into
    #
    # '''
    #
    # <replacement>:
    #  description description
    #  description description
    #
    # <replacement>:
    #  description description description
    # '''
    #
    # anywhere in the comment (the occurances of `pattern` need not be sequential)
    def convert_keyword(comment, pattern, replacement):
        comment = comment.split('\n')
        lines = []
        indent = -1
        inside_pattern = False
        for line in comment:
            space_len = len(line) - len(line.lstrip())
            # empty line
            if re.fullmatch(r'\s+', line):
                lines.append(line)
            # this line contains a pattern
            elif re.match(pattern, line):
                lines.append('')
                indent = space_len
                inside_pattern = True
                new_line = re.sub(pattern, replacement, line)
                # Move everything after the pattern to a new line if non-empty
                m = re.match(r'(\s*)(%s)\s*(\S.*$)' % replacement, new_line)
                if m:
                    lines.append(m.group(1) + m.group(2))
                    lines.append(' ' * (indent + 1) + m.group(3))
                else:
                    lines.append(line)
            # line does not contain a param and we have finished with all the param lines
            elif indent >= space_len or not inside_pattern:
                lines.append(line)
            # we have not yet encountered a param line
            elif indent >= space_len and inside_pattern:
                inside_pattern = False
                lines.append(line)
            # we're inside a pattern description
            else:
                # adjust indentation
                line = re.sub(r'^\s*(\S.*$)', ' ' * (indent + 1) + r'\1', line)
                lines.append(line)
        return '\n'.join(lines)

    s = convert_parameters(s)
    s = re.sub(r'(\s*)[\\@]tparam%s?\s+%s' % (param_group, cpp_group),
               r'\1Template parameter ``\2``: ', s)

    for in_, out_ in {
            'returns': 'Returns',
            'return': 'Returns',
            'authors': 'Authors',
            'author': 'Author',
            'copyright': 'Copyright',
            'date': 'Date',
            'remark': 'Remark',
            'sa': 'See Also',
            'see': 'See Also',
            'extends': 'Extends',
            'throws': 'Throws',
            'throw': 'Throws'
    }.items():
        s = convert_keyword(s, r'\s*[\\@]%s\s*' % in_, r'%s:' % out_)

    s = re.sub(r'[\\@]details\s*', r'\n\n', s)
    s = re.sub(r'[\\@]brief\s*', r'', s)
    s = re.sub(r'[\\@]short\s*', r'', s)
    s = re.sub(r'[\\@]ref\s*', r'', s)

    s = re.sub(r'(`[^`]+`)(?!`)', r':code:\1', s, flags=re.DOTALL)
    s = re.sub(r'[\\@]code\s?(.*?)\s?[\\@]endcode',
               r"```\n\1\n```\n",
               s,
               flags=re.DOTALL)

    s = re.sub(r'[\\@]warning\s?(.*?)\s?\n\n',
               r'$.. warning::\n\n\1\n\n',
               s,
               flags=re.DOTALL)
    # Deprecated expects a version number for reST and not for Doxygen. Here the first word of the
    # doxygen directives is assumed to correspond to the version number
    s = re.sub(r'[\\@]deprecated\s(.*?)\s?(.*?)\s?\n\n',
               r'$.. deprecated:: \1\n\n\2\n\n',
               s,
               flags=re.DOTALL)
    s = re.sub(r'[\\@]since\s?(.*?)\s?\n\n',
               r'.. versionadded:: \1\n\n',
               s,
               flags=re.DOTALL)
    s = re.sub(r'[\\@]todo\s?(.*?)\s?\n\n',
               r'$.. todo::\n\n\1\n\n',
               s,
               flags=re.DOTALL)

    # replace html tags
    s = re.sub(r'<a\s+href\s*=\s*[\'"]([^\'"]*)[\'"]>\s*(((?!</).)*)\s*</a>',
               r'`\2 <\1>`_',
               s,
               flags=re.DOTALL)
    s = re.sub(r'<tt>(.*?)</tt>', r'``\1``', s, flags=re.DOTALL)
    s = re.sub(r'<pre>(.*?)</pre>', r"```\n\1\n```\n", s, flags=re.DOTALL)
    s = re.sub(r'<em>(.*?)</em>', r'*\1*', s, flags=re.DOTALL)
    s = re.sub(r'<b>(.*?)</b>', r'**\1**', s, flags=re.DOTALL)
    s = re.sub(r'<li>', r'\n\n* ', s)
    s = re.sub(r'</?ul>', r'', s)
    s = re.sub(r'</li>', r'\n\n', s)

    # replace inline math formulas
    s = re.sub(r'[\\@]f\$(.*?)[\\@]f\$', r':math:`\1`', s, flags=re.DOTALL)
    # replace multiline math formulas
    s = re.sub(r'[\\@]f\[(.*?)[\\@]f\]', r'.. math::\1', s, flags=re.DOTALL)

    # This might be useful in the future. Not using at the moment as
    # even some Python comments use :: notation to refer to stuff
    # # replace :: with . as long as it is not preceded by std
    # s = re.sub(r'(?<!std)\b::\b', r'.', s)

    # replace words starting with hashtags
    s = re.sub(r'(?<=\s)#(%s)' % param_group, r'``\1``', s)
    # remove percentage signs before words
    s = re.sub(r'%(\w)', r'\1', s)

    s = s.replace('``true``', '``True``')
    s = s.replace('``false``', '``False``')

    return s.rstrip().lstrip('\n')


def extract(filename, node, prefix, output):
    if not (node.location.file is None
            or os.path.samefile(d(node.location.file.name), filename)):
        return 0
    if node.kind in RECURSE_LIST:
        sub_prefix = prefix
        if node.kind not in PREFIX_BLACKLIST:
            if len(sub_prefix) > 0:
                sub_prefix += '_'
            sub_prefix += d(node.spelling)
        for i in node.get_children():
            extract(filename, i, sub_prefix, output)
    if node.kind in PRINT_LIST:
        comment = d(node.raw_comment) if node.raw_comment is not None else ''
        comment = process_comment(comment)
        sub_prefix = prefix
        if len(sub_prefix) > 0:
            sub_prefix += '_'
        if len(node.spelling) > 0:
            name = sanitize_name(sub_prefix + d(node.spelling))
            output.append((name, filename, comment))


class ExtractionThread(Thread):
    def __init__(self, filename, parameters, output):
        Thread.__init__(self)
        self.filename = filename
        self.parameters = parameters
        self.output = output
        job_semaphore.acquire()

    def run(self):
        global errors_detected
        print('Processing "%s" ..' % self.filename, file=sys.stderr)
        try:
            index = cindex.Index(cindex.conf.lib.clang_createIndex(
                False, True))
            tu = index.parse(self.filename, self.parameters)
            extract(self.filename, tu.cursor, '', self.output)
        except BaseException:
            errors_detected = True
            raise
        finally:
            job_semaphore.release()


def read_args(args):
    parameters = []
    filenames = []
    if "-x" not in args:
        parameters.extend(['-x', 'c++'])
    if not any(it.startswith("-std=") for it in args):
        parameters.append('-std=c++11')
    parameters.append('-Wno-pragma-once-outside-header')

    if platform.system() == 'Darwin':
        dev_path = '/Applications/Xcode.app/Contents/Developer/'
        lib_dir = dev_path + 'Toolchains/XcodeDefault.xctoolchain/usr/lib/'
        sdk_dir = dev_path + 'Platforms/MacOSX.platform/Developer/SDKs'
        libclang = lib_dir + 'libclang.dylib'

        if os.path.exists(libclang):
            cindex.Config.set_library_path(os.path.dirname(libclang))

        if os.path.exists(sdk_dir):
            sysroot_dir = os.path.join(sdk_dir, next(os.walk(sdk_dir))[1][0])
            parameters.append('-isysroot')
            parameters.append(sysroot_dir)
    elif platform.system() == 'Windows':
        if 'LIBCLANG_PATH' in os.environ:
            library_file = os.environ['LIBCLANG_PATH']
            if os.path.isfile(library_file):
                cindex.Config.set_library_file(library_file)
            else:
                raise FileNotFoundError(
                    "Failed to find libclang.dll! "
                    "Set the LIBCLANG_PATH environment variable to provide a path to it."
                )
        else:
            library_file = ctypes.util.find_library('libclang.dll')
            if library_file is not None:
                cindex.Config.set_library_file(library_file)
    elif platform.system() == 'Linux':
        # LLVM switched to a monolithical setup that includes everything under
        # /usr/lib/llvm{version_number}/. We glob for the library and select
        # the highest version
        def folder_version(d):
            return [int(ver) for ver in re.findall(r'(?<!lib)(?<!\d)\d+', d)]

        llvm_dir = max(
            (path for libdir in ['lib64', 'lib', 'lib32']
             for path in glob('/usr/%s/llvm-*' % libdir)
             if os.path.exists(os.path.join(path, 'lib', 'libclang.so.1'))),
            default=None,
            key=folder_version)

        # Ability to override LLVM/libclang paths
        if 'LLVM_DIR_PATH' in os.environ:
            llvm_dir = os.environ['LLVM_DIR_PATH']
        elif llvm_dir is None:
            raise FileNotFoundError(
                "Failed to find a LLVM installation providing the file "
                "/usr/lib{32,64}/llvm-{VER}/lib/libclang.so.1. Make sure that "
                "you have installed the packages libclang1-{VER} and "
                "libc++-{VER}-dev, where {VER} refers to the desired "
                "Clang/LLVM version (e.g. 11). You may alternatively override "
                "the automatic search by specifying the LIBLLVM_DIR_PATH "
                "(for the LLVM base directory) and/or LIBCLANG_PATH (if "
                "libclang is located at a nonstandard location) environment "
                "variables.")

        if 'LIBCLANG_PATH' in os.environ:
            libclang_dir = os.environ['LIBCLANG_PATH']
        else:
            libclang_dir = max(
                (os.path.join(llvm_dir, libdir, 'libclang.so')
                 for libdir in ['lib64', 'lib', 'lib32'] if os.path.exists(
                     os.path.join(llvm_dir, libdir, 'libclang.so'))),
                default=None,
                key=folder_version)

        cindex.Config.set_library_file(libclang_dir)
        cpp_dirs = []

        if '-stdlib=libc++' not in args:
            cpp_dirs.append(
                max(glob('/usr/include/c++/*'),
                    default=None,
                    key=folder_version))

            cpp_dirs.append(
                max(glob(
                    '/usr/include/%s-linux-gnu/c++/*' % platform.machine()),
                    default=None,
                    key=folder_version))
        else:
            cpp_dirs.append(os.path.join(llvm_dir, 'include', 'c++', 'v1'))

        if 'CLANG_INCLUDE_DIR' in os.environ:
            cpp_dirs.append(os.environ['CLANG_INCLUDE_DIR'])
        else:
            cpp_dirs.append(
                max(glob(os.path.join(llvm_dir, 'lib', 'clang', '*',
                                      'include')),
                    default=None,
                    key=folder_version))

        cpp_dirs.append('/usr/include/%s-linux-gnu' % platform.machine())
        cpp_dirs.append('/usr/include')

        # Capability to specify additional include directories manually
        if 'CPP_INCLUDE_DIRS' in os.environ:
            cpp_dirs.extend([
                cpp_dir for cpp_dir in os.environ['CPP_INCLUDE_DIRS'].split()
                if os.path.exists(cpp_dir)
            ])

        for cpp_dir in cpp_dirs:
            if cpp_dir is None:
                continue
            parameters.extend(['-isystem', cpp_dir])

    for item in args:
        if item.startswith('-'):
            parameters.append(item)
        else:
            filenames.append(item)

    if len(filenames) == 0:
        raise NoFilenamesError("args parameter did not contain any filenames")

    return parameters, filenames


def extract_all(args):
    parameters, filenames = read_args(args)
    output = []
    for filename in filenames:
        thr = ExtractionThread(filename, parameters, output)
        thr.start()

    print('Waiting for jobs to finish ..', file=sys.stderr)
    for _ in range(job_count):
        job_semaphore.acquire()

    return output


def write_header(comments, out_file=sys.stdout):
    print('''/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       n1
#define __DOC2(n1, n2)                                   n1##_##n2
#define __DOC3(n1, n2, n3)                               n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __CAT2(__doc_, __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__)))
#define SINGLE_LINE_DOC(...)                             __CAT2(__singlelinedoc_, __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__)))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

''',
          file=out_file)
    name_ctr = 1
    name_prev = None
    for name, _, comment in list(sorted(comments, key=lambda x: (x[0], x[1]))):
        if name == name_prev:
            name_ctr += 1
            name = name + "_%i" % name_ctr
        else:
            name_prev = name
            name_ctr = 1
        print('\nstatic const char *%s =%sR"doc(%s)doc";' %
              (name, '\n' if '\n' in comment else ' ', comment),
              file=out_file)
        # For some reason sphinx messes up enum class values if they span more than
        # a single line. So we create another parameter to represent the comment
        # squished into a single line
        single_line_comment = ' '.join(
            [i.strip() for i in comment.split('\n') if i.strip() != ""])
        print('\nstatic const char *%s =%sR"doc(%s)doc";' %
              (name[:2] + "singleline" + name[2:],
               '\n' if '\n' in comment else ' ', single_line_comment),
              file=out_file)

    print('''
#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
''',
          file=out_file)


def mkdoc(args, width, output=None):
    if width != None:
        global docstring_width
        docstring_width = int(width)
    comments = extract_all(args)
    if errors_detected:
        return

    if output:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output)),
                        exist_ok=True)
            with open(output, 'w') as out_file:
                write_header(comments, out_file)
        except:
            # In the event of an error, don't leave a partially-written
            # output file.
            try:
                os.unlink(output)
            except:
                pass
            raise
    else:
        write_header(comments)
