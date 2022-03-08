# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import onnx.defs
import io
import os
import numpy as np  # type: ignore
import textwrap
import subprocess


# Remove leading new lines and then set the identation on a
# multi line string.
def format_method(x, indent=0):
    if x.startswith('\n'):
        x = x[1:]
    x = textwrap.dedent(x)
    x = textwrap.indent(x, ' ' * indent)
    return x


overrideOP = {
    "ai.onnx.AveragePool:7": {
        "attributes": {
            "auto_pad": {
                "deprecated": True
            }
        }
    },
    "ai.onnx.Conv:1": {
        "verifyInput": True,
    },
    "ai.onnx.Conv:11": {
        "verifyInput": True,
    },
    "ai.onnx.AveragePool:1": {
        "verifyInput": True,
    },
    "ai.onnx.AveragePool:7": {
        "verifyInput": True,
    },
    "ai.onnx.AveragePool:10": {
        "verifyInput": True,
    },
    "ai.onnx.AveragePool:11": {
        "verifyInput": True,
    },
    "ai.onnx.MaxPool:1": {
        "verifyInput": True
    },
    "ai.onnx.MaxPool:8": {
        "verifyInput": True
    },
    "ai.onnx.MaxPool:10": {
        "verifyInput": True
    },
    "ai.onnx.MaxPool:11": {
        "verifyInput": True
    },
    "ai.onnx.Pad:2": {
        "verifyInput": True,
    }
}


class Schema:
    def __init__(self):
        self.domains = {}


class Domain:
    def __init__(self, name):
        self.name = name
        self.opsets = {}
        self.operations = []

    def AddOpDef(self, op):

        # Only go as far back as opset 6
        opset_version = op.version
        if op.version <= 6:
            opset_version = 6

        if str(opset_version) not in self.opsets:
            #Create new opset if not already defined
            opset = Opset(self, opset_version)
            self.opsets[str(opset_version)] = opset

        # Add the op into the right opset
        self.opsets[str(opset_version)].operators.append(op)

        # Add the op to the list of all ops
        self.operations.append(op)
        op.opset = self.opsets[str(opset_version)]

    def CppName(self):
        """ Return a c++ complient name for the operations """

        return "".join([s.capitalize() for s in self.name.split('.')])

    def isLatestOpVersion(self, op):
        """ If the op is the latest defined version """

        for _op in self.operations:
            if _op.name == op.name:
                if _op.version > op.version:
                    return False

        return True


class Opset:
    def __init__(self, domain, version):
        self.domain = domain
        self.version = version
        self.operators = []

    def RemoveDuplicates(self):
        """
        When putting ops into the opset, the opset can end up with multiple
        versions of the same op, this function will remove duplicates with the
        smallest version number
        """

        ops = []
        for op in self.operators:

            found = [x for x in ops if x.name == op.name]
            if len(found) > 0:
                ops.remove(found[0])
                ops.append(op)

            else:
                ops.append(op)

        self.operators = ops


class Attribute:
    def __init__(self, op, name, type, default):
        self.name = name
        self.op = op
        self.type = type
        self.default = default
        self.required = True

    def isFloat(self):
        return self.type == onnx.defs.OpSchema.AttrType.FLOAT

    def isList(self):
        if self.type == onnx.defs.OpSchema.AttrType.INTS:
            return True
        elif self.type == onnx.defs.OpSchema.AttrType.FLOATS:
            return True
        elif self.type == onnx.defs.OpSchema.AttrType.STRINGS:
            return True
        else:
            return False

    def isTensor(self):
        return (self.type == onnx.defs.OpSchema.AttrType.TENSOR
                or self.type == onnx.defs.OpSchema.AttrType.SPARSE_TENSOR)

    def CppType(self):
        """
        Determine the C++ type for an attribute
        """

        # Special case of Cast where we replace int with DataType
        if self.op.name == "Cast":
            return "const std::string&"
        elif self.type == onnx.defs.OpSchema.AttrType.INT:
            if self.required:
                return 'int64_t'
            else:
                if self.hasDefault():
                    return 'int64_t'
                else:
                    return "nonstd::optional<int64_t>"
        elif self.type == onnx.defs.OpSchema.AttrType.INTS:
            # Special case for axes in reduce operators as we need to distinguish
            # default params from empty params. In future we may want to
            # all optional parameters nonstd::optional of some sort.
            # TODO T21033: Investigate all other cases
            if self.op.name.lower().find("reduce") >= 0 and self.name == "axes":
                return "nonstd::optional<std::vector<int64_t>>"
            else:
                return 'const std::vector<int64_t>&'
        elif self.type == onnx.defs.OpSchema.AttrType.FLOAT:
            if self.required:
                return 'float'
            else:
                if self.hasDefault():
                    return 'float'
                else:
                    return "nonstd::optional<float>"
        elif self.type == onnx.defs.OpSchema.AttrType.FLOATS:
            return 'const std::vector<float>&'
        elif self.type == onnx.defs.OpSchema.AttrType.STRING:
            if self.required:
                return 'const std::string&'
            else:
                if self.hasDefault():
                    return 'const std::string&'
                else:
                    return "nonstd::optional<std::string>"
        elif self.type == onnx.defs.OpSchema.AttrType.STRINGS:
            return 'const std::vector<std::string>&'
        # Special case of Loop, If, Scan where we replace
        # onnx::GraphProto with Builder
        elif self.type == onnx.defs.OpSchema.AttrType.GRAPH:
            return 'const Builder&'
        elif self.type == onnx.defs.OpSchema.AttrType.TENSOR:
            return 'const ConstVoidData& '
        elif self.type == onnx.defs.OpSchema.AttrType.SPARSE_TENSOR:
            return 'const ConstVoidData& '
        else:
            return 'unknown'

    def isBoostOptional(self):
        if not self.required:
            if not self.hasDefault():
                if self.type == onnx.defs.OpSchema.AttrType.INT:
                    return True
                elif self.type == onnx.defs.OpSchema.AttrType.FLOAT:
                    return True
                elif self.type == onnx.defs.OpSchema.AttrType.STRING:
                    return True
        # TODO T21033: Investigate all other cases
        if self.type == onnx.defs.OpSchema.AttrType.INTS and self.op.name.lower(
        ).find("reduce") >= 0 and self.name == "axes":
            return True

        return False

    def hasDefault(self):
        if len(str(self.default)) == 0:
            return False
        else:
            if (self.type == onnx.defs.OpSchema.AttrType.TENSOR
                    or self.type == onnx.defs.OpSchema.AttrType.SPARSE_TENSOR):
                # Not sure how to express a tensor as a default value
                return False
            else:
                return True

    def hasDefaultValue(self):
        if self.required:
            return False
        if len(str(self.default)) == 0:
            if (self.type == onnx.defs.OpSchema.AttrType.TENSOR
                    or self.type == onnx.defs.OpSchema.AttrType.SPARSE_TENSOR):
                # Not sure how to express a tensor as a default value
                return False
            else:
                return True

        return True

    def hasPrimitiveDefaultValue(self):
        if len(str(self.default)) == 0:
            return False

        if self.type == onnx.defs.OpSchema.AttrType.INT:
            return True
        if self.type == onnx.defs.OpSchema.AttrType.FLOAT:
            return True

        return False

    def DefaultValue(self):
        """
        Return the default value for an attribute
        If there is a default value return that, else
        if the attribute is not required return the default value
        that the code can use to decide if the attribute can be
        left out
        """

        if len(str(self.default)) == 0:

            # Optional but not default
            if self.type == onnx.defs.OpSchema.AttrType.INT:
                return "nonstd::optional<int64_t>()"
            elif self.type == onnx.defs.OpSchema.AttrType.FLOAT:
                return "nonstd::optional<float>()"
            elif self.type == onnx.defs.OpSchema.AttrType.INTS:
                # Special case for axes in reduce operators as we need to distinguish
                # default params from an empty access list. In future we may want to
                # all optional parameters nonstd::optional of some sort.
                # TODO T21033: Investigate all other cases
                if self.op.name.lower().find(
                        "reduce") >= 0 and self.name == "axes":
                    return "nonstd::optional<std::vector<int64_t>>()"
                else:
                    return 'std::vector<int64_t>()'
            elif self.type == onnx.defs.OpSchema.AttrType.FLOATS:
                return 'std::vector<float>()'
            elif self.type == onnx.defs.OpSchema.AttrType.STRINGS:
                return 'std::vector<std::string>()'
            elif self.type == onnx.defs.OpSchema.AttrType.GRAPH:
                return 'onnx::GraphProto()'
            elif self.type == onnx.defs.OpSchema.AttrType.TENSOR:
                return '0'
            elif self.type == onnx.defs.OpSchema.AttrType.SPARSE_TENSOR:
                return '0'
            elif self.type == onnx.defs.OpSchema.AttrType.STRING:
                return 'std::string()'
            else:
                return 'UNKNOWN'

        else:

            if self.type == onnx.defs.OpSchema.AttrType.INT:
                return self.default.i
            elif self.type == onnx.defs.OpSchema.AttrType.FLOAT:
                value = np.round(self.default.f, 5)
                return str(value) + "f"
            elif self.type == onnx.defs.OpSchema.AttrType.STRING:
                return "\"" + self.default.s.decode("utf-8") + "\""
            elif self.type == onnx.defs.OpSchema.AttrType.INTS:
                return 'std::vector<int64_t>()'
            elif self.type == onnx.defs.OpSchema.AttrType.FLOATS:
                return 'std::vector<float>()'
            elif self.type == onnx.defs.OpSchema.AttrType.STRINGS:
                return 'std::vector<std::string>()'
            elif self.type == onnx.defs.OpSchema.AttrType.GRAPH:
                return 'onnx::GraphProto()'
            elif self.type == onnx.defs.OpSchema.AttrType.TENSOR:
                return '0'
            elif self.type == onnx.defs.OpSchema.AttrType.SPARSE_TENSOR:
                return '0'
            else:
                return '??'

    def isDeprecated(self):

        # The auto_pad attribute is deprecated in all ops
        if self.name == "auto_pad":
            return True

        if self.op.fullName() in overrideOP:
            if "attributes" in overrideOP[self.op.fullName()]:
                if self.name in overrideOP[self.op.fullName()]['attributes']:
                    if "deprecated" in overrideOP[
                            self.op.fullName()]["attributes"][self.name]:
                        return overrideOP[self.op.fullName()]["attributes"][
                            self.name]["deprecated"]
        return False

    def isRequired(self):
        if self.op.fullName() in overrideOP:
            if "attributes" in overrideOP[self.op.fullName()]:
                if self.name in overrideOP[self.op.fullName()]["attributes"]:
                    if "required" in overrideOP[
                            self.op.fullName()]["attributes"][self.name]:
                        return overrideOP[self.op.fullName()]["attributes"][
                            self.name]["required"]
        return None


class Operation:
    def __init__(self, name, version, support, onnx_schema=None):
        self.opset = None
        self.name = name
        self.version = version
        self.support = support
        self.attributes = []

        self.inputs = 0
        self.min_input = 1
        self.max_input = 1

        self.outputs = 0
        self.min_ouput = 1
        self.max_ouput = 1

        self.onnx_schema = onnx_schema

    def __lt__(self, other):
        """ Sort based on name """
        return self.name < other.name

    def CppName(self):
        """
        Return a C++ name for the operation
        Need the replace C++ key words
        """

        keywords = ["and", "or", "not", "xor", "if"]

        cppname = self.name.lower()
        if cppname in keywords:
            cppname = "logical_" + cppname

        return cppname

    def CppId(self):
        return self.name + "_" + str(self.version)

    def fullName(self):
        return "{}.{}:{}".format(self.opset.domain.name, self.name,
                                 self.version)

    def verifyInput(self):

        if self.fullName() in overrideOP:
            if "verifyInput" in overrideOP[self.fullName()]:
                return overrideOP[self.fullName()]["verifyInput"]
        return False


def spaces(n):
    """
    Return a string of spaces the same length as in the input string
    """
    return ' ' * n


def parseDefinitions():
    """ convert the schema definition to the internal representation """
    schema = Schema()

    for s in onnx.defs.get_all_schemas_with_history():

        domain = s.domain
        if domain == "":
            domain = "ai.onnx"

        if domain not in schema.domains:

            schema.domains[domain] = Domain(domain)

        op = Operation(s.name, s.since_version, s.support_level, s)

        for i in s.inputs:
            op.inputs = op.inputs + 1

        op.min_input = s.min_input

        if (s.max_input == 2147483647):
            op.max_input = -1
        else:
            op.max_input = s.max_input

        op.min_output = s.min_output

        if (s.max_output == 2147483647):
            op.max_output = -1
        else:
            op.max_output = s.max_output

        for k, v in s.attributes.items():

            attribute = Attribute(op, v.name, v.type, v.default_value)
            attribute.required = v.required

            op.attributes.append(attribute)

        schema.domains[domain].AddOpDef(op)

    for k, d in schema.domains.items():
        for v, opset in d.opsets.items():
            opset.RemoveDuplicates()

    for k, v in schema.domains.items():
        for op in v.operations:
            print("{}:{}:{} i:{}-{} o:{}-{}".format(
                k, op.name, op.version, op.min_input, op.max_input,
                op.min_output, op.max_output))
            if op.attributes is not None:
                for a in sorted(op.attributes, key=lambda x: x.hasDefault()):
                    print("- {} {} V:{}={} R:{} D:{}".format(
                        a.name, a.type, a.hasDefaultValue(), a.DefaultValue(),
                        a.required, a.isDeprecated()))
                    if op.fullName() in overrideOP:
                        if a.isRequired() is not None:
                            a.required = a.isRequired()
    return schema


def addHeader(f: io.TextIOWrapper, opset_version: int) -> None:
    f.write("// Copyright (c) 2018 Graphcore Ltd. All rights reserved.\n")
    f.write("/*\n")
    f.write(" * THIS IS AN AUTOGENERATED FILE, DO NOT EDIT DIRECTLY\n")
    f.write(" *\n")
    f.write(" * To regenerate this file run the gen_operators.py script\n")
    f.write(" */\n")
    # Add guard for hpp files
    file_base_name = os.path.basename(f.name)
    if file_base_name.endswith(".hpp"):
        f.write(
            f"#ifndef GUARD_NEURALNET_{file_base_name.upper().replace('.', '_')}\n"
        )
        f.write(
            f"#define GUARD_NEURALNET_{file_base_name.upper().replace('.', '_')}\n"
        )
    # Include the docs in the popart_opset#.gen.cpp files.
    if opset_version:
        f.write(f'#include "popart/docs/opset{opset_version}_docs.hpp"\n')


def genBuilderHpp(filename: str, schema: Schema) -> None:
    with io.open(filename, 'w') as f:

        addHeader(f, None)
        f.write("""
#include <string>
#include <vector>

#include "popart/debugcontext.hpp"
#include "popart/names.hpp"
#include "popart/domainopset.hpp"
#include "popart/vendored/optional.hpp"

namespace popart {
class Builder;
""")

        for k, v, in schema.domains.items():
            if k != 'ai.onnx':
                continue

            for opset_version, opset in sorted(v.opsets.items(),
                                               key=lambda x: int(x[0])):

                classname = v.CppName() + "Opset" + opset_version

                if int(opset_version) == 6:
                    baseclass = "DomainOpSet"
                else:
                    baseclass = v.CppName() + "Opset" + str(
                        int(opset_version) - 1)

                f.write("class {} : private {} {{\n".format(
                    classname, baseclass))
                f.write("\n")
                f.write("  protected:\n")
                f.write("    using {}::impl;\n".format(baseclass))

                f.write("  public:\n")
                f.write(
                    "    {}(std::unique_ptr<BuilderImpl>& impl_) : {}(impl_) {{}} \n"
                    .format(classname, baseclass))
                f.write("\n")

                f.write("    // return the opset version\n")
                f.write(
                    "    int getOpsetVersion() const override {{ return {};}} \n"
                    .format(opset_version))
                f.write("\n")

                seen = []
                for op in sorted(v.operations, key=lambda x: x.CppName()):
                    found = [x for x in opset.operators if x.name == op.name]

                    if len(found) == 0 and \
                       op.version < int(opset_version) and \
                       int(opset_version) > 6 and \
                       op.name not in seen:
                        f.write("    using {}::{};\n".format(
                            baseclass, op.CppName()))
                        seen.append(op.name)
                # Add a newline after the using statements.
                if len(seen) > 0:
                    f.write('\n')

                for op in sorted(opset.operators, key=lambda x: x.name):
                    f.write("    /**\n")
                    f.write("     * Add the '{}' to the model\n".format(
                        op.name))
                    f.write("     *\n")

                    if v.isLatestOpVersion(op):
                        f.write(
                            "     * https://github.com/onnx/onnx/blob/master/docs/Operators.md#{}\n"
                            .format(op.name))
                    else:
                        f.write(
                            "     * https://github.com/onnx/onnx/blob/master/docs/Changelog.md#{}-{}\n"
                            .format(op.name, op.version))

                    f.write("     *\n")

                    if op.inputs > 0:
                        f.write(
                            "     * \param args List of input tensor ids\n")

                    if op.min_output != op.max_output:
                        f.write(
                            "     * \param num_outputs The number of output tensor ids\n"
                        )

                    if int(opset_version) == 11 and op.name == 'Constant':
                        f.write(
                            format_method(
                                """
                            * \param value The 'value' attribute"
                            * \param is_value_sparse If true, set the 'sparse_value' attribute
                            """, 5))
                    else:
                        for a in sorted(op.attributes,
                                        key=lambda x: x.hasDefault()):
                            if not a.isDeprecated():
                                f.write(
                                    "     * \param {} The '{}' attribute \n".
                                    format(a.name, a.name))
                    f.write(
                        "     * \param name Optional identifier for the operation\n"
                    )
                    if op.max_output > 1:
                        f.write(
                            "     * \\return A list of normalized output tensors\n"
                        )
                    else:
                        f.write(
                            "     * \\return The normalized output tensor ids\n"
                        )
                    f.write("     */\n")

                    # Handle special case Constant_11
                    if int(opset_version) == 11 and op.name == 'Constant':
                        x = """
                            TensorId
                            constant(const ConstVoidData&  value,
                                     bool is_value_sparse = false,
                                     const DebugContext &debugContext = {});
                            """
                        x = format_method(x, 5)
                        f.write(x)
                        f.write('\n')
                    else:
                        if op.max_output == 1:
                            f.write("    TensorId\n")
                        else:
                            f.write("    std::vector<TensorId>\n")

                        f.write("    {}(".format(op.CppName()))
                        if op.inputs > 0:
                            f.write("const std::vector<TensorId>& args,\n")

                        # In the case of a variable number outputs, set the number of ouputs
                        if op.min_output != op.max_output:
                            f.write("     {}unsigned num_outputs,\n".format(
                                spaces(len(op.CppName()))))

                        for a in sorted(op.attributes,
                                        key=lambda x: x.hasDefault() or not x.
                                        required):
                            if not a.isDeprecated():
                                f.write("     {}{} {}".format(
                                    spaces(len(op.CppName())), a.CppType(),
                                    a.name))
                                if a.hasDefaultValue():
                                    f.write(" = {}".format(a.DefaultValue()))
                                f.write(",\n")
                        f.write(
                            "     {}const popart::DebugContext& debugContext = {});\n"
                            .format(spaces(len(op.CppName())), '{}'))
                        f.write("\n")
                f.write("};\n")
                f.write("\n")
            f.write("} // namespace popart\n")
            f.write("#endif")

    subprocess.run(["clang-format", "-i", filename])


def genBuilderCpp(filename: str, schema: Schema):
    with io.open(filename, 'w') as f:

        addHeader(f, None)

        f.write("""
#include "builder_helper.hpp"
#include "builderdebuginfo.hpp"
#include "popart/builder.gen.hpp"
namespace popart {
""")

        for k, v, in schema.domains.items():
            if k != 'ai.onnx':
                continue

            for opset_version, opset in sorted(v.opsets.items(),
                                               key=lambda x: int(x[0])):

                classname = v.CppName() + "Opset" + opset_version

                for op in sorted(opset.operators):
                    if int(opset_version) == 11 and op.name == 'Constant':
                        x = """
                            TensorId
                            AiOnnxOpset11::constant(const ConstVoidData&  value,
                                                    bool is_value_sparse,
                                                    const DebugContext& debugContext) {
                              std::map<std::string, popart::any> attributes;
                              if (is_value_sparse) {
                                  throw error("Attributes of type `sparse_tensor' are currently not supported.");
                              } else {
                                  attributes["value"] = value;
                              }

                              BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, {}, attributes);
                              attributes.insert({sDebugInfoId, di.getId()});

                              auto outputs = impl->op(Onnx::Operators::Constant_11,
                                              getOpsetVersion(),
                                              {},
                                              attributes,
                                              {di});
                              di.setOutputs(outputs);
                              return outputs.at(0);
                            }
                            """
                        x = format_method(x)
                        f.write(x)
                        f.write('\n')
                        continue
                    if op.max_output == 1:
                        f.write("TensorId\n")
                    else:
                        f.write("std::vector<TensorId>\n")

                    f.write("{}::{}(".format(classname, op.CppName()))

                    if op.inputs > 0:
                        f.write("const std::vector<TensorId>& args,\n")

                    # In the case of a variable number outputs, set the number of ouputs
                    if op.min_output != op.max_output:
                        f.write("{}  {} unsigned num_outputs,\n".format(
                            spaces(len(classname)), spaces(len(op.CppName()))))

                    for a in sorted(
                            op.attributes,
                            key=lambda x: x.hasDefault() or not x.required):
                        if not a.isDeprecated():
                            f.write("{}  {} {} {},\n".format(
                                spaces(len(classname)),
                                spaces(len(op.CppName())), a.CppType(),
                                a.name))
                    f.write(
                        "{}  {} const popart::DebugContext& debugContext) {{\n"
                        .format(spaces(len(classname)),
                                spaces(len(op.CppName()))))

                    f.write(
                        "  std::map<std::string, popart::any> attributes;\n")
                    for a in op.attributes:
                        if not a.isDeprecated():
                            if a.required:
                                isLoopOrScanBody = (
                                    op.name == "Loop"
                                    or op.name == "Scan") and a.name == "body"
                                isIfBranch = op.name == "If" and (
                                    a.name == "else_branch"
                                    or a.name == "then_branch")
                                if isLoopOrScanBody or isIfBranch:
                                    f.write(
                                        "  // Special case where we convert from a Builder object to an\n"
                                    )
                                    f.write(
                                        "  // onnx::GraphProto object so as not to expose the onnx class\n"
                                    )
                                    f.write("  // at the API level\n")
                                    f.write(
                                        "  attributes[\"{}\"] = io::getModelFromString({}.getModelProto()).graph();\n"
                                        .format(a.name, a.name))
                                elif op.name == "Cast":
                                    f.write(
                                        "  // Special case where we cast from DataType to int\n"
                                    )
                                    f.write(
                                        "  DataType toDataType = dataTypeFromString({});\n"
                                        .format(a.name))
                                    f.write(
                                        "  attributes[\"{}\"] = static_cast<int>(onnxutil::getTPDataType(toDataType));\n"
                                        .format(a.name, a.name))
                                else:
                                    f.write(
                                        "  attributes[\"{}\"] = {};\n".format(
                                            a.name, a.name))
                            elif a.isTensor():
                                f.write("  attributes[\"{}\"] = {};\n".format(
                                    a.name, a.name))
                            else:
                                if a.isList() and not a.isBoostOptional():
                                    f.write("  if (!{}.empty()) {{\n".format(
                                        a.name))
                                elif a.isFloat() and not a.isBoostOptional():
                                    f.write(
                                        "  if (std::abs({} - {}) >  std::numeric_limits<{}>::epsilon()) {{\n"
                                        .format(a.name, a.DefaultValue(),
                                                a.CppType()))
                                else:
                                    if a.hasPrimitiveDefaultValue():
                                        f.write("  // Workaround Onnx not " +
                                                "applying default values " +
                                                "during type/shape inference" +
                                                "\n  {\n")
                                    else:
                                        f.write("  if ({} != {}) {{\n".format(
                                            a.name, a.DefaultValue()))

                                if a.isBoostOptional():
                                    f.write("    attributes[\"{}\"] = *{};\n".
                                            format(a.name, a.name))
                                else:
                                    f.write("    attributes[\"{}\"] = {};\n".
                                            format(a.name, a.name))
                                f.write("  }\n")

                    if op.inputs > 0:
                        f.write(
                            "  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, args, attributes);\n"
                        )
                    else:
                        f.write(
                            "  BuilderDebugInfo di(debugContext, __POPART_FUNCTION_NAME__, {}, attributes);\n"
                        )

                    f.write(
                        "  attributes.insert({sDebugInfoId, di.getId()});\n")

                    f.write("  auto outputs = impl->op(Onnx::Operators::{},\n".
                            format(op.CppId()))

                    # Add the opset version
                    f.write("                  getOpsetVersion(),\n")

                    # Add the input tensors
                    if op.inputs > 0:
                        f.write("                  args,\n")
                    else:
                        f.write("                  {},\n")

                    if op.min_output != op.max_output:
                        f.write("                  num_outputs,\n")

                    f.write("                  attributes,\n")
                    f.write("                  {di}")

                    if op.verifyInput():
                        f.write(",\n")
                        f.write(
                            "                  [this](const std::vector<TensorId> &inputs_,\n"
                        )
                        f.write(
                            "                         std::map<std::string, popart::any> attributes_) {\n"
                        )
                        f.write(
                            "                     verify_{}_{}(this->impl, inputs_, attributes_);\n"
                            .format(classname, op.CppId()))
                        f.write("                  }")

                    f.write(");\n")
                    f.write("  di.setOutputs(outputs);\n")
                    if op.max_output == 1:
                        f.write("  return outputs[0];\n")
                    else:
                        f.write("  return outputs;\n")

                    f.write("}\n")
                    f.write("\n")
        f.write("} // namespace popart\n")
    subprocess.run(["clang-format", "-i", filename])


def genPythonBuilderBinds(schema: Schema) -> None:
    """
    Generate the python bindings for all of the onnx operators, per opset.

    Each opset's operators will be stored in a different file. This speeds
    up compile time.
    """

    for k, v, in schema.domains.items():
        if k != 'ai.onnx':
            continue

        ops = []

        for opset_version, opset in sorted(v.opsets.items(),
                                           key=lambda x: int(x[0])):
            opset_dir = os.path.join("python", "popart",
                                     f"popart_opset{opset_version}")
            os.makedirs(opset_dir, exist_ok=True)
            filename = os.path.join(opset_dir,
                                    f"popart_opset{opset_version}.gen.cpp")
            with io.open(filename, 'w') as f:
                addHeader(f, opset_version)
                # Add the include file.
                f.write(f"""#include <pybind11/functional.h>

                #include <pybind11/numpy.h>
                #include <pybind11/pybind11.h>
                #include <pybind11/stl.h>

                #include "../shared_cpp/np_utils.hpp"
                #include "../shared_cpp/pyarray_accessor.hpp"

                #include "popart/builder.hpp"
                #include "popart/tensors.hpp"
                #include "popart/version.hpp"

                namespace py = pybind11;
                using namespace popart;

                PYBIND11_MODULE(popart_opset{opset_version}, m) {{
                """)
                # Add all ops in the this op set
                for op in opset.operators:

                    # Create a list of op with the greatest version number less than the opset version
                    # This is to deal with the opset-6
                    found = [x for x in ops if x.name == op.name]
                    if len(found) > 0:
                        ops.remove(found[0])

                    ops.append(op)

                classname = v.CppName() + "Opset" + opset_version

                # For each opset
                f.write(f"py::class_<{classname}>(m, \"{classname}\")\n")
                for op in sorted(ops):

                    def getFunc(f, op):
                        # Operator
                        f.write(f"  .def(\"{op.CppName()}\",\n")

                        # Special case of the constant operator
                        if op.name == "Constant":
                            if (op.version == 11):
                                x = f"""
                                    []({classname} &opset, py::array array, bool is_value_sparse, const DebugContext& debugContext) {{
                                    array = makeContiguous(array);
                                    ConstVoidData initData;
                                    initData.data = array.request().ptr;
                                    initData.info = getTensorInfo(array);
                                    return opset.constant(initData, is_value_sparse, debugContext);
                                    }},
                                    py::arg("value"),
                                    py::arg("is_value_sparse") = false,
                                    py::arg("debugContext") = std::string())
                                    """
                                x = format_method(x, 7)
                                f.write(x)
                                return
                            else:
                                x = f"""
                                    []({classname} &opset, py::array array, const DebugContext& debugContext) {{
                                    array = makeContiguous(array);
                                    ConstVoidData initData;
                                    initData.data = array.request().ptr;
                                    initData.info = getTensorInfo(array);
                                    return opset.constant(initData, debugContext);
                                    }},
                                    """
                                x = format_method(x, 7)
                                f.write(x)
                        # Special case for the constantofshape operator
                        elif op.name == "ConstantOfShape":
                            x = f"\
                            []({classname} &opset,\
                                const std::vector<TensorId> &args,\
                                py::array array,\
                                const DebugContext &debugContext) {{\
                                ConstVoidData initData;\
                                initData.data = array.request().ptr;\
                                initData.info = getTensorInfo(array);\
                                return opset.constantofshape(args, initData, debugContext);\
                            }},"

                            x = format_method(x, 7)
                            f.write(x)
                        else:

                            # Add the lamda
                            f.write(f"[]({classname} &opset,\n")

                            if op.inputs > 0:
                                f.write("const std::vector<TensorId> &args,\n")

                            if op.min_output != op.max_output:
                                f.write("unsigned num_outputs,\n")

                            for a in sorted(op.attributes,
                                            key=lambda x: x.hasDefault() or
                                            not x.required):
                                if not a.isDeprecated():
                                    f.write(f"{a.CppType()} {a.name},\n")

                            f.write("const DebugContext& debugContext) -> ")

                            # Call the builder method
                            if op.max_output == 1:
                                f.write("TensorId")
                            else:
                                f.write("std::vector<TensorId>")

                            f.write("{\n")
                            f.write(f"return opset.{op.CppName()}(")

                            if op.inputs > 0:
                                f.write("args,\n")

                            if op.min_output != op.max_output:
                                f.write("num_outputs,\n")

                            for a in sorted(op.attributes,
                                            key=lambda x: x.hasDefault() or
                                            not x.required):
                                if not a.isDeprecated():
                                    f.write(f"{a.name},\n")

                            f.write("debugContext);\n")
                            f.write("},\n")

                        # Define the python function arguments
                        if op.inputs > 0:
                            f.write("py::arg(\"args\"),\n")

                        if op.min_output != op.max_output:
                            f.write("py::arg(\"num_outputs\"),\n")

                        for a in sorted(op.attributes,
                                        key=lambda x: x.hasDefault() or not x.
                                        required):
                            if not a.isDeprecated():
                                f.write(f"py::arg(\"{a.name}\")")

                                if a.hasDefaultValue():
                                    f.write(f" = {a.DefaultValue()}")
                                f.write(",\n")

                        f.write("py::arg(\"debugContext\") = std::string(),\n")
                        f.write(
                            f"OPSET{opset_version}_DOC(popart, {op.CppName()}, opset{opset_version}))\n"
                        )

                    getFunc(f, op)

                f.write(";\n")
                f.write("}\n")

            subprocess.run(["clang-format", "-i", filename])


def genPythonDocs(schema: Schema) -> None:
    for k, v, in schema.domains.items():
        if k != 'ai.onnx':
            continue

        ops = []

        for opset_version, opset in sorted(v.opsets.items(),
                                           key=lambda x: int(x[0])):
            filename = f"willow/include/popart/docs/opset{opset_version}_docs.hpp"
            with io.open(filename, 'w') as f:
                write_docs_header(f, opset_version)

                # Add all ops in the this op set
                for op in opset.operators:

                    # Create a list of op with the greatest version number less than the opset version
                    # This is to deal with the opset-6
                    found = [x for x in ops if x.name == op.name]
                    if len(found) > 0:
                        ops.remove(found[0])

                    ops.append(op)
                for op in sorted(ops):
                    f.write(
                        f"static const char *__doc_popart_{op.CppName()}_opset{opset_version} =\n"
                    )
                    f.write(f"R\"doc({op.onnx_schema.doc})doc\";\n\n")
                f.write("#endif")

            subprocess.run(["clang-format", "-i", filename])


def write_docs_header(f: io.TextIOWrapper, opset_version: int) -> None:
    f.write("// Copyright (c) 2021 Graphcore Ltd. All rights reserved.\n")
    f.write(f"#ifndef GUARD_OPSET{opset_version}_DOCS_HPP\n")
    f.write(f"#define GUARD_OPSET{opset_version}_DOCS_HPP")

    def_str = """
/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by gen_operators.py.
*/

#define __OPSETX_EXPAND(x) x
#define __OPSETX_COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...) COUNT
#define __OPSETX_VA_SIZE(...) __OPSETX_EXPAND(__OPSETX_COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __OPSETX_CAT1(a, b) a##b
#define __OPSETX_CAT2(a, b) __OPSETX_CAT1(a, b)
#define __OPSETX_DOC1(n1) __doc_##n1
#define __OPSETX_DOC2(n1, n2) __doc_##n1##_##n2
#define __OPSETX_DOC3(n1, n2, n3) __doc_##n1##_##n2##_##n3
#define __OPSETX_DOC4(n1, n2, n3, n4) __doc_##n1##_##n2##_##n3##_##n4
#define __OPSETX_DOC5(n1, n2, n3, n4, n5) __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __OPSETX_DOC6(n1, n2, n3, n4, n5, n6)                                         \
  __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __OPSETX_DOC7(n1, n2, n3, n4, n5, n6, n7)                                     \
  __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define OPSETX_DOC(...)                                                               \
  __OPSETX_EXPAND(__OPSETX_EXPAND(__OPSETX_CAT2(__OPSETX_DOC, __OPSETX_VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
"""

    f.write(def_str.replace("OPSETX", "OPSET" + str(opset_version)))


def genOpIdentifiersHpp(filename: str, schema: Schema) -> None:
    with io.open(filename, 'w') as f:
        addHeader(f, None)
        f.write("""
#include "popart/attributes.hpp"
#include "popart/error.hpp"
#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"

namespace popart {

namespace Onnx {
namespace Operators {
""")
        for k, v, in schema.domains.items():
            if k != 'ai.onnx':
                continue

            ops = v.operations
            '''
            for opset_version, opset in sorted(v.opsets.items(),
                                               key=lambda x: int(x[0])):

                print("Opset {}", opset_version)
                 # Add all ops in the this op set
                for op in opset.operators:
                    ops.append(op)
            '''

            for op in sorted(ops):
                f.write("const static OperatorIdentifier " + op.name + "_" +
                        str(op.version))

                if (op.min_input == op.max_input):
                    numInputs = str(op.min_input)
                else:
                    numInputs = "{{{}, {}}}".format(str(op.min_input),
                                                    str(op.max_input))

                if (op.min_output == op.max_output):
                    numOutputs = str(op.min_output)
                else:
                    numOutputs = str(op.max_output)
                    #numOutputs = "{{{}, {}}}".format(str(op.min_output), str(op.max_output))

                f.write("(Domain::ai_onnx, \"{}\", {}, {}, {});".format(
                    op.name, str(op.version), numInputs, numOutputs))

                f.write("\n")

        f.write("}\n")
        f.write("\n")

        f.write("namespace AiOnnx {\n")
        for k, v, in schema.domains.items():
            if k != 'ai.onnx':
                continue

            for opset_version, opset in sorted(v.opsets.items(),
                                               key=lambda x: int(x[0])):

                f.write("namespace OpSet{} {{\n".format(str(opset_version)))
                # Add all ops in the this op set

                seen = []

                for op in sorted(v.operations, key=lambda x: x.name):
                    found = [
                        x for x in v.operations if x.name == op.name
                        and x.version <= int(opset_version)
                    ]

                    if (op.name not in seen):
                        if (len(found) > 0):
                            seen.append(found[-1].name)
                            f.write(
                                "const static OperatorIdentifier {} = Operators::{}_{};"
                                .format(op.name, op.name,
                                        str(found[-1].version)))
                            f.write("\n")

                f.write("}\n")

        f.write("}\n")

        f.write("}\n")

        f.write("}\n")

        f.write("#endif")


# Given a domain, return the highest version of each op
# whose version is lower than or equal to the opsetVersion.
def getOpsInOpset(domain: Domain, opsetVersion: int) -> list:
    result = {}
    for op in domain.operations:
        if op.version <= opsetVersion:
            if op.name not in result:
                result[op.name] = op
            elif result[op.name].version < op.version:
                result[op.name] = op

    result = {k: v for k, v in result.items() if not v.onnx_schema.deprecated}
    result = {k: v.version for k, v in result.items()}

    return sorted([(name, version) for name, version in result.items()],
                  key=lambda x: x[0])


def genOpsetsHpp(filename: str, schema: Schema) -> None:
    with open(filename, 'w') as f:
        addHeader(f, None)

        f.write("""
#include <map>
#include "popart/names.hpp"
#include "popart/operators.hpp"

namespace popart {
using OpTypeMap = std::map<OpType, OperatorIdentifier>;
using OpsetMap = std::map<std::pair<OpDomain, OpVersion>, OpTypeMap>;

// Using the getOpsets function protects us from error with the static initialization order.
// If `opsets` were some static class, we could not guarantee
// that it would be initialized before all the static
// OperatorIdentifiers it uses.
// But the return of `getOpsets` can be stored to a static
// variable, after we know the OperatorIdentifiers have been
// initialized.
OpsetMap getOpsets() {
  OpsetMap opsets;
""")

        for domain_name, domain in schema.domains.items():
            if domain_name != 'ai.onnx':
                continue

            domain_name = domain_name.replace('.', '_')

            for opset_version, opset in sorted(domain.opsets.items(),
                                               key=lambda x: int(x[0]),
                                               reverse=True):
                ops = getOpsInOpset(domain, int(opset_version))

                f.write(
                    f'  opsets[{{"{domain_name.replace("_", ".")}", {opset_version}}}] = {{\n'
                )

                def format_entry(x):
                    name = x[0]
                    version = x[1]
                    return f'{{"{name}", Onnx::AiOnnx::OpSet{opset_version}::{name}}}'

                entries = [format_entry(x) for x in ops]
                entries = ',\n    '.join(entries)
                f.write('    ' + entries)
                f.write('};\n\n')

        f.write("""
  return opsets;
}
} // namespace popart
#endif
""")
    subprocess.run(["clang-format", "-i", filename])


def main():

    schema = parseDefinitions()

    genBuilderHpp('willow/include/popart/builder.gen.hpp', schema)
    genBuilderCpp('willow/src/builder.gen.cpp', schema)
    genOpIdentifiersHpp('willow/include/popart/onnxoperators.gen.hpp', schema)
    genPythonBuilderBinds(schema)
    genPythonDocs(schema)
    genOpsetsHpp('willow/src/opsets.gen.hpp', schema)


if __name__ == '__main__':
    main()
