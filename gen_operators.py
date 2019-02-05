import onnx.defs
import io
import numpy as np  # type: ignore

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
    "ai.onnx.AveragePool:1": {
        "verifyInput": True,
    },
    "ai.onnx.AveragePool:7": {
        "verifyInput": True,
    },
    "ai.onnx.MaxPool:1": {
        "verifyInput": True
    },
    "ai.onnx.MaxPool:8": {
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
    When putting ops into the opset, the opset can end up with 
    multiple version of the same op, this function will remove duplicates with the
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

    def CppType(self):
        """
    Determine the C++ type for an attribute
    """

        # Special case of Cast where we replace int with DataType
        if self.op.name == "Cast" and self.op.version == 6:
            return "DataType"
        elif self.type == onnx.defs.OpSchema.AttrType.INT:
            if self.required:
                return 'int64_t'
            else:
                if self.hasDefault():
                    return 'int64_t'
                else:
                    return "boost::optional<int64_t>"
        elif self.type == onnx.defs.OpSchema.AttrType.INTS:
            return 'const std::vector<int64_t>&'
        elif self.type == onnx.defs.OpSchema.AttrType.FLOAT:
            if self.required:
                return 'float'
            else:
                if self.hasDefault():
                    return 'float'
                else:
                    return "boost::optional<float>"
        elif self.type == onnx.defs.OpSchema.AttrType.FLOATS:
            return 'const std::vector<float>&'
        elif self.type == onnx.defs.OpSchema.AttrType.STRING:
            if self.required:
                return 'const std::string&'
            else:
                if self.hasDefault():
                    return 'const std::string&'
                else:
                    return "boost::optional<std::string>"
        elif self.type == onnx.defs.OpSchema.AttrType.STRINGS:
            return 'const std::vector<std::string>&'
        elif self.type == onnx.defs.OpSchema.AttrType.GRAPH:
            return 'int'
        elif self.type == onnx.defs.OpSchema.AttrType.TENSOR:
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

        return False

    def hasDefault(self):
        if len(str(self.default)) == 0:
            return False
        else:
            return True

    def hasDefaultValue(self):
        if self.required:
            return False
        if len(str(self.default)) == 0:
            return True

        return True

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
                return "boost::optional<int64_t>()"
            elif self.type == onnx.defs.OpSchema.AttrType.FLOAT:
                return "boost::optional<float>()"
            elif self.type == onnx.defs.OpSchema.AttrType.INTS:
                return 'std::vector<int64_t>()'
            elif self.type == onnx.defs.OpSchema.AttrType.FLOATS:
                return 'std::vector<float>()'
            elif self.type == onnx.defs.OpSchema.AttrType.STRINGS:
                return 'std::vector<std::string>()'
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
                return '0'
            elif self.type == onnx.defs.OpSchema.AttrType.TENSOR:
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


class Operation:
    def __init__(self, name, version, support):
        self.opset = None
        self.name = name
        self.version = version
        self.support = support
        self.attributes = []

        self.inputs = 0

        self.outputs = 0
        self.min_ouput = 1
        self.max_ouput = 1

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

        op = Operation(s.name, s.since_version, s.support_level)

        for i in s.inputs:
            op.inputs = op.inputs + 1

        op.min_output = s.min_output
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
            print("{}:{}:{} i:{} o:{}-{}".format(k, op.name, op.version,
                                                 op.inputs, op.min_output,
                                                 op.max_output))
            if op.attributes is not None:
                for a in sorted(op.attributes, key=lambda x: x.hasDefault()):
                    print("- {} {} V:{}={} R:{} D:{}".format(
                        a.name, a.type, a.hasDefaultValue(), a.DefaultValue(),
                        a.required, a.isDeprecated()))

    return schema


def addHeader(f):

    f.write("/*\n")
    f.write(" * THIS IS AN AUTOGENERATED FILE, DO NOT EDIT DIRECTLY\n")
    f.write(" *\n")
    f.write(" * To regenerated ths file run the gen_operators.py script\n")
    f.write(" */\n")


def genBuilderHpp(filename, schema):
    with io.open(filename, 'w') as f:

        addHeader(f)

        for k, v, in schema.domains.items():
            for opset_version, opset in sorted(v.opsets.items()):

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
                f.write("    int getOpsetVersion() const override {{ return {};}} \n".format(opset_version))
                f.write("\n")

                seen = []
                for op in v.operations:
                    found = [x for x in opset.operators if x.name == op.name]

                    if len(found) == 0 and \
                       op.version < int(opset_version) and \
                       int(opset_version) > 6 and \
                       op.name not in seen:
                        f.write("    using {}::{};\n".format(
                            baseclass, op.CppName()))
                        seen.append(op.name)

                for op in opset.operators:
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
                        f.write("     * \param args List of input tensor ids\n")

                    if op.min_output != op.max_output:
                        f.write(
                            "     * \param num_outputs The number of output tensor ids\n"
                        )

                    for a in sorted(
                            op.attributes, key=lambda x: x.hasDefault()):
                        if not a.isDeprecated():
                            f.write("     * \param {} The '{}' attribute \n".
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
                    if op.max_output > 1:
                        f.write("    std::vector<TensorId>\n")
                    else:
                        f.write("    TensorId\n")
                    f.write("    {}(".format(op.CppName()))
                    if op.inputs > 0:
                        f.write("const std::vector<TensorId>& args,\n".format(
                            op.CppName()))

                    # In the case of a variable number outputs, set the number of ouputs
                    if op.min_output != op.max_output:
                        f.write("     {}unsigned num_outputs,\n".format(
                            spaces(len(op.CppName()))))

                    for a in sorted(
                            op.attributes,
                            key=lambda x: x.hasDefault() or not x.required):
                        if not a.isDeprecated():
                            f.write("     {}{} {}".format(
                                spaces(len(op.CppName())), a.CppType(),
                                a.name))
                            if a.hasDefaultValue():
                                f.write(" = {}".format(a.DefaultValue()))
                            f.write(",\n")
                    f.write("     {}const std::string &name = \"\");\n".format(
                        spaces(len(op.CppName()))))
                    f.write("\n")

                f.write("};\n")
                f.write("\n")


def genBuilderCpp(filename, schema):
    with io.open(filename, 'w') as f:

        addHeader(f)

        for k, v, in schema.domains.items():
            for opset_version, opset in sorted(v.opsets.items()):

                classname = v.CppName() + "Opset" + opset_version

                for op in sorted(opset.operators):

                    if op.max_output > 1:
                        f.write("std::vector<TensorId>\n")
                    else:
                        f.write("TensorId\n")

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
                    f.write("{}  {} const std::string& name) {{\n".format(
                        spaces(len(classname)), spaces(len(op.CppName()))))

                    f.write(
                        "  std::map<std::string, boost::any> attributes;\n")
                    for a in op.attributes:
                        if not a.isDeprecated():
                            if a.required:
                                if op.name == "Cast":
                                    f.write(
                                        "  // Special case where we cast from DataType to int\n"
                                    )
                                    f.write(
                                        "  attributes[\"{}\"] = static_cast<int>(onnxutil::getTPDataType({}));\n"
                                        .format(a.name, a.name))
                                else:
                                    f.write(
                                        "  attributes[\"{}\"] = {};\n".format(
                                            a.name, a.name))
                            else:
                                if a.isList():
                                    f.write("  if (!{}.empty()) {{\n".format(
                                        a.name))
                                elif a.isFloat() and not a.isBoostOptional():
                                    f.write("  if (std::abs({} - {}) >  std::numeric_limits<{}>::epsilon()) {{\n".format(a.name, a.DefaultValue(), a.CppType()))
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

                    f.write("  return impl->op(Onnx::Operators::{},\n".format(
                        op.CppId()))

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
                    f.write("                  name")

                    if op.verifyInput():
                        f.write(",\n")
                        f.write(
                            "                  [this](std::vector<TensorId> inputs_,\n"
                        )
                        f.write(
                            "                         std::map<std::string, boost::any> attributes_) {\n"
                        )
                        f.write(
                            "                     verify_{}_{}(this->impl, inputs_, attributes_);\n"
                            .format(classname, op.CppId()))
                        f.write("                  }")

                    f.write(")")
                    if op.max_output == 1:
                        f.write("[0]")
                    f.write(";\n")

                    f.write("}\n")
                    f.write("\n")


def genPythonBuilderBinds(filename, schema):

    with io.open(filename, 'w') as f:

        addHeader(f)

        for k, v, in schema.domains.items():

            ops = []

            for opset_version, opset in sorted(v.opsets.items()):

                # Add all ops in the this op set
                for op in opset.operators:

                    # Creeate a list of op with the greatest version number less than the opset version
                    # This is to deal with the opset-6
                    found = [x for x in ops if x.name == op.name]
                    if len(found) > 0:
                        ops.remove(found[0])

                    ops.append(op)

                classname = v.CppName() + "Opset" + opset_version

                # For each opset
                f.write("py::class_<{}>(m, \"{}\")\n".format(
                    classname, classname))
                for op in sorted(ops):

                    # Operator
                    f.write("  .def(\"{}\",\n".format(op.CppName()))

                    # Special case of the constant operator
                    if op.name == "Constant":
                        f.write(
                            "       []({} &opset, py::array array, const std::string& name) {{\n"
                            .format(classname))
                        f.write("          ConstVoidData initData;\n")
                        f.write(
                            "          initData.data = array.request().ptr;\n")
                        f.write(
                            "          initData.info = getTensorInfo(array);\n"
                        )
                        f.write(
                            "          return opset.constant(initData, name);\n"
                        )
                        f.write("       },\n")
                    else:

                        # Add the lamda
                        f.write("       []({} &opset,\n".format(classname))

                        if op.inputs > 0:
                            f.write(
                                "          const std::vector<TensorId> &args,\n"
                            )

                        if op.min_output != op.max_output:
                            f.write("          unsigned num_outputs,\n")

                        for a in sorted(
                                op.attributes,
                                key=lambda x: x.hasDefault() or not x.required
                        ):
                            if not a.isDeprecated():
                                f.write("          {} {},\n".format(
                                    a.CppType(), a.name))

                        f.write("          const std::string &name) -> ")

                        # Call the builder method
                        if op.max_output > 1:
                            f.write("std::vector<TensorId>")
                        else:
                            f.write("TensorId")

                        f.write("{\n")
                        f.write("           return opset.{}(".format(
                            op.CppName()))

                        if op.inputs > 0:
                            f.write("args,\n".format(op.CppName()))

                        if op.min_output != op.max_output:
                            f.write(
                                "                         {}num_outputs,\n".
                                format(spaces(len(op.CppName()))))

                        for a in sorted(
                                op.attributes,
                                key=lambda x: x.hasDefault() or not x.required
                        ):
                            if not a.isDeprecated():
                                f.write(
                                    "                         {}{},\n".format(
                                        spaces(len(op.CppName())), a.name))

                        f.write("                         {}name);\n".format(
                            spaces(len(op.CppName()))))
                        f.write("       },\n")

                    # Define the python function arguments
                    if op.inputs > 0:
                        f.write("       py::arg(\"args\"),\n")

                    if op.min_output != op.max_output:
                        f.write("       py::arg(\"num_outputs\"),\n")

                    for a in sorted(
                            op.attributes,
                            key=lambda x: x.hasDefault() or not x.required):
                        if not a.isDeprecated():
                            f.write("       py::arg(\"{}\")".format(a.name))

                            if a.hasDefaultValue():
                                f.write(" = {}".format(a.DefaultValue()))
                            f.write(",\n")

                    f.write(
                        "       py::arg(\"debugPrefix\") = std::string())\n")

                f.write(";\n")


def genOpIdentifiers(filename, schema):
    with io.open(filename, 'w') as f:
        addHeader(f)
        pass


def main():

    schema = parseDefinitions()

    #genOpIdentifiers('gen/opidentifier_gen.hpp', schema)
    genBuilderHpp('willow/include/poponnx/builder.h.gen', schema)
    genBuilderCpp('willow/src/builder.cpp.gen', schema)
    genPythonBuilderBinds('python/poponnx.cpp.gen', schema)


if __name__ == '__main__':
    main()
