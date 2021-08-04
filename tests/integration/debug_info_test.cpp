// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#define BOOST_TEST_MODULE DebugInfoTest

#include <boost/test/unit_test.hpp>
#include <builderdebuginfo.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/onnxdebuginfo.hpp>
#include <popart/op.hpp>
#include <popart/op/add.hpp>
#include <popart/opidentifier.hpp>
#include <popart/tensordebuginfo.hpp>

#include <iostream>
#include <stdio.h>
#include <poplar/DebugContext.hpp>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <popart/builder.hpp>
#include <popart/vendored/optional.hpp>

#include <onnx/onnx_pb.h>

class TemporaryFileManager {

public:
  TemporaryFileManager(const char *ext) {
    // create temporary file name
    name = std::string(std::tmpnam(nullptr)) + "." + ext;
  }

  std::string name;

  ~TemporaryFileManager() {
    // delete the file
    remove(name.c_str());
  }
};

// Short alias for this namespace
namespace pt = boost::property_tree;

BOOST_AUTO_TEST_CASE(DebugInfo_Test) {

  // Test BuilderDebugInfo
  {
    TemporaryFileManager tfm("json");
    poplar::DebugInfo::initializeStreamer(
        tfm.name, poplar::DebugSerializationFormat::JSON);

    {
      popart::DebugContext dc(
          popart::SourceLocation("function_name", "file_name", 42));
      std::map<std::string, popart::any> args = {
          {"arg1", std::string("test")},
          {"arg2", (int64_t)42},
          {"arg3", std::vector<int64_t>({1, 2, 3})},
          {"arg4", (unsigned)42},
          {"arg5", (float)0.042},
          {"arg6", nonstd::optional<int64_t>(42)},
          {"arg7", nonstd::optional<int64_t>()},
          {"arg8", std::vector<std::string>({"a", "b", "c"})},
          {"arg9", nonstd::optional<float>(1.234)},
          {"arg10", std::vector<float>({0.1, 0.2, 0.3})}};
      popart::BuilderDebugInfo dbi(
          dc, std::string("test_operation"), {"A", "B"}, args, {"C"});
    }

    poplar::DebugInfo::closeStreamer();

    // Create a root
    pt::ptree root;

    // Load the json file in this ptree
    pt::read_json(tfm.name, root);

    BOOST_CHECK(1 == root.get_child("contexts").size());
    BOOST_CHECK("api" == root.get_child("contexts")
                             .front()
                             .second.get_child("category")
                             .get_value<std::string>());
    BOOST_CHECK("popartbuilder" == root.get_child("contexts")
                                       .front()
                                       .second.get_child("layer")
                                       .get_value<std::string>());

    // Expect arg7 to not be included
    BOOST_CHECK(9 == root.get_child("contexts")
                         .front()
                         .second.get_child("attributes")
                         .size());
    BOOST_CHECK("test" == root.get_child("contexts")
                              .front()
                              .second.get_child("attributes")
                              .get_child("arg1")
                              .get_value<std::string>());
    BOOST_CHECK("42" == root.get_child("contexts")
                            .front()
                            .second.get_child("attributes")
                            .get_child("arg2")
                            .get_value<std::string>());
    BOOST_CHECK("[1 2 3]" == root.get_child("contexts")
                                 .front()
                                 .second.get_child("attributes")
                                 .get_child("arg3")
                                 .get_value<std::string>());
    BOOST_CHECK("42" == root.get_child("contexts")
                            .front()
                            .second.get_child("attributes")
                            .get_child("arg4")
                            .get_value<std::string>());
    BOOST_CHECK("0.042000" == root.get_child("contexts")
                                  .front()
                                  .second.get_child("attributes")
                                  .get_child("arg5")
                                  .get_value<std::string>());
    BOOST_CHECK("42" == root.get_child("contexts")
                            .front()
                            .second.get_child("attributes")
                            .get_child("arg6")
                            .get_value<std::string>());
    BOOST_CHECK("[a b c]" == root.get_child("contexts")
                                 .front()
                                 .second.get_child("attributes")
                                 .get_child("arg8")
                                 .get_value<std::string>());
    BOOST_CHECK("1.234" == root.get_child("contexts")
                               .front()
                               .second.get_child("attributes")
                               .get_child("arg9")
                               .get_value<std::string>());
    BOOST_CHECK("[0.1 0.2 0.3]" == root.get_child("contexts")
                                       .front()
                                       .second.get_child("attributes")
                                       .get_child("arg10")
                                       .get_value<std::string>());

    BOOST_CHECK(
        2 ==
        root.get_child("contexts").front().second.get_child("inputs").size());
    BOOST_CHECK(
        1 ==
        root.get_child("contexts").front().second.get_child("outputs").size());

    BOOST_CHECK(42 == root.get_child("contexts")
                          .front()
                          .second.get_child("location.lineNumber")
                          .get_value<int>());
    {
      int fileNameIndex = root.get_child("contexts")
                              .front()
                              .second.get_child("location.fileName")
                              .get_value<int>();
      auto it = root.get_child("stringTable").begin();
      std::advance(it, fileNameIndex);
      BOOST_CHECK("file_name" == (*it).second.get_value<std::string>());
    }
    {
      int functionNameIndex = root.get_child("contexts")
                                  .front()
                                  .second.get_child("location.functionName")
                                  .get_value<int>();
      auto it = root.get_child("stringTable").begin();
      std::advance(it, functionNameIndex);
      BOOST_CHECK("function_name" == (*it).second.get_value<std::string>());
    }
  }

  // Test BuilderVarDebugInfo
  {
    TemporaryFileManager tfm("json");
    poplar::DebugInfo::initializeStreamer(
        tfm.name, poplar::DebugSerializationFormat::JSON);

    {
      popart::DebugContext dc(
          popart::SourceLocation("function_name", "file_name", 42));
      popart::BuilderVarDebugInfo dbi(dc, std::string("test_Var"), "A");
    }

    poplar::DebugInfo::closeStreamer();

    // Create a root
    pt::ptree root;

    // Load the json file in this ptree
    pt::read_json(tfm.name, root);

    BOOST_CHECK(1 == root.get_child("contexts").size());
    BOOST_CHECK("variable" == root.get_child("contexts")
                                  .front()
                                  .second.get_child("category")
                                  .get_value<std::string>());
    BOOST_CHECK("popartbuilder" == root.get_child("contexts")
                                       .front()
                                       .second.get_child("layer")
                                       .get_value<std::string>());

    BOOST_CHECK("A" == root.get_child("contexts")
                           .front()
                           .second.get_child("tensorId")
                           .get_value<std::string>());

    BOOST_CHECK(42 == root.get_child("contexts")
                          .front()
                          .second.get_child("location.lineNumber")
                          .get_value<int>());
    {
      int fileNameIndex = root.get_child("contexts")
                              .front()
                              .second.get_child("location.fileName")
                              .get_value<int>();
      auto it = root.get_child("stringTable").begin();
      std::advance(it, fileNameIndex);
      BOOST_CHECK("file_name" == (*it).second.get_value<std::string>());
    }
    {
      int functionNameIndex = root.get_child("contexts")
                                  .front()
                                  .second.get_child("location.functionName")
                                  .get_value<int>();
      auto it = root.get_child("stringTable").begin();
      std::advance(it, functionNameIndex);
      BOOST_CHECK("function_name" == (*it).second.get_value<std::string>());
    }
  }

  // Test BuilderVarDebugInfo
  {
    TemporaryFileManager tfm("json");
    poplar::DebugInfo::initializeStreamer(
        tfm.name, poplar::DebugSerializationFormat::JSON);

    {
      popart::DebugContext dc(
          popart::SourceLocation("function_name", "file_name", 42));
      popart::BuilderVarDebugInfo dbi(dc,
                                      std::string("test_Var"),
                                      "A",
                                      {popart::DataType::FLOAT, {3, 4, 5}});
    }

    poplar::DebugInfo::closeStreamer();

    // Create a root
    pt::ptree root;

    // Load the json file in this ptree
    pt::read_json(tfm.name, root);

    BOOST_CHECK(1 == root.get_child("contexts").size());
    BOOST_CHECK("variable" == root.get_child("contexts")
                                  .front()
                                  .second.get_child("category")
                                  .get_value<std::string>());
    BOOST_CHECK("popartbuilder" == root.get_child("contexts")
                                       .front()
                                       .second.get_child("layer")
                                       .get_value<std::string>());

    BOOST_CHECK("A" == root.get_child("contexts")
                           .front()
                           .second.get_child("tensorId")
                           .get_value<std::string>());
    BOOST_CHECK("[3 4 5]" == root.get_child("contexts")
                                 .front()
                                 .second.get_child("shape")
                                 .get_value<std::string>());
    BOOST_CHECK("FLOAT" == root.get_child("contexts")
                               .front()
                               .second.get_child("type")
                               .get_value<std::string>());

    BOOST_CHECK(42 == root.get_child("contexts")
                          .front()
                          .second.get_child("location.lineNumber")
                          .get_value<int>());
    {
      int fileNameIndex = root.get_child("contexts")
                              .front()
                              .second.get_child("location.fileName")
                              .get_value<int>();
      auto it = root.get_child("stringTable").begin();
      std::advance(it, fileNameIndex);
      BOOST_CHECK("file_name" == (*it).second.get_value<std::string>());
    }
    {
      int functionNameIndex = root.get_child("contexts")
                                  .front()
                                  .second.get_child("location.functionName")
                                  .get_value<int>();
      auto it = root.get_child("stringTable").begin();
      std::advance(it, functionNameIndex);
      BOOST_CHECK("function_name" == (*it).second.get_value<std::string>());
    }
  }

  // Test OnnxOpDebugInfo
  {
    TemporaryFileManager tfm("json");
    poplar::DebugInfo::initializeStreamer(
        tfm.name, poplar::DebugSerializationFormat::JSON);

    {
      popart::DebugContext dc(
          popart::SourceLocation("function_name", "file_name", 42));

      ONNX_NAMESPACE::NodeProto node;
      node.set_name("tommy");
      node.set_op_type("Add");
      node.set_domain("acme.com");

      node.add_input("A");
      node.add_input("B");
      node.add_output("C");

      auto a1 = node.add_attribute();
      a1->set_type(onnx::AttributeProto::FLOAT);
      a1->set_name("attribute_float");
      a1->set_f(0.0123);

      auto a2 = node.add_attribute();
      a2->set_type(onnx::AttributeProto::INT);
      a2->set_name("attribute_int");
      a2->set_i(42);

      auto a3 = node.add_attribute();
      a3->set_type(onnx::AttributeProto::TENSOR);
      a3->set_name("attribute_tensor");
      auto *t = a3->mutable_t();
      t->set_name("A");
      t->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
      t->add_dims(2);
      t->add_dims(3);

      auto a4 = node.add_attribute();
      a4->set_type(onnx::AttributeProto::GRAPH);
      a4->set_name("attribute_graph");

      auto a5 = node.add_attribute();
      a5->set_type(onnx::AttributeProto::SPARSE_TENSOR);
      a5->set_name("attribute_sparse_tensor");

      auto a6 = node.add_attribute();
      a6->set_type(onnx::AttributeProto::FLOATS);
      a6->set_name("attribute_floats");
      a6->add_floats(0.1);
      a6->add_floats(0.2);

      auto a7 = node.add_attribute();
      a7->set_type(onnx::AttributeProto::INTS);
      a7->set_name("attribute_ints");
      a7->add_ints(1);
      a7->add_ints(2);

      auto a8 = node.add_attribute();
      a8->set_type(onnx::AttributeProto::STRINGS);
      a8->set_name("attribute_strings");
      a7->add_strings("one");
      a7->add_strings("two");

      auto a9 = node.add_attribute();
      a9->set_type(onnx::AttributeProto::TENSORS);
      a9->set_name("attribute_tensors");

      auto a10 = node.add_attribute();
      a10->set_type(onnx::AttributeProto::GRAPHS);
      a10->set_name("attribute_graphs");

      auto a11 = node.add_attribute();
      a11->set_type(onnx::AttributeProto::SPARSE_TENSORS);
      a11->set_name("attribute_sparse_tensors");

      popart::OnnxOpDebugInfo op(dc, node);
    }

    poplar::DebugInfo::closeStreamer();

    // Create a root
    pt::ptree root;

    // Load the json file in this ptree
    pt::read_json(tfm.name, root);

    BOOST_CHECK(1 == root.get_child("contexts").size());

    BOOST_CHECK("op" == root.get_child("contexts")
                            .front()
                            .second.get_child("category")
                            .get_value<std::string>());
    BOOST_CHECK("onnx" == root.get_child("contexts")
                              .front()
                              .second.get_child("layer")
                              .get_value<std::string>());
    BOOST_CHECK("acme.com" == root.get_child("contexts")
                                  .front()
                                  .second.get_child("domain")
                                  .get_value<std::string>());
    BOOST_CHECK("tommy" == root.get_child("contexts")
                               .front()
                               .second.get_child("opName")
                               .get_value<std::string>());
    BOOST_CHECK("Add" == root.get_child("contexts")
                             .front()
                             .second.get_child("opType")
                             .get_value<std::string>());
    BOOST_CHECK(
        2 ==
        root.get_child("contexts").front().second.get_child("input").size());
    BOOST_CHECK(
        1 ==
        root.get_child("contexts").front().second.get_child("output").size());

    BOOST_CHECK("0.0123" == root.get_child("contexts")
                                .front()
                                .second.get_child("attribute.attribute_float")
                                .get_value<std::string>());
    BOOST_CHECK("[0.1 0.2]" ==
                root.get_child("contexts")
                    .front()
                    .second.get_child("attribute.attribute_floats")
                    .get_value<std::string>());
    BOOST_CHECK("42" == root.get_child("contexts")
                            .front()
                            .second.get_child("attribute.attribute_int")
                            .get_value<std::string>());
    BOOST_CHECK("[1 2]" == root.get_child("contexts")
                               .front()
                               .second.get_child("attribute.attribute_ints")
                               .get_value<std::string>());
    BOOST_CHECK("[]" == root.get_child("contexts")
                            .front()
                            .second.get_child("attribute.attribute_strings")
                            .get_value<std::string>());
    BOOST_CHECK("[2 3]" ==
                root.get_child("contexts")
                    .front()
                    .second.get_child("attribute.attribute_tensor.dims")
                    .get_value<std::string>());
    BOOST_CHECK("A" == root.get_child("contexts")
                           .front()
                           .second.get_child("attribute.attribute_tensor.name")
                           .get_value<std::string>());
    BOOST_CHECK("INT32" ==
                root.get_child("contexts")
                    .front()
                    .second.get_child("attribute.attribute_tensor.dataType")
                    .get_value<std::string>());
  }

  // Test OnnxVariableOpDebugInfo
  {
    TemporaryFileManager tfm("json");
    poplar::DebugInfo::initializeStreamer(
        tfm.name, poplar::DebugSerializationFormat::JSON);

    {
      popart::DebugContext dc(
          popart::SourceLocation("function_name", "file_name", 42));

      ONNX_NAMESPACE::TensorProto tensor;
      tensor.set_name("A");
      tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
      tensor.add_dims(2);
      tensor.add_dims(3);
      popart::OnnxVariableDebugInfo op(dc, tensor);
    }

    poplar::DebugInfo::closeStreamer();

    // Create a root
    pt::ptree root;

    // Load the json file in this ptree
    pt::read_json(tfm.name, root);

    BOOST_CHECK(1 == root.get_child("contexts").size());

    BOOST_CHECK("variable" == root.get_child("contexts")
                                  .front()
                                  .second.get_child("category")
                                  .get_value<std::string>());
    BOOST_CHECK("onnx" == root.get_child("contexts")
                              .front()
                              .second.get_child("layer")
                              .get_value<std::string>());

    BOOST_CHECK("[2 3]" == root.get_child("contexts")
                               .front()
                               .second.get_child("tensorProto.dims")
                               .get_value<std::string>());
    BOOST_CHECK("A" == root.get_child("contexts")
                           .front()
                           .second.get_child("tensorProto.name")
                           .get_value<std::string>());
    BOOST_CHECK("INT32" == root.get_child("contexts")
                               .front()
                               .second.get_child("tensorProto.dataType")
                               .get_value<std::string>());
  }

  // Test OnnxVariableOpDebugInfo
  {
    TemporaryFileManager tfm("json");
    poplar::DebugInfo::initializeStreamer(
        tfm.name, poplar::DebugSerializationFormat::JSON);

    {
      popart::DebugContext dc(
          popart::SourceLocation("function_name", "file_name", 42));

      ONNX_NAMESPACE::ValueInfoProto valueInfo;
      valueInfo.set_name("A");
      auto t  = valueInfo.mutable_type();
      auto tt = t->mutable_tensor_type();
      tt->set_elem_type(2);
      auto s = tt->mutable_shape();
      s->add_dim()->set_dim_value(43);
      s->add_dim()->set_dim_value(43);
      popart::OnnxVariableDebugInfo op(dc, valueInfo);
    }

    poplar::DebugInfo::closeStreamer();

    // Create a root
    pt::ptree root;

    // Load the json file in this ptree
    pt::read_json(tfm.name, root);

    BOOST_CHECK(1 == root.get_child("contexts").size());

    BOOST_CHECK("variable" == root.get_child("contexts")
                                  .front()
                                  .second.get_child("category")
                                  .get_value<std::string>());
    BOOST_CHECK("onnx" == root.get_child("contexts")
                              .front()
                              .second.get_child("layer")
                              .get_value<std::string>());

    BOOST_CHECK("A" == root.get_child("contexts")
                           .front()
                           .second.get_child("valueInfoProto.name")
                           .get_value<std::string>());

    BOOST_CHECK("[43 43]" == root.get_child("contexts")
                                 .front()
                                 .second.get_child("valueInfoProto.type.shape")
                                 .get_value<std::string>());
  }

  // Test OpDebugInfo
  {
    TemporaryFileManager tfm("json");
    poplar::DebugInfo::initializeStreamer(
        tfm.name, poplar::DebugSerializationFormat::JSON);

    popart::OpId op_id = -1;

    {
      popart::DebugContext dc(
          popart::SourceLocation("function_name", "file_name", 42));

      popart::Ir ir;
      popart::GraphId graphid("toplevel");
      popart::Graph graph(ir, graphid);
      popart::Op::Settings settings(graph, "my_add");
      popart::AddOp op(popart::Onnx::Operators::Add_6, settings);

      popart::Tensor t1("InputA", popart::TensorType::Stream, graph);
      op.input->insert(0, &t1);
      popart::Tensor t2("InputB", popart::TensorType::Stream, graph);
      op.input->insert(1, &t2);
      popart::Tensor t3("OutputC", popart::TensorType::Stream, graph);
      op.output->insert(0, &t3);

      op_id = op.id;
      op.finalizeDebugInfo();
    }

    poplar::DebugInfo::closeStreamer();

    // Create a root
    pt::ptree root;

    // Load the json file in this ptree
    pt::read_json(tfm.name, root);

    // 3 tensors & 1 op
    BOOST_CHECK(4 == root.get_child("contexts").size());
    BOOST_CHECK("op" == root.get_child("contexts")
                            .back()
                            .second.get_child("category")
                            .get_value<std::string>());
    BOOST_CHECK("popart" == root.get_child("contexts")
                                .back()
                                .second.get_child("layer")
                                .get_value<std::string>());

    BOOST_CHECK("toplevel" == root.get_child("contexts")
                                  .back()
                                  .second.get_child("graphId")
                                  .get_value<std::string>());

    BOOST_CHECK(op_id == root.get_child("contexts")
                             .back()
                             .second.get_child("instanceId")
                             .get_value<int>());
    BOOST_CHECK("ai.onnx.Add:6" == root.get_child("contexts")
                                       .back()
                                       .second.get_child("opid")
                                       .get_value<std::string>());
    BOOST_CHECK(
        2 ==
        root.get_child("contexts").back().second.get_child("inputs").size());
    BOOST_CHECK(
        1 ==
        root.get_child("contexts").back().second.get_child("outputs").size());
    BOOST_CHECK("NO" == root.get_child("contexts")
                            .back()
                            .second.get_child("attributes.recompute")
                            .get_value<std::string>());

    // File name & function will be from within popart
  }

  // Test TensorDebugInfo
  {
    TemporaryFileManager tfm("json");
    poplar::DebugInfo::initializeStreamer(
        tfm.name, poplar::DebugSerializationFormat::JSON);

    {
      popart::DebugContext dc(
          popart::SourceLocation("function_name", "file_name", 42));
      popart::TensorInfo info("FLOAT", "(2,2)");
      popart::TensorType type = popart::TensorType::Stream;
      popart::TensorDebugInfo dbi(dc, "TensorA", info, type);
    }

    poplar::DebugInfo::closeStreamer();

    // Create a root
    pt::ptree root;

    // Load the json file in this ptree
    pt::read_json(tfm.name, root);

    BOOST_CHECK(1 == root.get_child("contexts").size());
    BOOST_CHECK("tensor" == root.get_child("contexts")
                                .front()
                                .second.get_child("category")
                                .get_value<std::string>());
    BOOST_CHECK("popart" == root.get_child("contexts")
                                .front()
                                .second.get_child("layer")
                                .get_value<std::string>());

    BOOST_CHECK("TensorA" == root.get_child("contexts")
                                 .front()
                                 .second.get_child("tensorId")
                                 .get_value<std::string>());

    BOOST_CHECK("[2 2]" == root.get_child("contexts")
                               .front()
                               .second.get_child("shape")
                               .get_value<std::string>());

    BOOST_CHECK("float32" == root.get_child("contexts")
                                 .front()
                                 .second.get_child("elementType")
                                 .get_value<std::string>());

    BOOST_CHECK("Stream" == root.get_child("contexts")
                                .front()
                                .second.get_child("type")
                                .get_value<std::string>());
    {
      int fileNameIndex = root.get_child("contexts")
                              .front()
                              .second.get_child("location.fileName")
                              .get_value<int>();
      auto it = root.get_child("stringTable").begin();
      std::advance(it, fileNameIndex);
      BOOST_CHECK("file_name" == (*it).second.get_value<std::string>());
    }
    {
      int functionNameIndex = root.get_child("contexts")
                                  .front()
                                  .second.get_child("location.functionName")
                                  .get_value<int>();
      auto it = root.get_child("stringTable").begin();
      std::advance(it, functionNameIndex);
      BOOST_CHECK("function_name" == (*it).second.get_value<std::string>());
    }
  }

  // Test TensorDebugInfo
  {
    TemporaryFileManager tfm("json");
    poplar::DebugInfo::initializeStreamer(
        tfm.name, poplar::DebugSerializationFormat::JSON);

    {
      popart::DebugContext dc(
          popart::SourceLocation("function_name", "file_name", 42));
      popart::TensorType type = popart::TensorType::Stream;
      popart::TensorDebugInfo dbi(dc, "TensorA", type);
    }

    poplar::DebugInfo::closeStreamer();

    // Create a root
    pt::ptree root;

    // Load the json file in this ptree
    pt::read_json(tfm.name, root);

    BOOST_CHECK(1 == root.get_child("contexts").size());
    BOOST_CHECK("tensor" == root.get_child("contexts")
                                .front()
                                .second.get_child("category")
                                .get_value<std::string>());
    BOOST_CHECK("popart" == root.get_child("contexts")
                                .front()
                                .second.get_child("layer")
                                .get_value<std::string>());

    BOOST_CHECK("TensorA" == root.get_child("contexts")
                                 .front()
                                 .second.get_child("tensorId")
                                 .get_value<std::string>());

    BOOST_CHECK("Stream" == root.get_child("contexts")
                                .front()
                                .second.get_child("type")
                                .get_value<std::string>());
    {
      int fileNameIndex = root.get_child("contexts")
                              .front()
                              .second.get_child("location.fileName")
                              .get_value<int>();
      auto it = root.get_child("stringTable").begin();
      std::advance(it, fileNameIndex);
      BOOST_CHECK("file_name" == (*it).second.get_value<std::string>());
    }
    {
      int functionNameIndex = root.get_child("contexts")
                                  .front()
                                  .second.get_child("location.functionName")
                                  .get_value<int>();
      auto it = root.get_child("stringTable").begin();
      std::advance(it, functionNameIndex);
      BOOST_CHECK("function_name" == (*it).second.get_value<std::string>());
    }
  }
}