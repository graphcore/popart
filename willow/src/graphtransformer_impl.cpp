#include <popart/graphtransformer_impl.hpp>
#include <popart/onnxutil.hpp>
#include <popart/opidentifier.hpp>

// used for float to half conversion
#include <poplar/Target.hpp>

#include <onnx/checker.h>

namespace popart {

GraphTransformerImpl::GraphTransformerImpl(
    const std::string &modelProtoOrFilename) {
  model = onnxutil::getModelProto(modelProtoOrFilename);

  // Check imported model is valid.
  onnx::checker::check_model(model);
}

std::string GraphTransformerImpl::getModelProto() const {
  std::string output;
  model.SerializeToString(&output);
  return output;
}

void GraphTransformerImpl::convertFloatsToHalfs() {
  onnxutil::visitModelNodes(model, [](onnx::NodeProto &node) {
    for (unsigned att_i = 0; att_i < node.attribute_size(); ++att_i) {
      auto ptr_att              = node.mutable_attribute(att_i);
      onnx::AttributeProto &att = *ptr_att;

      if (!att.has_type()) {
        throw error("attribute has no type");
      }

      auto type = att.type();
      if (type == onnx::AttributeProto_AttributeType_TENSOR) {
        if (!att.t().has_data_type()) {
          throw error("Typeless tensor in convertFloatToHalf");
        }
        if (att.t().data_type() == onnx::TensorProto_DataType_UNDEFINED) {
          throw error("undefined tensor proto data type");
        }
        if (att.t().data_type() == onnx::TensorProto_DataType_FLOAT) {
          auto &tensor = *att.mutable_t();
          convertFloatTensorToHalf(tensor);
        }
      } else if (type == onnx::AttributeProto_AttributeType_GRAPH ||
                 type == onnx::AttributeProto_AttributeType_GRAPHS) {
        throw error("Attributes of type GRAPH and GRAPHS : need impl in "
                    "convertFloatToHalf");
      }
    }
  });

  onnxutil::visitModelInitializers(model, [](onnx::TensorProto &initializer) {
    if (initializer.data_type() == onnx::TensorProto_DataType_FLOAT) {
      convertFloatTensorToHalf(initializer);
    }
  });

  onnxutil::visitModelValueInfos(model, [](onnx::ValueInfoProto &value_info) {
    if (value_info.has_type()) {
      auto ptype = value_info.mutable_type();
      if (ptype->has_tensor_type()) {
        auto p_tensor_type = ptype->mutable_tensor_type();
        if (p_tensor_type->has_elem_type()) {
          auto elem_type = p_tensor_type->elem_type();
          if (elem_type == onnx::TensorProto_DataType_FLOAT) {
            p_tensor_type->set_elem_type(onnx::TensorProto_DataType_FLOAT16);
          }
        }
      }
    }
  });
}

void GraphTransformerImpl::convertFloatTensorToHalf(onnx::TensorProto &tp) {
  if (tp.data_type() != onnx::TensorProto_DataType_FLOAT) {
    auto descriptor     = onnx::TensorProto_DataType_descriptor();
    auto data_type_name = descriptor->FindValueByNumber(tp.data_type())->name();
    throw error("cannot set tensor type {} to type HALF", data_type_name);
  }
  auto mutableData = onnxutil::getMutableData(tp);
  auto floatData   = reinterpret_cast<const float *>(mutableData.data);

  auto n_elms = mutableData.info.nelms();
  std::vector<char> hValData(2 * n_elms);
  // poplar::copyFloatToDeviceHalf takes a Target as an argument, but doesn't
  // use it, so a dummy target can be used here.
  auto dummyTarget = poplar::Target();
  poplar::copyFloatToDeviceHalf(
      dummyTarget, floatData, hValData.data(), n_elms);

  tp.clear_float_data();
  tp.clear_raw_data();
  tp.set_raw_data(hValData.data(), hValData.size());
  tp.set_data_type(onnx::TensorProto_DataType_FLOAT16);

  tp.clear_float_data();
  tp.clear_raw_data();
  tp.set_raw_data(hValData.data(), hValData.size());
  tp.set_data_type(onnx::TensorProto_DataType_FLOAT16);
}

void GraphTransformerImpl::convertInitializersToConstants(
    const std::vector<TensorId> &ids) {
  auto *graph = model.mutable_graph();

  std::set<TensorId> initializer_names;
  for (auto &initializer : graph->initializer()) {
    initializer_names.insert(initializer.name());
  }

  for (auto &id : ids) {
    if (initializer_names.count(id) == 0) {
      throw error("TensorId {} not in the model initalizers", id);
    }
  }

  // The constants need to be before any consumers, so make a new list and then
  // append the existing list to it.
  google::protobuf::RepeatedPtrField<onnx::NodeProto> new_nodes;

  // First add in constants
  for (auto &id : ids) {
    auto *initializers = graph->mutable_initializer();
    for (auto initializer = initializers->begin();
         initializer != initializers->end();
         ++initializer) {
      if (initializer->name() == id) {
        auto *node = new_nodes.Add();
        node->set_name(id);
        node->set_op_type(Onnx::Operators::Constant_9.type);
        node->set_domain("");
        node->add_output(id);

        auto *attr = node->add_attribute();
        attr->set_name("value");
        attr->set_type(onnx::AttributeProto::TENSOR);
        auto *t = attr->mutable_t();
        *t      = *initializer;
        break;
      }
    }
  }

  // Append the previous nodes
  new_nodes.MergeFrom(graph->node());
  graph->mutable_node()->Swap(&new_nodes);

  // Now remove the initializers and inputs
  // TODO unexpectedly large models might benefit from
  // making this O(n^2) algorithm into the O(n) version
  // task T6416
  for (auto &id : ids) {
    auto *initializers = graph->mutable_initializer();
    for (auto initializer = initializers->begin();
         initializer != initializers->end();
         ++initializer) {
      if (initializer->name() == id) {
        initializers->erase(initializer);
        break;
      }
    }

    auto *inputs = graph->mutable_input();
    for (auto input = inputs->begin(); input != inputs->end(); ++input) {
      if (input->name() == id) {
        inputs->erase(input);
        break;
      }
    }
  }

  onnx::checker::check_model(model);
  return;
}

void GraphTransformerImpl::convertAllFixedPointInitializersToConstants() {
  auto graph = model.graph();
  std::vector<TensorId> to_const;
  for (auto &initializer : graph.initializer()) {
    if (getDataTypeInfoMap()
            .at(onnxutil::getDataType(initializer.data_type()))
            .isFixedPoint()) {
      to_const.push_back(initializer.name());
    }
  }
  convertInitializersToConstants(to_const);
}

void GraphTransformerImpl::removeUnusedInputs() {

  auto *graph = model.mutable_graph();

  // walk the nodes, gather names of all inputs to nodes
  std::set<TensorId> consumed;
  for (auto &node : graph->node()) {
    for (auto &i : node.input()) {
      consumed.emplace(i);
    }
  }

  // before doing expensive copying, we first check
  // that there are any unused inputs,
  bool isUnusedInput = false;
  for (auto &gi : graph->input()) {
    if (consumed.find(gi.name()) == consumed.end()) {
      isUnusedInput = true;
      break;
    }
  }

  // if there are no unused inputs, return
  if (!isUnusedInput) {
    return;
  }

  // store all the inital inputs and initializers, in
  // preparation for clearing and repopulating the graph
  std::vector<onnx::ValueInfoProto> initialInputs;
  for (auto &x : graph->input()) {
    initialInputs.push_back(x);
  }

  std::map<TensorId, onnx::TensorProto> initialInitializers;
  for (auto &x : graph->initializer()) {
    initialInitializers[x.name()] = x;
  }

  graph->clear_input();
  graph->clear_initializer();
  for (auto &iniIn : initialInputs) {
    TensorId id = iniIn.name();
    if (consumed.find(id) != consumed.end()) {
      onnx::ValueInfoProto *vip = graph->add_input();
      *vip                      = iniIn;

      auto found = initialInitializers.find(id);
      if (found != initialInitializers.end()) {
        onnx::TensorProto *tp = graph->add_initializer();
        *tp                   = found->second;
      }
    }
  }
}

void GraphTransformerImpl::prepareNodesForTraining() {
  static int counter = 0;
  auto *graph        = model.mutable_graph();
  for (int node_i = 0; node_i < graph->node_size(); ++node_i) {
    if (graph->node(node_i).op_type() == "BatchNormalization") {
      onnx::NodeProto *mNode = graph->mutable_node(node_i);
      for (int outIndex = graph->node(node_i).output_size(); outIndex < 5;
           ++outIndex) {
        std::stringstream tensorName;
        tensorName << mNode->op_type() << "_" << mNode->name() << "_oi"
                   << outIndex << "_" << counter;
        ++counter;
        mNode->add_output(tensorName.str());
      }
    }
  }
}

} // namespace popart
