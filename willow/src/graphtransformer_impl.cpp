// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graphtransformer_impl.hpp>
#include <popart/onnxutil.hpp>
#include <popart/opidentifier.hpp>

// used for float to half conversion
#include <poplar/Target.hpp>

#include <onnx/checker.h>

#include <popart/logging.hpp>

namespace popart {

GraphTransformerImpl::GraphTransformerImpl(
    const std::string &modelProtoOrFilename) {
  model = onnxutil::getModelProto(modelProtoOrFilename);

  // Check imported model is valid.
  ONNX_NAMESPACE::checker::check_model(model);
}

std::string GraphTransformerImpl::getModelProto() const {
  std::string output;
  model.SerializeToString(&output);
  return output;
}

void GraphTransformerImpl::convertFloatsToHalfs() {
  onnxutil::visitModelNodes(model, [](ONNX_NAMESPACE::NodeProto &node) {
    for (unsigned att_i = 0; att_i < node.attribute_size(); ++att_i) {
      auto ptr_att                        = node.mutable_attribute(att_i);
      ONNX_NAMESPACE::AttributeProto &att = *ptr_att;

      if (!att.has_type()) {
        throw error("attribute has no type");
      }

      auto type = att.type();
      if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
        if (!att.t().has_data_type()) {
          throw error("Typeless tensor in convertFloatToHalf");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
          throw error("undefined tensor proto data type");
        }
        if (att.t().data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          auto &tensor = *att.mutable_t();
          convertFloatTensorToHalf(tensor);
        }
      } else if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH ||
                 type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS) {
        throw error("Attributes of type GRAPH and GRAPHS : need impl in "
                    "convertFloatToHalf");
      } else {
        // not doing anything in this case, as this Transform is specific to
        // type
      }
    }
  });

  onnxutil::visitModelInitializers(
      model, [](ONNX_NAMESPACE::TensorProto &initializer) {
        if (initializer.data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
          convertFloatTensorToHalf(initializer);
        }
      });

  onnxutil::visitModelValueInfos(
      model, [](ONNX_NAMESPACE::ValueInfoProto &value_info) {
        if (value_info.has_type()) {
          auto ptype = value_info.mutable_type();
          if (ptype->has_tensor_type()) {
            auto p_tensor_type = ptype->mutable_tensor_type();
            if (p_tensor_type->has_elem_type()) {
              auto elem_type = p_tensor_type->elem_type();
              if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
                p_tensor_type->set_elem_type(
                    ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
              }
            }
          }
        }
      });
}

void GraphTransformerImpl::convertUINT8ToINT32() {
  onnxutil::visitModelNodes(model, [](ONNX_NAMESPACE::NodeProto &node) {
    for (unsigned att_i = 0; att_i < node.attribute_size(); ++att_i) {
      auto ptr_att                        = node.mutable_attribute(att_i);
      ONNX_NAMESPACE::AttributeProto &att = *ptr_att;

      if (!att.has_type()) {
        throw error("attribute has no type");
      }

      auto type = att.type();
      if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
        if (!att.t().has_data_type()) {
          throw error("Typeless tensor in convertUINT8ToINT32");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
          throw error("undefined tensor proto data type");
        }
        if (att.t().data_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
          auto &tensor = *att.mutable_t();
          convertUINT8TensorToINT32(tensor);
        }
      } else if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH ||
                 type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS) {
        throw error("Attributes of type GRAPH and GRAPHS : need impl in "
                    "convertUINT8ToINT32");
      } else {
        // not doing anything in this case, as this Transform is specific to
        // type
      }
    }
  });

  onnxutil::visitModelInitializers(
      model, [](ONNX_NAMESPACE::TensorProto &initializer) {
        if (initializer.data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
          convertUINT8TensorToINT32(initializer);
        }
      });

  onnxutil::visitModelValueInfos(
      model, [](ONNX_NAMESPACE::ValueInfoProto &value_info) {
        if (value_info.has_type()) {
          auto ptype = value_info.mutable_type();
          if (ptype->has_tensor_type()) {
            auto p_tensor_type = ptype->mutable_tensor_type();
            if (p_tensor_type->has_elem_type()) {
              auto elem_type = p_tensor_type->elem_type();
              if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
                p_tensor_type->set_elem_type(
                    ONNX_NAMESPACE::TensorProto_DataType_INT32);
              }
            }
          }
        }
      });
}

void GraphTransformerImpl::convertUINT16ToINT32() {
  onnxutil::visitModelNodes(model, [](ONNX_NAMESPACE::NodeProto &node) {
    for (unsigned att_i = 0; att_i < node.attribute_size(); ++att_i) {
      auto ptr_att                        = node.mutable_attribute(att_i);
      ONNX_NAMESPACE::AttributeProto &att = *ptr_att;

      if (!att.has_type()) {
        throw error("attribute has no type");
      }

      auto type = att.type();
      if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
        if (!att.t().has_data_type()) {
          throw error("Typeless tensor in convertUINT16ToINT32");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
          throw error("undefined tensor proto data type");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
          auto &tensor = *att.mutable_t();
          convertUINT16TensorToINT32(tensor);
        }
      } else if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH ||
                 type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS) {
        throw error("Attributes of type GRAPH and GRAPHS : need impl in "
                    "convertUINT16ToINT32");
      } else {
        // not doing anything in this case, as this Transform is specific to
        // type
      }
    }
  });

  onnxutil::visitModelInitializers(
      model, [](ONNX_NAMESPACE::TensorProto &initializer) {
        if (initializer.data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
          convertUINT16TensorToINT32(initializer);
        }
      });

  onnxutil::visitModelValueInfos(
      model, [](ONNX_NAMESPACE::ValueInfoProto &value_info) {
        if (value_info.has_type()) {
          auto ptype = value_info.mutable_type();
          if (ptype->has_tensor_type()) {
            auto p_tensor_type = ptype->mutable_tensor_type();
            if (p_tensor_type->has_elem_type()) {
              auto elem_type = p_tensor_type->elem_type();
              if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
                p_tensor_type->set_elem_type(
                    ONNX_NAMESPACE::TensorProto_DataType_INT32);
              }
            }
          }
        }
      });
}

void GraphTransformerImpl::convertINT8ToINT32() {
  onnxutil::visitModelNodes(model, [](ONNX_NAMESPACE::NodeProto &node) {
    for (unsigned att_i = 0; att_i < node.attribute_size(); ++att_i) {
      auto ptr_att                        = node.mutable_attribute(att_i);
      ONNX_NAMESPACE::AttributeProto &att = *ptr_att;

      if (!att.has_type()) {
        throw error("attribute has no type");
      }

      auto type = att.type();
      if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
        if (!att.t().has_data_type()) {
          throw error("Typeless tensor in convertINT8ToINT32");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
          throw error("undefined tensor proto data type");
        }
        if (att.t().data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
          auto &tensor = *att.mutable_t();
          convertINT8TensorToINT32(tensor);
        }
      } else if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH ||
                 type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS) {
        throw error("Attributes of type GRAPH and GRAPHS : need impl in "
                    "convertINT8ToINT32");
      } else {
        // not doing anything in this case, as this Transform is specific to
        // type
      }
    }
  });

  onnxutil::visitModelInitializers(
      model, [](ONNX_NAMESPACE::TensorProto &initializer) {
        if (initializer.data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_INT8) {
          convertINT8TensorToINT32(initializer);
        }
      });

  onnxutil::visitModelValueInfos(
      model, [](ONNX_NAMESPACE::ValueInfoProto &value_info) {
        if (value_info.has_type()) {
          auto ptype = value_info.mutable_type();
          if (ptype->has_tensor_type()) {
            auto p_tensor_type = ptype->mutable_tensor_type();
            if (p_tensor_type->has_elem_type()) {
              auto elem_type = p_tensor_type->elem_type();
              if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
                p_tensor_type->set_elem_type(
                    ONNX_NAMESPACE::TensorProto_DataType_INT32);
              }
            }
          }
        }
      });
}

void GraphTransformerImpl::convertINT16ToINT32() {
  onnxutil::visitModelNodes(model, [](ONNX_NAMESPACE::NodeProto &node) {
    for (unsigned att_i = 0; att_i < node.attribute_size(); ++att_i) {
      auto ptr_att                        = node.mutable_attribute(att_i);
      ONNX_NAMESPACE::AttributeProto &att = *ptr_att;

      if (!att.has_type()) {
        throw error("attribute has no type");
      }

      auto type = att.type();
      if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
        if (!att.t().has_data_type()) {
          throw error("Typeless tensor in convertINT16ToINT32");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
          throw error("undefined tensor proto data type");
        }
        if (att.t().data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT16) {
          auto &tensor = *att.mutable_t();
          convertINT16TensorToINT32(tensor);
        }
      } else if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH ||
                 type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS) {
        throw error("Attributes of type GRAPH and GRAPHS : need impl in "
                    "convertINT16ToINT32");
      } else {
        // not doing anything in this case, as this Transform is specific to
        // type
      }
    }
  });

  onnxutil::visitModelInitializers(
      model, [](ONNX_NAMESPACE::TensorProto &initializer) {
        if (initializer.data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_INT16) {
          convertINT16TensorToINT32(initializer);
        }
      });

  onnxutil::visitModelValueInfos(
      model, [](ONNX_NAMESPACE::ValueInfoProto &value_info) {
        if (value_info.has_type()) {
          auto ptype = value_info.mutable_type();
          if (ptype->has_tensor_type()) {
            auto p_tensor_type = ptype->mutable_tensor_type();
            if (p_tensor_type->has_elem_type()) {
              auto elem_type = p_tensor_type->elem_type();
              if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT16) {
                p_tensor_type->set_elem_type(
                    ONNX_NAMESPACE::TensorProto_DataType_INT32);
              }
            }
          }
        }
      });
}

void GraphTransformerImpl::convertINT64ToINT32() {
  onnxutil::visitModelNodes(model, [](ONNX_NAMESPACE::NodeProto &node) {
    for (unsigned att_i = 0; att_i < node.attribute_size(); ++att_i) {
      auto ptr_att                        = node.mutable_attribute(att_i);
      ONNX_NAMESPACE::AttributeProto &att = *ptr_att;

      if (!att.has_type()) {
        throw error("attribute has no type");
      }

      auto type = att.type();
      if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
        if (!att.t().has_data_type()) {
          throw error("Typeless tensor in convertINT64ToINT32");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
          throw error("undefined tensor proto data type");
        }
        if (att.t().data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
          auto &tensor = *att.mutable_t();
          convertINT64TensorToINT32(tensor);
        }
      } else if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH ||
                 type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS) {
        throw error("Attributes of type GRAPH and GRAPHS : need impl in "
                    "convertINT64ToINT32");
      } else {
        // not doing anything in this case, as this Transform is specific to
        // type
      }
    }
  });

  onnxutil::visitModelInitializers(
      model, [](ONNX_NAMESPACE::TensorProto &initializer) {
        if (initializer.data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_INT64) {
          convertINT64TensorToINT32(initializer);
        }
      });

  onnxutil::visitModelValueInfos(
      model, [](ONNX_NAMESPACE::ValueInfoProto &value_info) {
        if (value_info.has_type()) {
          auto ptype = value_info.mutable_type();
          if (ptype->has_tensor_type()) {
            auto p_tensor_type = ptype->mutable_tensor_type();
            if (p_tensor_type->has_elem_type()) {
              auto elem_type = p_tensor_type->elem_type();
              if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
                p_tensor_type->set_elem_type(
                    ONNX_NAMESPACE::TensorProto_DataType_INT32);
              }
            }
          }
        }
      });
}

void GraphTransformerImpl::convertDoublesToFloats() {
  onnxutil::visitModelNodes(model, [](ONNX_NAMESPACE::NodeProto &node) {
    for (unsigned att_i = 0; att_i < node.attribute_size(); ++att_i) {
      auto ptr_att                        = node.mutable_attribute(att_i);
      ONNX_NAMESPACE::AttributeProto &att = *ptr_att;

      if (!att.has_type()) {
        throw error("attribute has no type");
      }

      auto type = att.type();
      if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
        if (!att.t().has_data_type()) {
          throw error("Typeless tensor in convertDoublesToFloats");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
          throw error("undefined tensor proto data type");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
          auto &tensor = *att.mutable_t();
          convertDoubleTensorToFloat(tensor);
        }
      } else if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH ||
                 type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS) {
        throw error("Attributes of type GRAPH and GRAPHS : need impl in "
                    "convertDoublesToFloats");
      } else {
        // not doing anything in this case, as this Transform is specific to
        // type
      }
    }
  });

  onnxutil::visitModelInitializers(
      model, [](ONNX_NAMESPACE::TensorProto &initializer) {
        if (initializer.data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
          convertDoubleTensorToFloat(initializer);
        }
      });

  onnxutil::visitModelValueInfos(
      model, [](ONNX_NAMESPACE::ValueInfoProto &value_info) {
        if (value_info.has_type()) {
          auto ptype = value_info.mutable_type();
          if (ptype->has_tensor_type()) {
            auto p_tensor_type = ptype->mutable_tensor_type();
            if (p_tensor_type->has_elem_type()) {
              auto elem_type = p_tensor_type->elem_type();
              if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
                p_tensor_type->set_elem_type(
                    ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
              }
            }
          }
        }
      });
}

void GraphTransformerImpl::convertBFloats16ToFloat32() {
  onnxutil::visitModelNodes(model, [](ONNX_NAMESPACE::NodeProto &node) {
    for (unsigned att_i = 0; att_i < node.attribute_size(); ++att_i) {
      auto ptr_att                        = node.mutable_attribute(att_i);
      ONNX_NAMESPACE::AttributeProto &att = *ptr_att;

      if (!att.has_type()) {
        throw error("attribute has no type");
      }

      auto type = att.type();
      if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
        if (!att.t().has_data_type()) {
          throw error("Typeless tensor in convertBFloats16ToFloat32");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
          throw error("undefined tensor proto data type");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
          auto &tensor = *att.mutable_t();
          convertBFloat16TensorToFloat32(tensor);
        }
      } else if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH ||
                 type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS) {
        throw error("Attributes of type GRAPH and GRAPHS : need impl in "
                    "convertBFloats16ToFloat32");
      } else {
        // not doing anything in this case, as this Transform is specific to
        // type
      }
    }
  });

  onnxutil::visitModelInitializers(
      model, [](ONNX_NAMESPACE::TensorProto &initializer) {
        if (initializer.data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
          convertBFloat16TensorToFloat32(initializer);
        }
      });

  onnxutil::visitModelValueInfos(
      model, [](ONNX_NAMESPACE::ValueInfoProto &value_info) {
        if (value_info.has_type()) {
          auto ptype = value_info.mutable_type();
          if (ptype->has_tensor_type()) {
            auto p_tensor_type = ptype->mutable_tensor_type();
            if (p_tensor_type->has_elem_type()) {
              auto elem_type = p_tensor_type->elem_type();
              if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
                p_tensor_type->set_elem_type(
                    ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
              }
            }
          }
        }
      });
}

void GraphTransformerImpl::convertDoublesToHalfs() {
  onnxutil::visitModelNodes(model, [](ONNX_NAMESPACE::NodeProto &node) {
    for (unsigned att_i = 0; att_i < node.attribute_size(); ++att_i) {
      auto ptr_att                        = node.mutable_attribute(att_i);
      ONNX_NAMESPACE::AttributeProto &att = *ptr_att;

      if (!att.has_type()) {
        throw error("attribute has no type");
      }

      auto type = att.type();
      if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
        if (!att.t().has_data_type()) {
          throw error("Typeless tensor in convertDoublesToHalfs");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED) {
          throw error("undefined tensor proto data type");
        }
        if (att.t().data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
          auto &tensor = *att.mutable_t();
          convertDoubleTensorToHalf(tensor);
        }
      } else if (type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH ||
                 type == ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS) {
        throw error("Attributes of type GRAPH and GRAPHS : need impl in "
                    "convertDoublesToHalfs");
      } else {
        // not doing anything in this case, as this Transform is specific to
        // type
      }
    }
  });

  onnxutil::visitModelInitializers(
      model, [](ONNX_NAMESPACE::TensorProto &initializer) {
        if (initializer.data_type() ==
            ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
          convertDoubleTensorToHalf(initializer);
        }
      });

  onnxutil::visitModelValueInfos(
      model, [](ONNX_NAMESPACE::ValueInfoProto &value_info) {
        if (value_info.has_type()) {
          auto ptype = value_info.mutable_type();
          if (ptype->has_tensor_type()) {
            auto p_tensor_type = ptype->mutable_tensor_type();
            if (p_tensor_type->has_elem_type()) {
              auto elem_type = p_tensor_type->elem_type();
              if (elem_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
                p_tensor_type->set_elem_type(
                    ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
              }
            }
          }
        }
      });
}

void GraphTransformerImpl::convertFloatTensorToHalf(
    ONNX_NAMESPACE::TensorProto &tp) {
  if (tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    auto descriptor     = ONNX_NAMESPACE::TensorProto_DataType_descriptor();
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
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
}

void GraphTransformerImpl::convertDoubleTensorToHalf(
    ONNX_NAMESPACE::TensorProto &tp) {
  if (tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
    auto descriptor     = ONNX_NAMESPACE::TensorProto_DataType_descriptor();
    auto data_type_name = descriptor->FindValueByNumber(tp.data_type())->name();
    throw error("cannot set tensor type {} to type HALF", data_type_name);
  }
  auto mutableData = onnxutil::getMutableData(tp);
  auto doubleData  = reinterpret_cast<const double *>(mutableData.data);

  auto n_elms = mutableData.info.nelms();
  std::vector<char> hValData(2 * n_elms);
  // poplar::copyDoubleToDeviceHalf takes a Target as an argument, but doesn't
  // use it, so a dummy target can be used here.
  auto dummyTarget = poplar::Target();
  poplar::copyDoubleToDeviceHalf(
      dummyTarget, doubleData, hValData.data(), n_elms);

  tp.clear_double_data();
  tp.clear_raw_data();
  tp.set_raw_data(hValData.data(), hValData.size());
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
}

void GraphTransformerImpl::convertDoubleTensorToFloat(
    ONNX_NAMESPACE::TensorProto &tp) {
  if (tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
    auto descriptor     = ONNX_NAMESPACE::TensorProto_DataType_descriptor();
    auto data_type_name = descriptor->FindValueByNumber(tp.data_type())->name();
    throw error("cannot set tensor type {} to type Float", data_type_name);
  }
  auto mutableData = onnxutil::getMutableData(tp);
  auto doubleData  = reinterpret_cast<const double *>(mutableData.data);

  auto n_elms = mutableData.info.nelms();

  for (int i = 0; i < n_elms; i++)
    tp.add_float_data(static_cast<float>(doubleData[i]));
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  // Don't clear while we're using it.
  tp.clear_double_data();
  tp.clear_raw_data();
}

void GraphTransformerImpl::convertUINT8TensorToINT32(
    ONNX_NAMESPACE::TensorProto &tp) {
  if (tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    auto descriptor     = ONNX_NAMESPACE::TensorProto_DataType_descriptor();
    auto data_type_name = descriptor->FindValueByNumber(tp.data_type())->name();
    throw error("cannot set tensor type {} to type INT32", data_type_name);
  }
  auto mutableData = onnxutil::getMutableData(tp);
  auto uint8Data   = reinterpret_cast<const uint8_t *>(mutableData.data);

  auto n_elms = mutableData.info.nelms();

  for (int i = 0; i < n_elms; i++)
    tp.add_int32_data(static_cast<int32_t>(uint8Data[i]));
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);

  // Don't clear while we're using it.
  tp.clear_raw_data();
}

void GraphTransformerImpl::convertUINT16TensorToINT32(
    ONNX_NAMESPACE::TensorProto &tp) {
  if (tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT16) {
    auto descriptor     = ONNX_NAMESPACE::TensorProto_DataType_descriptor();
    auto data_type_name = descriptor->FindValueByNumber(tp.data_type())->name();
    throw error("cannot set tensor type {} to type INT32", data_type_name);
  }
  auto mutableData = onnxutil::getMutableData(tp);
  auto uint16Data  = reinterpret_cast<const uint16_t *>(mutableData.data);

  auto n_elms = mutableData.info.nelms();

  for (int i = 0; i < n_elms; i++)
    tp.add_int32_data(static_cast<int32_t>(uint16Data[i]));
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);

  // Don't clear while we're using it.
  tp.clear_raw_data();
}

void GraphTransformerImpl::convertINT8TensorToINT32(
    ONNX_NAMESPACE::TensorProto &tp) {
  if (tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    auto descriptor     = ONNX_NAMESPACE::TensorProto_DataType_descriptor();
    auto data_type_name = descriptor->FindValueByNumber(tp.data_type())->name();
    throw error("cannot set tensor type {} to type INT32", data_type_name);
  }
  auto mutableData = onnxutil::getMutableData(tp);
  auto int8Data    = reinterpret_cast<const int8_t *>(mutableData.data);

  auto n_elms = mutableData.info.nelms();

  for (int i = 0; i < n_elms; i++)
    tp.add_int32_data(static_cast<int32_t>(int8Data[i]));
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);

  // Don't clear while we're using it.
  tp.clear_raw_data();
}

void GraphTransformerImpl::convertINT16TensorToINT32(
    ONNX_NAMESPACE::TensorProto &tp) {
  if (tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT16) {
    auto descriptor     = ONNX_NAMESPACE::TensorProto_DataType_descriptor();
    auto data_type_name = descriptor->FindValueByNumber(tp.data_type())->name();
    throw error("cannot set tensor type {} to type INT32", data_type_name);
  }
  auto mutableData = onnxutil::getMutableData(tp);
  auto int16Data   = reinterpret_cast<const int16_t *>(mutableData.data);

  auto n_elms = mutableData.info.nelms();

  for (int i = 0; i < n_elms; i++)
    tp.add_int32_data(static_cast<int32_t>(int16Data[i]));
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);

  // Don't clear while we're using it.
  tp.clear_raw_data();
}

void GraphTransformerImpl::convertINT64TensorToINT32(
    ONNX_NAMESPACE::TensorProto &tp) {
  if (tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    auto descriptor     = ONNX_NAMESPACE::TensorProto_DataType_descriptor();
    auto data_type_name = descriptor->FindValueByNumber(tp.data_type())->name();
    throw error("cannot set tensor type {} to type INT64", data_type_name);
  }
  auto mutableData = onnxutil::getMutableData(tp);
  auto int64Data   = reinterpret_cast<const int64_t *>(mutableData.data);

  auto n_elms = mutableData.info.nelms();

  // Make sure data is within acceptable bounds for an int32 not to overflow
  for (int i = 0; i < n_elms; i++) {
    if (int64Data[i] > INT_MAX || int64Data[i] < INT_MIN) {
      throw error("In convertINT64TensorToINT32, cannot cast int64 to "
                  "int32: number is too large.");
    }
  }

  for (int i = 0; i < n_elms; i++)
    tp.add_int32_data(static_cast<int32_t>(int64Data[i]));
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);

  // Don't clear while we're using it.
  tp.clear_int64_data();
  tp.clear_raw_data();
}

void GraphTransformerImpl::convertBFloat16TensorToFloat32(
    ONNX_NAMESPACE::TensorProto &tp) {
  if (tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
    auto descriptor     = ONNX_NAMESPACE::TensorProto_DataType_descriptor();
    auto data_type_name = descriptor->FindValueByNumber(tp.data_type())->name();
    throw error("cannot set tensor type {} to type FLOAT32", data_type_name);
  }
  auto mutableData = onnxutil::getMutableData(tp);

  // Even if the data is BFloat 16 we extract it as int16, just so that be get
  // the 2 bytes out at once
  auto int16Data = reinterpret_cast<const int16_t *>(mutableData.data);

  // Initialize the recipient float32 array
  auto n_elms = mutableData.info.nelms();

  // The code below assumes floats are 32-bit. Let's check that. Note this is a
  // compile-time warning, not a run-time one.
  static_assert(sizeof(float) == sizeof(int32_t),
                "Expected floats to be 32-bit.");

  // To convert from bfloat to float32 we simply append 16 zeros (2 bytes)
  for (int i = 0; i < n_elms; i++) {
    int32_t buffer    = static_cast<int32_t>(int16Data[i]);
    buffer            = (buffer << 16);
    float floatBuffer = *reinterpret_cast<float *>(&buffer);
    tp.add_float_data(floatBuffer);
  }
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  // Don't clear while we're using it.
  tp.clear_raw_data();
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
  google::protobuf::RepeatedPtrField<ONNX_NAMESPACE::NodeProto> new_nodes;

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
        attr->set_type(ONNX_NAMESPACE::AttributeProto::TENSOR);
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

  ONNX_NAMESPACE::checker::check_model(model);
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
  std::vector<ONNX_NAMESPACE::ValueInfoProto> initialInputs;
  for (auto &x : graph->input()) {
    initialInputs.push_back(x);
  }

  std::map<TensorId, ONNX_NAMESPACE::TensorProto> initialInitializers;
  for (auto &x : graph->initializer()) {
    initialInitializers[x.name()] = x;
  }

  graph->clear_input();
  graph->clear_initializer();
  for (auto &iniIn : initialInputs) {
    TensorId id = iniIn.name();
    if (consumed.find(id) != consumed.end()) {
      ONNX_NAMESPACE::ValueInfoProto *vip = graph->add_input();
      *vip                                = iniIn;

      auto found = initialInitializers.find(id);
      if (found != initialInitializers.end()) {
        ONNX_NAMESPACE::TensorProto *tp = graph->add_initializer();
        *tp                             = found->second;
      }
    }
  }
}

void GraphTransformerImpl::prepareNodesForTraining() {
  static int counter = 0;
  auto *graph        = model.mutable_graph();
  for (int node_i = 0; node_i < graph->node_size(); ++node_i) {
    if (graph->node(node_i).op_type() == "BatchNormalization") {
      ONNX_NAMESPACE::NodeProto *mNode = graph->mutable_node(node_i);
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

void GraphTransformerImpl::saveInitializersExternally(
    const std::vector<TensorId> &ids,
    const std::string &fn) {
  onnxutil::saveInitializersExternally(model, ids, fn);
}

} // namespace popart
