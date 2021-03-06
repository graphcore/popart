// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <cmath>
#include <onnxpasses/nodepatterns/constfoldops.hpp>
#include <popart/onnxutil.hpp>

namespace popart {
namespace onnxpasses {

using namespace ONNX_NAMESPACE;

poprithms::compute::host::DType onnxToPoprithms(int t) {
  return getPoprithmsDType(onnxutil::getDataType(t));
}

// Support concatenation.
uint64_t wrapAxis(int64_t axis, uint64_t rnk0) {
  if (axis < 0) {
    axis += rnk0;
  }
  if (axis < 0 || axis >= rnk0) {
    std::ostringstream oss;
    oss << "Invalid call, wrapAxis(axis=" << axis << ", rnk0=" << rnk0
        << "). Expected -rnk0 <= axis < rnk0. ";
    throw error(oss.str());
  }
  return axis;
}

Constants AbsCFold::fold(const NodeProto &node, const Constants &inputs) {
  Constants res;
  res.insert({node.output(0), inputs.at(node.input(0)).abs()});
  return res;
}

Constants BinaryCFold::fold(const NodeProto &node, const Constants &inputs) {
  using namespace poprithms::compute;
  Constants res;
  res.insert({node.output(0),
              inputs.at(node.input(0))
                  .binary(node.op_type(), inputs.at(node.input(1)))});
  return res;
}

Constants CastCFold::fold(const NodeProto &node, const Constants &inputs) {
  auto attr = Attributes(node.attribute());
  int64_t t_;
  attr.set(t_, "to");
  Constants res;
  res.insert(
      {node.output(0), inputs.at(node.input(0)).to(onnxToPoprithms(t_))});
  return res;
}

Constants ConcatCFold::fold(const NodeProto &node, const Constants &inputs) {
  using namespace poprithms::compute;
  auto attr = Attributes(node.attribute());

  std::vector<host::Tensor> inputsT;
  for (int i = 0; i < node.input_size(); ++i) {
    inputsT.push_back(inputs.at(node.input(i)));
  }

  Constants res;
  res.insert({node.output(0),
              host::concat(inputsT,
                           wrapAxis(attr.getAttribute<Attributes::Int>("axis"),
                                    inputsT.at(0).rank_u64()))});
  return res;
}

Constants ConstantCFold::fold(const NodeProto &node, const Constants &inputs) {
  using namespace poprithms::compute;
  auto pTensor = [&node]() {
    for (auto &attr : node.attribute()) {
      if (attr.name() == "sparse_value") {
        throw error(
            "The Constant op attribute 'sparse_value' is not supported.");
      }
      if (attr.name() == "value") {
        return &attr.t();
      }
    }
    throw error("Could not find the 'value' attribute on the constant node");
  }();
  const auto voidData = onnxutil::getConstData(*pTensor);
  auto type           = onnxToPoprithms(pTensor->data_type());
  // Due to getConstData for bool.
  if (type == poprithms::compute::host::DType::Boolean) {
    type = poprithms::compute::host::DType::Int32;
  }

  Constants res;
  // It's a const!
  res.insert({node.output(0),
              host::Tensor::copy(type, voidData.info.shape(), voidData.data)});
  return res;
}

Constants ConstantOfShapeCFold::fold(const NodeProto &node,
                                     const Constants &inputs) {
  using namespace poprithms::compute;
  const auto outShape =
      poprithms::ndarray::Shape(inputs.at(node.input(0)).getInt64Vector());

  auto getValueAttribute = [](auto &nodeLocal) -> const onnx::TensorProto * {
    for (auto &attr : nodeLocal.attribute()) {
      if (attr.name() == "value") {
        return &attr.t();
      }
    }
    throw error(
        "Could not find the 'value' attribute on the ConstantOfShape node");
  };

  const ONNX_NAMESPACE::TensorProto *value = getValueAttribute(node);

  Constants res;
  if (value == nullptr) {
    // if no value provided, use DataType::FLOAT and value 0.0f
    const auto outTensor =
        host::Tensor::scalar(poprithms::compute::host::DType::Float32, 0.0)
            .expand(outShape);
    res.insert({node.output(0), outTensor});
  } else {
    // TensorData from attribute value
    ConstVoidData valueCVData = onnxutil::getConstData(*value);

    const double *valueData =
        reinterpret_cast<const double *>(valueCVData.data);
    const double numValue = *valueData;
    const auto type       = onnxToPoprithms(value->data_type());
    const auto outTensor =
        host::Tensor::scalar(type, numValue).expand(outShape);
    res.insert({node.output(0), outTensor});
  }
  return res;
}

Constants ExpCFold::fold(const NodeProto &node, const Constants &inputs) {
  using namespace poprithms::compute;
  const auto in0 = inputs.at(node.input(0));
  const auto outTensor =
      host::Tensor::scalar(in0.dtype(), M_E).expand(in0.shape());

  Constants res;
  res.insert({node.output(0), outTensor.pow(in0)});
  return res;
}

Constants ExpandCFold::fold(const NodeProto &node, const Constants &inputs) {
  const auto outShape = inputs.at(node.input(1)).getInt64Vector();

  Constants res;
  res.insert({node.output(0), inputs.at(node.input(0)).expand(outShape)});
  return res;
}

Constants FloorCFold::fold(const NodeProto &node, const Constants &inputs) {
  Constants res;
  res.insert({node.output(0), inputs.at(node.input(0)).floor()});
  return res;
}

Constants FmodCFold::fold(const NodeProto &node, const Constants &inputs) {
  Constants res;
  res.insert(
      {node.output(0), inputs.at(node.input(0)).mod(inputs.at(node.input(1)))});
  return res;
}

Constants GatherCFold::fold(const NodeProto &node, const Constants &inputs) {
  auto attr = Attributes(node.attribute());

  int64_t axis       = attr.getAttribute<Attributes::Int>("axis", 0);
  uint64_t dimension = static_cast<uint64_t>(axis);
  const std::vector<int64_t> where = inputs.at(node.input(1)).getInt64Vector();
  Constants res;
  res.insert(
      {node.output(0), inputs.at(node.input(0)).gather(dimension, where)});
  return res;
}

Constants IdentityCFold::fold(const NodeProto &node, const Constants &inputs) {
  Constants res;
  res.insert({node.output(0), inputs.at(node.input(0)).copy()});
  return res;
}

Constants IdentityLossCFold::fold(const NodeProto &node,
                                  const Constants &inputs) {
  using namespace poprithms::compute;
  auto attr = Attributes(node.attribute());
  Constants res;
  res.insert({node.output(0), host::Tensor::float64(1)});
  return res;
}

Constants NegCFold::fold(const NodeProto &node, const Constants &inputs) {
  Constants res;
  res.insert({node.output(0), inputs.at(node.input(0)).neg()});
  return res;
}

Constants ReciprocalCFold::fold(const NodeProto &node,
                                const Constants &inputs) {
  Constants res;
  res.insert({node.output(0), inputs.at(node.input(0)).reciprocal()});
  return res;
}

Constants ReluCFold::fold(const NodeProto &node, const Constants &inputs) {
  Constants res;
  res.insert({node.output(0), inputs.at(node.input(0)).relu()});
  return res;
}

Constants ReshapeCFold::fold(const NodeProto &node, const Constants &inputs) {
  const auto nIns = node.input_size();

  if (nIns != 1) {
    // Enable this error check when 2-input-reshape pattern added.
    // throw error("ReshapeCFold expects one input, but has {} inputs.", nIns);
    Constants res;
    return res;
  }

  auto attr           = Attributes(node.attribute());
  const auto shapeOut = attr.getAttribute<Attributes::Ints>("shape", {});

  Constants res;
  res.insert(
      {node.output(0),
       inputs.at(node.input(0)).reshape(poprithms::ndarray::Shape(shapeOut))});
  return res;
}

Constants ScaleCFold::fold(const NodeProto &node, const Constants &inputs) {
  using namespace poprithms::compute;

  auto attr = Attributes(node.attribute());
  float scale;
  attr.set(scale, "scale");
  const auto in0     = inputs.at(node.input(0));
  const auto in0Type = in0.dtype();
  const auto f       = host::Tensor::float64(scale).to(in0Type);
  Constants res;
  res.insert({node.output(0), (in0 * f).to(in0Type)});
  return res;
}

Constants SliceCFold::fold(const NodeProto &node, const Constants &inputs) {
  auto attr = Attributes(node.attribute());

  const auto nIns = node.input_size();
  // If version 1, these are attributes.
  auto starts = attr.getAttribute<Attributes::Ints>("starts", {});
  auto ends   = attr.getAttribute<Attributes::Ints>("ends", {});
  auto axes   = attr.getAttribute<Attributes::Ints>("axes", {});
  auto steps  = attr.getAttribute<Attributes::Ints>("steps", {});
  if (nIns == 1) {
    // assert that opset 1
  } else {
    starts = inputs.at(node.input(1)).getInt64Vector();
    ends   = inputs.at(node.input(2)).getInt64Vector();
    if (nIns > 3) {
      axes = inputs.at(node.input(3)).getInt64Vector();
    }
    if (nIns > 4) {
      steps = inputs.at(node.input(4)).getInt64Vector();
    }
  }

  Constants res;
  res.insert({node.output(0),
              inputs.at(node.input(0))
                  .slice(poprithms::ndarray::Starts(starts),
                         poprithms::ndarray::Ends(ends),
                         poprithms::ndarray::Steps(steps),
                         poprithms::ndarray::Dims(axes))});
  return res;
}

Constants SqueezeCFold::fold(const NodeProto &node, const Constants &inputs) {
  auto attr = Attributes(node.attribute());

  const auto in0  = inputs.at(node.input(0));
  const auto axes = attr.getAttribute<Attributes::Ints>("axes", {});

  const auto tSqueeze = axes.empty()
                            ? in0.squeeze()
                            : in0.squeeze(getAxes_u64(axes, in0.rank_u64()));

  Constants res;
  res.insert({node.output(0), tSqueeze});
  return res;
}

Constants TransposeCFold::fold(const NodeProto &node, const Constants &inputs) {
  auto attr = Attributes(node.attribute());

  const auto perm = attr.getAttribute<Attributes::Ints>("perm", {});
  const auto tIn  = inputs.at(node.input(0));
  poprithms::util::Permutation px({perm.cbegin(), perm.cend()});
  if (perm.empty()) {
    px = poprithms::util::Permutation::reverse(tIn.rank_u64());
  }
  Constants res;
  res.insert({node.output(0), tIn.dimShuffle(px)});
  return res;
}

Constants UnsqueezeCFold::fold(const NodeProto &node, const Constants &inputs) {
  auto attr = Attributes(node.attribute());

  const auto in0  = inputs.at(node.input(0));
  const auto axes = attr.getAttribute<Attributes::Ints>("axes", {});

  Constants res;
  res.insert({node.output(0),
              in0.unsqueeze(getAxes_u64(axes, axes.size() + in0.rank_u64()))});
  return res;
}

// ShapeCFold takes a tensor as input and outputs an 1D int64 tensor containing
// the shape of the input tensor.
// Example: For input tensor of shape (2,3,5) output tensor is {2,3,5}.
Constants ShapeCFold::fold(const NodeProto &node, const Constants &inputs) {
  using namespace poprithms::compute;

  const auto inShape = inputs.at(node.input(0)).shape();

  Constants res;
  res.insert({node.output(0),
              host::Tensor::int64({inShape.rank_i64()}, inShape.get())});

  return res;
}

} // namespace onnxpasses
} // namespace popart
