// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <onnx/onnx_pb.h>
#include <onnxpasses/nodepatterns/constfoldops.hpp>
#include <onnxutil.hpp>
#include <ostream>
#include <vector>
#include <poprithms/compute/host/tensor.hpp>
#include <poprithms/ndarray/accessors.hpp>
#include <poprithms/ndarray/dtype.hpp>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/permutation.hpp>

#include "popart/attributes.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/util.hpp"
#include "popart/voiddata.hpp"
#include "poprithmshosttensor.hpp"

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
    // if no value provided, use DataType::FLOAT and value 0, as per spec.
    const auto outTensor = host::Tensor::float32(0.0f).expand(outShape);
    res.insert({node.output(0), outTensor});
  } else {
    // TensorData from attribute value
    ConstVoidData valueCVData = onnxutil::getConstData(*value);
    const auto type           = onnxToPoprithms(value->data_type());
    const auto outTensor =
        host::Tensor::copy(type, {}, valueCVData.data).expand(outShape);
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

Constants ReduceProdCFold::fold(const NodeProto &node,
                                const Constants &inputs) {
  const auto in0 = inputs.at(node.input(0));
  auto in0_shape = in0.shape().get();
  auto in0_rank  = in0_shape.size();

  // Prepare default axes in case they're not specified
  std::vector<int64_t> default_axes(in0_rank);
  std::iota(default_axes.begin(), default_axes.end(), 0);

  // Fetch the axes and keepdims
  auto attr = Attributes(node.attribute());
  std::vector<int64_t> axes =
      attr.getAttribute<Attributes::Ints>("axes", default_axes);
  const auto keepdims = attr.getAttribute<Attributes::Int>("keepdims", 1);

  // Mimic ReduceOp::setup
  // Check the axes are all in the right range.
  validateReduceAxes(axes, in0_rank, "ReduceProdCFold::fold");
  // Normalize to positive axes.
  normalizeReduceAxes(axes, in0_rank);

  // Calculate resulting shape after reduction
  auto reducedShapeInitializer(in0_shape);
  for (auto i : axes)
    reducedShapeInitializer[i] = 1;

  auto reducedShape = poprithms::ndarray::Shape(reducedShapeInitializer);

  auto reducedTensor = in0.reduceProduct(reducedShape);

  if (!keepdims) { // remove the reduced axes if necessary
    std::vector<int64_t> prunedShapeInitializer;
    for (int i = 0; i < reducedShapeInitializer.size(); ++i) {
      bool shouldBeKept =
          (std::find(axes.begin(), axes.end(), i) == axes.end());
      if (shouldBeKept) {
        prunedShapeInitializer.push_back(reducedShapeInitializer[i]);
      }
    }
    reducedTensor = reducedTensor.reshape(prunedShapeInitializer);
  }

  Constants res;
  res.insert({node.output(0), reducedTensor});

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

  auto attr     = Attributes(node.attribute());
  auto shapeOut = attr.getAttribute<Attributes::Ints>("shape", {});

  const poprithms::compute::host::Tensor &constant(inputs.at(node.input(0)));

  // Handle a negative dim in the reshape
  size_t nonMinusOneProd = 1;
  int64_t minusOneDim    = -1;
  for (size_t dim = 0; dim < shapeOut.size(); dim++) {
    if (shapeOut[dim] == -1) {
      minusOneDim = dim;
    } else if (shapeOut[dim] == 0) {
      shapeOut[dim] = constant.shape().dim(dim);
      nonMinusOneProd *= shapeOut[dim];
    } else {
      nonMinusOneProd *= shapeOut[dim];
    }
  }

  if (minusOneDim != -1) {
    shapeOut[minusOneDim] = constant.nelms() / nonMinusOneProd;
  }

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

Constants ShapeCFold::fold(const NodeProto &node, const Constants &inputs) {
  using namespace poprithms::compute;

  const auto inShape = inputs.at(node.input(0)).shape();

  Constants res;
  res.insert({node.output(0),
              host::Tensor::int64({inShape.rank_i64()}, inShape.get())});

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

} // namespace onnxpasses
} // namespace popart
