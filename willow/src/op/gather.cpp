#include <algorithm>
#include <string>
#include <vector>

#include <poponnx/makeunique.hpp>
#include <poponnx/op/gather.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

GatherOp::GatherOp(const OperatorIdentifier &_opid,
                   Ir *_ir,
                   const std::string &name,
                   const Attributes &_attr)
    : Op(_opid, _ir, name, _attr) {
  nAtts.setIfPresent(axis, "axis");
}

std::unique_ptr<Op> GatherOp::clone() const {
  return make_unique<GatherOp>(*this);
}

int64_t GatherOp::getAxis() const { return axis; }

void GatherOp::setup() {
  // ONNX allows the axis attribute to be negative
  axis = axis % inShape(dataInIndex()).size(); // axis in the range [-m+1, m-1]
  axis += inShape(dataInIndex()).size();       // axis in the range [0, 2m-1]
  axis = axis % inShape(dataInIndex()).size(); // axis in the range [0, m-1]

  // Replace the axis dimension with the indices shape
  auto data_shape            = inShape(dataInIndex());
  const auto indices_shape   = inShape(indicesInIndex());
  const auto insertion_point = data_shape.erase(data_shape.begin() + axis);

  data_shape.insert(
      insertion_point, indices_shape.begin(), indices_shape.end());

  // Use the computed shape with the data input type
  outInfo(outIndex()) =
      TensorInfo(inInfo(dataInIndex()).dataType(), data_shape);
}

namespace {
static OpCreator<GatherOp> gatherOpCreator(Onnx::Operators::Gather);
} // namespace

} // namespace poponnx
