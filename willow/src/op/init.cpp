#include <algorithm>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/init.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

InitOp::InitOp(const OperatorIdentifier &_opid,
               const TensorInfo &tensor_info_,
               const TensorType &tensor_type_,
               const InitType &init_type_,
               const Op::Settings &settings_)
    : Op(_opid, settings_), tensor_info(tensor_info_),
      tensor_type(tensor_type_), init_type(init_type_) {}

std::unique_ptr<Op> InitOp::clone() const {
  return std::make_unique<InitOp>(*this);
}

void InitOp::setup() {
  outInfo(getOutIndex()) = tensor_info;
  output->tensor(getOutIndex())->setTensorType(tensor_type);
}

} // namespace popart
