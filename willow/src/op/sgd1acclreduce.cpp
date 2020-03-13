// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/op/sgd1acclreduce.hpp>
#include <popart/opserialiser.hpp>

namespace popart {

std::unique_ptr<Op>
SGD1AcclReduceOp::cloneWithNewName(const TensorId &x) const {
  return std::make_unique<SGD1AcclReduceOp>(x, settings);
}

std::unique_ptr<Op> SGD1AcclReduceOp::clone() const {
  return std::make_unique<SGD1AcclReduceOp>(*this);
}

std::map<InIndex, TensorId> SGD1AcclReduceOp::optimizerInputs() const {
  return {};
}

void SGD1AcclReduceOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
}

SGD1AcclReduceOp::SGD1AcclReduceOp(const TensorId &varToUpdate,
                                   const Op::Settings &opSettings)
    : VarUpdateWithoutUpdaterOp(Onnx::CustomOperators::SGD1AcclReduce,
                                varToUpdate,
                                opSettings) {}

} // namespace popart
