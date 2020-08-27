// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/gather.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/pad.hpp>
#include <popart/op/reducesum.hpp>
#include <popart/op/scale.hpp>
#include <popart/op/subsample.hpp>
#include <popart/opmanager.hpp>
#include <popart/patterns/optoidentitypattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

bool OpToIdentityPattern::matches(Op *op) const {
  return op->canBeReplacedByIdentity();
}

std::vector<std::unique_ptr<Op>> OpToIdentityPattern::sequence(Op *op) const {
  std::vector<std::unique_ptr<Op>> seq;

  // The GatherOp matches this pattern, but has more than 1 inputs. This was
  // resulting in an IdentityOp with two inputs. For the GatherOp, it is
  // suitable just to disconnect all but the first inputs, but this should
  // probably be an error.
  if (op->isConvertibleTo<GatherOp>()) {
    for (auto &index_tensor : op->input->tensorMap()) {
      int index   = index_tensor.first;
      auto tensor = index_tensor.second;
      if (index != 0) {
        op->disconnectInTensor(index, tensor);
      }
    }
  }

  // It should be an error to replace an op with multiple inputs.
  if (op->input->n() > 1) {
    throw error("Can not replace op with {} inputs with IdentityOp.",
                op->input->n());
  }

  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Identity, op));

  return seq;
}

namespace {
static PatternCreator<OpToIdentityPattern>
    opToIdentityPattern(PreAliasPatternType::OptoIdentity, "OpToIdentity");
}

} // namespace popart
