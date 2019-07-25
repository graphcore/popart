#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/conv.hpp>
#include <popart/patterns/updateinplaceprioritiesforipu.hpp>
#include <popart/tensor.hpp>

namespace popart {

// Check for sequence:
//
// [Conv] -+
//         |
//         + -> X -> [Conv]
//         |
// [....] -+
//
// or:
//
// [Conv] -+
//         |
//         + -> X -> [ElementWiseUnary] -> [Conv]
//         |
// [....] -+
//
// where X is the op argument
template <typename OP> bool connectsConvs(const OP &op, InIndex argIndex) {
  const Tensor *inT = op.inTensor(argIndex);
  if (!inT->hasProducer() || !inT->getProducer()->isConvertibleTo<ConvOp>()) {
    return false;
  }

  auto isConsumedByConv = [](const Tensor *t) {
    for (auto consumer : t->consumers.getOps()) {
      if (consumer->isConvertibleTo<ConvOp>() &&
          consumer->inTensor(ConvOp::getDataInIndex()) == t) {
        return true;
      }
    }
    return false;
  };

  auto outT = op.outTensor(op.getOutIndex());
  if (isConsumedByConv(outT)) {
    return true;
  }

  for (Op *consumer : outT->consumers.getOps()) {
    if (consumer->isConvertibleTo<ElementWiseUnaryOp>()) {
      auto consumerOutT =
          consumer->outTensor(ElementWiseUnaryOp::getOutIndex());
      if (isConsumedByConv(consumerOutT)) {
        return true;
      }
    }
  }

  return false;
}

void UpdateInplacePrioritiesForIpu::applyImpl(AddOp &op) const {
  for (auto &id_p : op.inplacePriorityDefault()) {
    OperatorIdentifier id = std::get<0>(id_p);
    float priority        = std::get<1>(id_p);

    InIndex argIndex;
    if (id == Onnx::CustomOperators::AddLhsInplace) {
      argIndex = op.getArg0InIndex();
    } else if (id == Onnx::CustomOperators::AddRhsInplace) {
      argIndex = op.getArg1InIndex();
    } else {
      throw error("unrecognised inplace op {}", id);
    }

    if (connectsConvs(op, argIndex)) {
      priority += 10.0f;
    }

    op.setInplacePriority(id, priority);
  }
}

void UpdateInplacePrioritiesForIpu::apply(Op *op) const {
  if (op->isConvertibleTo<AddOp>()) {
    applyImpl(*dynamic_cast<AddOp *>(op));
  }
}

} // namespace popart
