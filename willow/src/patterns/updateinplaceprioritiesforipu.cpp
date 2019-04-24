#include <poponnx/ir.hpp>
#include <poponnx/op/add.hpp>
#include <poponnx/op/conv.hpp>
#include <poponnx/patterns/updateinplaceprioritiesforipu.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

bool UpdateInplacePrioritiesForIpuPattern::matches(Op *op) const {
  // Currently the only op this pattern supports is the add op
  return op->isConvertibleTo<AddOp>();
}

std::vector<const Tensor *>
UpdateInplacePrioritiesForIpuPattern::touches(Op *) const {
  return {};
}

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

bool UpdateInplacePrioritiesForIpuPattern::applyImpl(AddOp &op) const {
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

bool UpdateInplacePrioritiesForIpuPattern::apply(Op *op) const {
  if (op->isConvertibleTo<AddOp>()) {
    return applyImpl(*dynamic_cast<AddOp *>(op));
  } else {
    throw error("Unsupported op type {}", op->opid);
  }
}

namespace {
static PatternCreator<UpdateInplacePrioritiesForIpuPattern>
    updateInplacePrioritiesForIpuPattern(
        PreAliasPatternType::UPDATEINPLACEPRIORITIESFORIPU,
        "UpdateInplacePrioritiesForIpuPattern");
}

} // namespace poponnx
