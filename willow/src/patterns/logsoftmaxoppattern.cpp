// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/ir.hpp>
#include <popart/op/log.hpp>
#include <popart/op/logsoftmax.hpp>
#include <popart/op/softmax.hpp>
#include <popart/patterns/logsoftmaxoppattern.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

bool LogSoftmaxOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<LogSoftmaxOp>();
}

// output = log(softmax(x))
std::vector<std::unique_ptr<Op>> LogSoftmaxOpPattern::sequence(Op *op) const {
  std::vector<std::unique_ptr<Op>> seq;

  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Softmax, op));

  auto axis      = dynamic_cast<LogSoftmaxOp *>(op)->getAxis();
  auto softmaxOp = dynamic_cast<SoftmaxOp *>(seq.at(0).get());
  softmaxOp->setAxis(axis);

  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Log, op));

  return seq;
}

namespace {
static PatternCreator<LogSoftmaxOpPattern>
    LogSoftmaxOpPattern(PreAliasPatternType::LOGSOFTMAXOP, "LogSoftmaxOp");
}

} // namespace popart
