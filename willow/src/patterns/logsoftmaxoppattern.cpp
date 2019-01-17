#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/log.hpp>
#include <poponnx/op/logsoftmax.hpp>
#include <poponnx/op/softmax.hpp>
#include <poponnx/patterns/logsoftmaxoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool LogSoftmaxOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<LogSoftmaxOp>();
}

// output = log(softmax(x))
std::vector<std::unique_ptr<Op>> LogSoftmaxOpPattern::sequence(Op *op) const {
  std::vector<std::unique_ptr<Op>> seq;

  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Softmax, op, {}));
  seq.push_back(makeReplacementOp(Onnx::AiOnnx::OpSet9::Log, op, {}));

  return seq;
}

namespace {
static PatternCreator<LogSoftmaxOpPattern>
    LogSoftmaxOpPattern(PatternType::LOGSOFTMAXOP, "LogSoftmaxOp");
}

} // namespace poponnx
