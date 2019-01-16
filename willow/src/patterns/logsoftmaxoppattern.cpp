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
bool LogSoftmaxOpPattern::apply(Op *op) const {
  auto input  = op->inTensor(LogSoftmaxOp::getInIndex());
  auto output = op->outTensor(LogSoftmaxOp::getOutIndex());

  // create the new ops
  auto softmax = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Softmax, op);
  auto log     = makeReplacementOpInIr(Onnx::AiOnnx::OpSet9::Log, op);

  // Remove the LogSoftmaxOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  op->pir->eraseOp(op->id);

  // Connect up the new ops
  softmax->connectInTensor(SoftmaxOp::getInIndex(), input->id);
  auto softmax_out = createIntermediateTensorId(output->id);
  softmax->createAndConnectOutTensor(SoftmaxOp::getOutIndex(), softmax_out);
  softmax->setup();

  log->connectInTensor(LogOp::getInIndex(), softmax_out);
  log->connectOutTensor(LogOp::getOutIndex(), output->id);
  log->setup();

  return true;
}

namespace {
static PatternCreator<LogSoftmaxOpPattern>
    LogSoftmaxOpPattern(PatternType::LOGSOFTMAXOP, "LogSoftmaxOp");
}

} // namespace poponnx
