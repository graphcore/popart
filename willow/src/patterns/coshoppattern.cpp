#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/add.hpp>
#include <poponnx/op/cosh.hpp>
#include <poponnx/op/exp.hpp>
#include <poponnx/op/negate.hpp>
#include <poponnx/op/scale.hpp>
#include <poponnx/patterns/coshoppattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorinfo.hpp>

namespace poponnx {

bool CoshOpPattern::matches(Op *op) const {
  return op->isConvertibleTo<CoshOp>();
}

std::vector<const Tensor *> CoshOpPattern::touches(Op *) const { return {}; }

// output = (exp(input) + exp(-input)) * 0.5
bool CoshOpPattern::apply(Op *op) const {
  auto input  = op->inTensor(CoshOp::getInIndex());
  auto output = op->outTensor(CoshOp::getOutIndex());

  auto ir = op->pir;

  // create the new ops
  auto negate_op =
      make_unique<NegateOp>(OpConstructorBundle{OpType::NEGATE, ir, {}});
  auto exp1_op = make_unique<ExpOp>(OpConstructorBundle{OpType::EXP, ir, {}});
  auto exp2_op = make_unique<ExpOp>(OpConstructorBundle{OpType::EXP, ir, {}});
  auto add_op  = make_unique<AddOp>(OpConstructorBundle{OpType::ADD, ir, {}});
  auto scale_op =
      make_unique<ScaleOp>(OpConstructorBundle{OpType::SCALE, ir, {}}, 0.5f);

  // move ops into ir
  auto negate = negate_op.get();
  auto exp1   = exp1_op.get();
  auto exp2   = exp2_op.get();
  auto add    = add_op.get();
  auto scale  = scale_op.get();
  ir->moveIntoIr(std::move(negate_op));
  ir->moveIntoIr(std::move(exp1_op));
  ir->moveIntoIr(std::move(exp2_op));
  ir->moveIntoIr(std::move(add_op));
  ir->moveIntoIr(std::move(scale_op));

  // Remove the CoshOp
  op->disconnectAllInputs();
  op->disconnectAllOutputs();
  ir->eraseOp(op->id);

  // Connect up the new ops
  exp1->connectInTensor(ExpOp::getInIndex(), input->id);
  exp1->createAndConnectOutTensor(ExpOp::getOutIndex(),
                                  createImtermediateTensorId(input->id));
  exp1->setup();

  negate->connectInTensor(NegateOp::getInIndex(), input->id);
  negate->createAndConnectOutTensor(NegateOp::getOutIndex(),
                                    createImtermediateTensorId(input->id));
  negate->setup();

  exp2->connectInTensor(ExpOp::getInIndex(),
                        negate->outTensor(NegateOp::getOutIndex())->id);
  exp2->createAndConnectOutTensor(ExpOp::getOutIndex(),
                                  createImtermediateTensorId(input->id));
  exp2->setup();

  add->connectInTensor(AddOp::getArg0InIndex(),
                       exp1->outTensor(ExpOp::getOutIndex())->id);
  add->connectInTensor(AddOp::getArg1InIndex(),
                       exp2->outTensor(ExpOp::getOutIndex())->id);
  add->createAndConnectOutTensor(AddOp::getOutIndex(),
                                 createImtermediateTensorId(input->id));
  add->setup();

  scale->connectInTensor(ScaleOp::getInIndex(),
                         add->outTensor(ScaleOp::getOutIndex())->id);
  scale->connectOutTensor(ScaleOp::getOutIndex(), output->id);

  return true;
}

namespace {
static PatternCreator<CoshOpPattern> CoshOpPattern(PatternType::COSHOP,
                                                   "CoshOp");
}

} // namespace poponnx
