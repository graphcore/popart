#include <poponnx/makeunique.hpp>
#include <poponnx/op/sum.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>

namespace poponnx {

SumOp::SumOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : Op(_opid, settings_) {
  // TODO : Do not broadcast in version 6
}

std::unique_ptr<Op> SumOp::clone() const { return make_unique<SumOp>(*this); }

void SumOp::setup() {
  outInfo(getOutIndex()) = inInfo(0);
  for (int i = 1; i < input->n(); ++i) {
    outInfo(getOutIndex()) = npOut(outInfo(getOutIndex()), inInfo(i));
  }
}

namespace {
static OpCreator<SumOp> sumOpCreator({Onnx::Operators::Sum_6,
                                      Onnx::Operators::Sum_8});
} // namespace

} // namespace poponnx
