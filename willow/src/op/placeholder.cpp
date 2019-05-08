#include <poponnx/makeunique.hpp>
#include <poponnx/op/placeholder.hpp>

namespace poponnx {

PlaceholderOp::PlaceholderOp(const OperatorIdentifier &opid_,
                             const Op::Settings &settings_)
    : Op(opid_, settings_) {}

std::unique_ptr<Op> PlaceholderOp::clone() const {
  return make_unique<PlaceholderOp>(*this);
}

} // namespace poponnx
