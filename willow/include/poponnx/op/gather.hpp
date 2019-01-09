#ifndef GUARD_NEURALNET_GATHER_HPP
#define GUARD_NEURALNET_GATHER_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class GatherOp : public Op {
public:
  GatherOp(const OperatorIdentifier &_opid,
           Ir *_ir,
           const std::string &name = "",
           const Attributes &_attr = {});

  std::unique_ptr<Op> clone() const final;
  void setup() final;

  // Which axis to gather on.
  int64_t getAxis() const;

  static InIndex dataInIndex() { return 0; }
  static InIndex indicesInIndex() { return 1; }
  static InIndex outIndex() { return 0; }

private:
  int64_t axis = 0;
};

} // namespace poponnx

#endif
