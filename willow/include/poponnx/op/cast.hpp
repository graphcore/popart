#ifndef GUARD_NEURALNET_CAST_HPP
#define GUARD_NEURALNET_CAST_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class CastOp : public Op {
public:
  CastOp(const OperatorIdentifier &_opid,
         DataType _to,
         const Op::Settings &settings);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;
  void setup() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

private:
  DataType to;
};

} // namespace poponnx

#endif
