#ifndef GUARD_NEURALNET_GLOBALMAXPOOL_HPP
#define GUARD_NEURALNET_GLOBALMAXPOOL_HPP

#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/receptive.hpp>

namespace poponnx {

class GlobalMaxPoolOp : public Op {
public:
  GlobalMaxPoolOp(const OperatorIdentifier &_opid,
                  const Op::Settings &settings);

  std::unique_ptr<Op> clone() const final;

  void setup() override;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

  Shape getSpatialK() const { return kernel; }
  Shape getStrides() const;
  Shape getLowerPads() const;
  Shape getUpperPads() const;

  void appendAttributes(OpSerialiserBase &) const override;

private:
  Shape kernel;
};

class GlobalMaxPoolGradOp : public Op {
public:
  GlobalMaxPoolGradOp(const GlobalMaxPoolOp &);
  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;
  void setup() final;

  static InIndex getPrePooledInIndex() { return 0; }
  static InIndex getPooledInIndex() { return 1; }
  static InIndex getGradPooledInIndex() { return 2; }
  static OutIndex getOutIndex() { return 0; }

  const GlobalMaxPoolOp *getCloneOfCreator() const;

  void appendAttributes(OpSerialiserBase &) const override;

private:
  // The shape and type of the input to the
  // forward op which creates this backwards op
  TensorInfo unpooledInfo;
  // A copy of the forward op which creates
  // this backwards op. Note
  // 1) backends will need a copy of this op to determine
  //    how to do the backwards pass (padding, striding, etc)
  // 2) we DON'T store a pointer to the creating forward op,
  //    which might be optimised out and deleted
  std::unique_ptr<Op> cloneOfCreator;
};

} // namespace poponnx

#endif
