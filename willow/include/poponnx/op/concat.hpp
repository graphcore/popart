#ifndef GUARD_NEURALNET_CONCAT_HPP
#define GUARD_NEURALNET_CONCAT_HPP

#include <poponnx/op.hpp>

namespace poponnx {

class ConcatOp : public Op {
public:
  ConcatOp(const OperatorIdentifier &_opid,
           int64_t axis_,
           const Op::Settings &settings);

  std::unique_ptr<Op> clone() const override;
  void setup() final;

  std::vector<std::unique_ptr<Op>> getGradOps() final;

  int64_t getAxis() const;

  // note that this is not final, ConcatInplaceOp overrides it
  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const override;

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &) const override;

  static InIndex getInIndex(InIndex index) { return index; }
  static OutIndex getOutIndex() { return 0; }

  view::RegMap fwdRegMap(InIndex) const final;
  view::RegMap bwdRegMap(OutIndex) const final;
  // "uses" is still the full input region
  // "aliases" is still the empty region
  // "modifies" is still the empty region

  void appendAttributes(std::stringstream &ss,
                        const std::string &tab) const override;

  static Shape getOutputShape(int64_t axis,
                              const std::vector<const Shape *> inputs);

private:
  int64_t axis = 0;

  // suppose input tensors have shapes,
  // 0: [2,5,3]
  // 1: [2,6,3]
  // 2: [2,1,3]
  // and axis = 1.
  // then the output tensor has shape [2,12,3], and
  // the regions in the output that inputs corresponds
  // to are,
  // 0: [0:2,  0:5,  0:3]
  // 1: [0:2,  5:11, 0:3]
  // 2: [0:2, 11:12, 0:3]
  // outOffests are where these regions start/end along "axis", so
  // in this case {0,5,11,12}
  std::vector<int64_t> outOffsets;

  void regMapPreChecks(InIndex inIndex) const;
};

// An inplace variant of the concat op
class ConcatInplaceOp : public ConcatOp {
public:
  ConcatInplaceOp(const ConcatOp &concatOp, int64_t axis_);

  std::unique_ptr<Op> clone() const override;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final {
    return {};
  }

  std::unique_ptr<Op>
  getInplaceVariant(const OperatorIdentifier &o) const final {
    // this throws an error
    return Op::getInplaceVariant(o);
  }

  // The whole of the used area is aliased. "modifies" is still empty
  view::Region aliases(InIndex index) const final { return uses(index); }
};

class ConcatGradOp : public Op {
public:
  ConcatGradOp(const ConcatOp &op, InIndex input);
  ConcatGradOp(const ConcatInplaceOp &op, InIndex input);

  std::unique_ptr<Op> clone() const override;
  void setup() override;

  const std::vector<GradInOutMapper> &gradInputInfo() const final;
  const std::map<int, int> &gradOutToNonGradIn() const final;

  int64_t getAxis() const;
  int64_t getStart() const;
  int64_t getEnd() const;

  static InIndex getInIndex() { return 0; }
  static OutIndex getOutIndex() { return 0; }

protected:
  // An unsafe constructor that allows using any OperatorIdentifier
  ConcatGradOp(const OperatorIdentifier &_opid,
               const ConcatGradOp &concat_grad_op);

private:
  int64_t axis;
  int64_t start;
  int64_t end;
  InIndex fwdInput;

  TensorInfo gradInfo;
  std::map<int, int> gradOutToNonGradInInfo;
};

} // namespace poponnx

#endif
