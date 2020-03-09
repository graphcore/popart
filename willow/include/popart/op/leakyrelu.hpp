#ifndef GUARD_NEURALNET_LRELU_HPP
#define GUARD_NEURALNET_LRELU_HPP

#include <popart/op/elementwise.hpp>

namespace popart {

class LeakyReluOpBaseAttributes {
public:
  LeakyReluOpBaseAttributes(float _alpha) : alpha(_alpha) {}

  float getAlpha() const { return alpha; }

private:
  float alpha;
};

class LeakyReluOp : public ElementWiseUnaryOp,
                    public LeakyReluOpBaseAttributes {
public:
  LeakyReluOp(const OperatorIdentifier &_opid,
              float _alpha,
              const Op::Settings &settings_);
  std::unique_ptr<Op> clone() const final;
  std::vector<std::unique_ptr<Op>> getGradOps() final;

  std::vector<std::tuple<OperatorIdentifier, float>>
  inplacePriorityDefault() const final;

  void appendAttributes(popart::OpSerialiserBase &os) const override;
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;

  std::unique_ptr<Op> getInplaceVariant(const OperatorIdentifier &) const final;
};

class LeakyReluInplaceOp : public ElementWiseInplaceUnaryOp,
                           public LeakyReluOpBaseAttributes {
public:
  LeakyReluInplaceOp(const LeakyReluOp &);
  std::unique_ptr<Op> clone() const final;

  void appendAttributes(popart::OpSerialiserBase &os) const override;
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;
};

class LeakyReluGradOp : public ElementWiseNonLinearUnaryGradOp,
                        public LeakyReluOpBaseAttributes {
public:
  LeakyReluGradOp(const LeakyReluOp &);
  std::unique_ptr<Op> clone() const final;

  void appendAttributes(popart::OpSerialiserBase &os) const override;
  void appendOutlineAttributes(popart::OpSerialiserBase &os) const override;
};

} // namespace popart

#endif