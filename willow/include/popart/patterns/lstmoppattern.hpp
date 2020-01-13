#include <popart/patterns/pattern.hpp>

namespace popart {

class LSTMPattern : public PreAliasPattern {
public:
  bool matches(Op *op) const override;

  std::vector<const Tensor *> touches(Op *) const override { return {}; }

  bool apply(Op *op) const override;
};

} // namespace popart
