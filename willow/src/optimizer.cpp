#include <willow/optimizer.hpp>

namespace willow {

SGD::SGD(float l) : Optimizer(), learnRate_(l) {}

float SGD::learnRate() { return learnRate_; }

std::unique_ptr<Optimizer> SGD::clone() const {
  return std::unique_ptr<Optimizer>(new SGD(*this));
}

} // namespace willow
