#include <popart/vectorandset.hpp>

namespace popart {

const std::vector<std::string> &VectorAndSet::v() const { return v_vals; }

bool VectorAndSet::contains(std::string name) const {
  return m_vals.count(name) == 1;
}

VectorAndSet::~VectorAndSet() = default;

VectorAndSet::VectorAndSet() {}

VectorAndSet::VectorAndSet(const std::vector<std::string> &vals)
    : v_vals(vals) {
  for (auto &v : v_vals) {
    m_vals.insert(v);
  }
}

void VectorAndSet::reset(const std::vector<std::string> &vals) {

  // Replace the old with the new
  v_vals = vals;

  // Clear and initialise the m_vals set
  m_vals.clear();
  for (auto &v : v_vals) {
    m_vals.insert(v);
  }
}

void VectorAndSet::insert(const std::string &id) {
  if (m_vals.find(id) == m_vals.end()) {
    v_vals.push_back(id);
    m_vals.insert(id);
  }
}

} // namespace popart
