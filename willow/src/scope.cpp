#include <sstream>
#include <poponnx/op.hpp>
#include <poponnx/scope.hpp>

namespace poponnx {

void Scope::pop() { names.pop_back(); }

Scope Scope::operator/(const std::string &name) const {
  Scope result(*this);
  result.names.push_back(name);
  return result;
}

Scope Scope::getCommonParent(const Scope &other) const {
  Scope result;

  for (int i = 0; i < std::min(names.size(), other.names.size()); i++) {
    if (names[i] == other.names[i]) {
      result.names.push_back(names[i]);
    } else {
      break;
    }
  }

  return result;
}

bool Scope::operator==(const Scope &other) const {
  if (names.size() != other.names.size()) {
    return false;
  }

  for (int i = 0; i < names.size(); i++) {
    if (names[i] != other.names[i]) {
      return false;
    }
  }

  return true;
}

bool Scope::operator!=(const Scope &other) const { return !(*this == other); }

std::string Scope::str() const {
  if (names.size() == 0) {
    return "";
  }

  std::stringstream ss;
  for (int i = 0; i < names.size() - 1; i++) {
    ss << names[i] << "/";
  }
  ss << names.back();

  return ss.str();
}

bool Scope::isSubscope(const Scope &other) const {
  if (names.size() <= other.names.size()) {
    return false;
  }

  for (int i = 0; i < other.names.size(); i++) {
    if (names[i] != other.names[i]) {
      return false;
    }
  }

  return true;
}

Scope Scope::getCommonParent(const std::vector<Op *> &ops) {
  if (ops.size() == 0) {
    return Scope();
  }

  auto new_scope = ops[0]->getScope();

  for (int i = 0; i < ops.size(); i++) {
    new_scope = new_scope.getCommonParent(ops[i]->getScope());
  }

  return new_scope;
};

std::ostream &operator<<(std::ostream &ss, const Scope &scope) {
  ss << scope.str();
  return ss;
}

} // namespace poponnx
