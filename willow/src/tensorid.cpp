// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/tensorid.hpp>

#include <ostream>

namespace popart {

// Operators where TensorId is RHS
bool operator<(const char *lhs, const TensorId &rhs) { return lhs < rhs.str(); }
bool operator<(const std::string &lhs, const TensorId &rhs) {
  return lhs < rhs.str();
}

bool operator==(const char *lhs, const TensorId &rhs) {
  return lhs == rhs.str();
}
bool operator==(const std::string &lhs, const TensorId &rhs) {
  return lhs == rhs.str();
}

bool operator!=(const char *lhs, const TensorId &rhs) {
  return lhs != rhs.str();
}
bool operator!=(const std::string &lhs, const TensorId &rhs) {
  return lhs != rhs.str();
}

TensorId &TensorId::operator+=(const TensorId &other) {
  id += other.id;
  return *this;
}
TensorId &TensorId::operator+=(const char *other) {
  id += other;
  return *this;
}
TensorId &TensorId::operator+=(const std::string &other) {
  id += other;
  return *this;
}

std::string operator+(const char *lhs, const TensorId &tId) {
  return lhs + tId.str();
}
std::string operator+(const std::string &lhs, const TensorId &tId) {
  return lhs + tId.str();
}

std::string operator+=(const char *lhs, const TensorId &tId) {
  return lhs + tId.str();
}
std::string operator+=(const TensorId &lhs, const TensorId &tId) {
  return lhs.str() + tId.str();
}

// Stream operators
std::ostream &operator<<(std::ostream &ss, const TensorId &tensor_id) {
  ss << tensor_id.str();
  return ss;
}

} // namespace popart
