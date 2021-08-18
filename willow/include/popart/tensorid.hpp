// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_TENSORID_HPP
#define GUARD_NEURALNET_TENSORID_HPP
#include <ostream>
#include <string>

namespace popart {

class TensorId {
public:
  TensorId() : id(""){};
  TensorId(const TensorId &tId) : id(tId.id){};
  TensorId(const std::string &id_) : id(id_){};
  TensorId(const char *id_) : id(id_){};

  // Operation overloads
  bool operator<(const TensorId &other) const { return this->id < other.id; }
  bool operator<(const char *other) const { return this->id < other; }
  bool operator<(const std::string &other) const { return this->id < other; }

  bool operator==(const TensorId &other) const { return this->id == other.id; }
  bool operator==(const char *other) const { return this->id == other; }
  bool operator==(const std::string &other) const { return this->id == other; }

  bool operator!=(const TensorId &other) const { return this->id != other.id; }
  bool operator!=(const char *other) const { return this->id != other; }
  bool operator!=(const std::string &other) const { return this->id != other; }

  TensorId operator+(const TensorId other) const {
    return TensorId(id + other.id);
  }
  TensorId operator+(const char *other) const { return TensorId(id + other); }
  TensorId operator+(const std::string &other) const {
    return TensorId(id + other);
  }

  TensorId &operator+=(const TensorId &other);
  TensorId &operator+=(const char *other);
  TensorId &operator+=(const std::string &other);

  // Return string
  const std::string &str() const { return id; };

private:
  std::string id;
};

// Operators where TensorId is RHS
bool operator<(const char *lhs, const TensorId &rhs);
bool operator<(const std::string &lhs, const TensorId &rhs);

bool operator==(const char *lhs, const TensorId &rhs);
bool operator==(const std::string &lhs, const TensorId &rhs);

bool operator!=(const char *lhs, const TensorId &rhs);
bool operator!=(const std::string &lhs, const TensorId &rhs);

std::string operator+(const char *lhs, const TensorId &tId);
std::string operator+(const std::string &lhs, const TensorId &tId);

std::string operator+=(const char *lhs, const TensorId &tId);
std::string operator+=(const std::string &lhs, const TensorId &tId);

// Stream operators
std::ostream &operator<<(std::ostream &, const TensorId &);

// Hash value for boost
inline std::size_t hash_value(const TensorId &tId) {
  return std::hash<std::string>()(tId.str());
}

} // namespace popart

// To enable hashing of new classes, this is the recommended approach from
// https://en.cppreference.com/w/cpp/utility/hash
namespace std {
template <> struct hash<popart::TensorId> {
  std::size_t operator()(popart::TensorId const &tId) const noexcept {
    return hash<string>{}(tId.str());
  }
};
} // namespace std

#endif