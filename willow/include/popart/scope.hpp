// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_SCOPE_HPP_
#define POPART_WILLOW_INCLUDE_POPART_SCOPE_HPP_

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>
#include <popart/names.hpp>

namespace popart {

class Op;

class Scope {
public:
  static std::string delimiter() { return sNameDelimiter; }

  bool empty() const { return names.empty(); }

  void pop();

  Scope getCommonParent(const Scope &) const;

  size_t depth() const { return names.size(); }

  bool operator==(const Scope &) const;
  bool operator!=(const Scope &) const;

  std::string str() const;

  Scope operator/(const std::string &name) const;

  operator std::string() { return str(); }

  // Is this scope contained within the argument scope
  bool isSubscope(const Scope &) const;

  static Scope getCommonParent(const std::vector<Op *> &);

  const std::vector<std::string> getScopeNames() const;

private:
  std::vector<std::string> names;
};

std::ostream &operator<<(std::ostream &, const Scope &);

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_SCOPE_HPP_
