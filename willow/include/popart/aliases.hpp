// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_ALIASES_HPP_
#define POPART_WILLOW_INCLUDE_POPART_ALIASES_HPP_

#include <set>
#include <string>
#include <unordered_map>
#include <popart/chains.hpp>
#include <popart/names.hpp>

namespace popart {

struct PTensorCmp;
class Tensor;

class Aliases {
public:
  Aliases() {}
  virtual ~Aliases() = default;

  Aliases(const Aliases &) = delete;
  Aliases &operator=(const Aliases &rhs) = default;

  void clearAliases();
  void updateAliases(Tensor *t1,
                     Tensor *t2,
                     view::Regions inRegions,
                     view::RegMap fwdMap,
                     view::RegMap bwdMap,
                     std::string fwdLinkName = "None",
                     std::string bwdLinkName = "None");
  view::Regions getAliasRegions(Tensor *from, Tensor *to) const;

  // all non-empty alias Chains to "to"
  // returned map M will always have M[to] = "the identity chain"
  //......"from"...."chains"............................"to"
  //       ^         ^                                   ^
  std::unordered_map<Tensor *, view::Chains> aliasChainsTo(Tensor *to) const;

  // all non-empty alias Chains from "from"
  // returned map M will always have M[from] = "the identity chain"
  //......"to"......"chains".............................."from"
  //       ^         ^                                     ^
  std::unordered_map<Tensor *, view::Chains>
  aliasChainsFrom(Tensor *from) const;

  view::Chains getChainsFromTo(Tensor *from, Tensor *to) const;

  void addAllAliases(const Aliases &other);

  std::set<Tensor *, PTensorCmp> getTensors() const;

private:
  // all non-empty Chains
  //                "to"........................."from"...."chains"
  //                 ^                            ^         ^
  std::unordered_map<Tensor *, std::unordered_map<Tensor *, view::Chains>>
      aliasChainsToKey;

  // the mirror of the above
  std::unordered_map<Tensor *, std::unordered_map<Tensor *, view::Chains>>
      aliasChainsFromKey;

  // return M[t], but with guaranteed identity Chains from t
  std::unordered_map<Tensor *, view::Chains> getAliasChains(
      const std::unordered_map<Tensor *,
                               std::unordered_map<Tensor *, view::Chains>> &M,
      Tensor *t) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_ALIASES_HPP_
