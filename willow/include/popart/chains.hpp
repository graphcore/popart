#ifndef GUARD_NEURALNET_CHAINS_HPP
#define GUARD_NEURALNET_CHAINS_HPP

#include <memory>
#include <string>
#include <vector>
#include <popart/names.hpp>
#include <popart/region.hpp>

// we currently only consider inplacing ops with 1 output. this can be
// generalised in the future if we decide it is necessary

namespace popart {
namespace view {

// a class for mapping a Region to another Region
// by (1) applying a filter and then (2) mapping it
class Link {
public:
  // A link with the identity region mapper, so that regmap(r) = r.
  static Link getIdentity(const Region &filter);

  Link(const Region &r_filter, const RegMap &r2r_mapper);
  Link(const Region &r_filter,
       const RegMap &r2r_mapper,
       const std::string &dbName);
  Regions apply(const Region &r) const { return regmap(filter.intersect(r)); }
  const Region &getFilter() const { return filter; }
  bool contains(const Link &rhs) const;
  void append(std::ostream &) const;

private:
  Region filter;
  RegMap regmap;
  std::string debugName;
};

std::ostream &operator<<(std::ostream &, const Link &);

// a sequence of Links
class Chain {
public:
  // a single indentity Link
  static Chain getIdentity(const Region &);

  Chain(const Link &l) { links = {l}; }
  Regions apply(const Region &) const;
  void append(const Chain &);
  const std::vector<Link> &getLinks() const { return links; }
  // Returns true when apply(a full tensor region) = empty region
  bool untraversable() const;
  bool contains(const Chain &rhs) const;

private:
  std::vector<Link> links;
};

std::ostream &operator<<(std::ostream &, const Chain &);

// a set of parallel Chain objects
class Chains {

public:
  // a single identity Chain
  static Chains getIdentity(const Region &);
  static Chains getIdentity(const Shape &);

  // default constructor has no Chain objects
  Chains() = default;
  Chains(const Link &);
  Chains(const std::vector<Chain> &);
  Chains series(const Chains &) const;
  Chains parallel(const Chains &) const;
  Regions apply(const Region &r) const;
  bool isEmpty() const;

  const std::vector<Chain> &getChainUnion() const { return chain_union; }

private:
  // TODO : consider rather a vector of lists
  std::vector<Chain> chain_union;
};

std::ostream &operator<<(std::ostream &, const Chains &);

} // namespace view
} // namespace popart

#endif
