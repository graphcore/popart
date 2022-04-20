// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>
#include <popart/chains.hpp>
#include <popart/error.hpp>

#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/region.hpp"

namespace popart {
namespace view {

bool Chains::isEmpty() const {
  for (const auto &c : chain_union) {
    if (!c.untraversable()) {
      return false;
    }
  }
  return true;
}

bool Chain::untraversable() const {
  if (links.size() == 0) {
    throw error(
        "call to untraversable with empty links, links should never be empty");
  }
  auto regions = apply(links.front().getFilter());
  return std::all_of(regions.begin(), regions.end(), [](const Region &r) {
    return r.isEmpty();
  });
}

void Link::append(std::ostream &ost) const {
  ost << debugName << " with filter ";
  filter.append(ost);
}

std::ostream &operator<<(std::ostream &ost, const Link &l) {
  l.append(ost);
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const Chain &c) {
  ost << "Chain:\n";
  for (auto l : c.getLinks()) {
    ost << "" << l << '\n';
  }
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const Chains &cs) {
  ost << "Chains:\n";
  for (auto c : cs.getChainUnion()) {
    ost << c << '\n';
  }
  return ost;
}

Chains Chains::getIdentity(const Region &f) {
  return Chains({Chain::getIdentity(f)});
}

Chains Chains::getIdentity(const Shape &s) {
  return Chains::getIdentity(Region::getFull(s));
}

Chain Chain::getIdentity(const Region &f) {
  return Chain(Link::getIdentity(f));
}

Link Link::getIdentity(const Region &f) {
  return Link(
      f, [](const Region &r) { return view::Regions(1, r); }, "Identity Link");
}

Link::Link(const Region &filt, const RegMap &rm)
    : Link(filt, rm, "Nameless Link") {}

Link::Link(const Region &filt, const RegMap &rm, const std::string &dbName)
    : filter(filt), regmap(rm), debugName(dbName) {
  // Filter should not modify access type
  filter.setAccessType(AccessType::None);
}

bool Link::contains(const Link &rhs) const {
  for (auto r0 : rhs.apply(rhs.getFilter())) {
    bool contained = false;
    for (auto r1 : apply(filter)) {
      contained |= r1.contains(r0);
      if (contained)
        break;
    }
    if (!contained)
      return false;
  }
  return filter.contains(rhs.getFilter());
}

Chains::Chains(const Link &link) { chain_union.push_back({link}); }

void Chain::append(const Chain &tail) {
  links.insert(links.end(), tail.links.begin(), tail.links.end());
}

namespace {

Chains filter(std::vector<Chain> chains) {
  std::vector<Chain> new_chain_union;
  for (auto &c : chains) {
    bool contains = false;
    for (int64_t i = 0; i < new_chain_union.size(); ++i) {
      if (new_chain_union[i].contains(c)) {
        contains = true;
        break;
      } else if (c.contains(new_chain_union[i])) {
        new_chain_union[i] = c;
        contains           = true;
        break;
      }
    }
    if (!contains)
      new_chain_union.push_back(c);
  }
  return Chains(new_chain_union);
}

} // namespace

Chains Chains::series(const Chains &tail) const {
  std::vector<Chain> new_chain_union;
  for (const auto &chain0 : chain_union) {
    for (const auto &chain1 : tail.chain_union) {
      Chain newChain = chain0;
      newChain.append(chain1);
      // check if anything can get through this chain
      if (!newChain.untraversable()) {
        new_chain_union.push_back(newChain);
      }
    }
  }
  return filter(new_chain_union);
}

Chains::Chains(const std::vector<Chain> &chains) : chain_union(chains) {}

Chains Chains::parallel(const Chains &chains) const {
  std::vector<Chain> new_chain_union;
  new_chain_union.insert(
      new_chain_union.end(), chain_union.begin(), chain_union.end());
  new_chain_union.insert(new_chain_union.end(),
                         chains.chain_union.begin(),
                         chains.chain_union.end());
  auto filtered = filter(new_chain_union);
  return filtered;
}

Regions Chain::apply(const Region &regIn) const {
  Regions currentRegions(1, regIn);
  Regions nextRegions;

  for (const Link &link : links) {
    for (auto r0 : currentRegions) {
      auto regions = link.apply(r0);
      nextRegions.insert(nextRegions.end(), regions.begin(), regions.end());
    }
    currentRegions = mergeRegions(nextRegions);
    nextRegions.clear();
    if (std::all_of(currentRegions.begin(),
                    currentRegions.end(),
                    [](const Region &r) { return r.isEmpty(); })) {
      if (currentRegions.size() > 0) {
        return {currentRegions.front()};
      } else {
        return {};
      }
    }
  }
  return currentRegions;
}

bool Chain::contains(const Chain &rhs) const {
  auto r0s = apply(getLinks().front().getFilter());
  auto r1s = rhs.apply(rhs.getLinks().front().getFilter());

  if (r0s.size() == 0 && r1s.size() == 0) {
    return true;
  } else if (r0s.size() == 0) {
    return false;
  } else if (r1s.size() == 0) {
    return true;
  }

  std::sort(r0s.begin(),
            r0s.end(),
            [](const view::Region &a, const view::Region &b) -> bool {
              return a.getLower() < b.getLower();
            });

  std::vector<std::set<int64_t>> cuts(r0s[0].rank());
  for (auto &r : r0s) {
    for (int64_t i = 0; i < r.rank(); ++i) {
      cuts[i].insert(r.getLower()[i]);
      cuts[i].insert(r.getUpper()[i]);
    }
  }

  Regions rs;
  rs.insert(rs.end(), r0s.begin(), r0s.end());
  for (auto &r : r1s) {
    auto rc = r.cut(cuts);
    rs.insert(rs.end(), rc.begin(), rc.end());
  }
  rs = mergeRegions(rs);

  std::sort(rs.begin(),
            rs.end(),
            [](const view::Region &a, const view::Region &b) -> bool {
              return a.getLower() < b.getLower();
            });

  return rs == r0s && getLinks().front().getFilter().contains(
                          rhs.getLinks().front().getFilter());
}

Regions Chains::apply(const Region &regIn) const {
  Regions regions;
  for (const Chain &chain : chain_union) {
    Regions rs = chain.apply(regIn);
    for (Region r : rs) {
      if (!r.isEmpty()) {
        if (std::find(regions.begin(), regions.end(), r) == regions.end()) {
          regions.push_back(r);
        }
      }
    }
  }
  return regions;
}

} // namespace view
} // namespace popart
