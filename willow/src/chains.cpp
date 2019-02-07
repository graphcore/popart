#include <algorithm>
#include <poponnx/chains.hpp>
#include <poponnx/error.hpp>

namespace poponnx {
namespace view {

bool Chain::untraversable() const {
  if (links.size() == 0) {
    throw error(
        "call to untraversable with empty links, links should never be empty");
  }
  return apply(links[0].getFilter()).isEmpty();
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
  return Link(f, [](const Region &r) { return r; });
}

Link::Link(const Region &filt, const RegMap &rm) : filter(filt), regmap(rm) {}

Chains::Chains(const Link &link) { chain_union.push_back({link}); }

void Chain::append(const Chain &tail) {
  links.insert(links.end(), tail.links.begin(), tail.links.end());
}

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

  // TODO : more filtering of duplicate / subregion chains, ids, etc (T6707)

  return Chains(new_chain_union);
}

Chains::Chains(const std::vector<Chain> &chains) : chain_union(chains) {}

Chains Chains::parallel(const Chains &chains) const {
  std::vector<Chain> new_chain_union = chain_union;
  for (auto &chain : chains.chain_union) {
    new_chain_union.push_back(chain);
  }

  // TODO : more checks for duplicates (T6707)
  return Chains(new_chain_union);
}

Region Chain::apply(const Region &regIn) const {
  Region r = regIn;
  for (const Link &link : links) {
    r = link.apply(r);
    if (r.isEmpty()) {
      return r;
    }
  }
  return r;
}

Regions Chains::apply(const Region &regIn) const {
  Regions regions;
  for (const Chain &chain : chain_union) {
    Region r = chain.apply(regIn);
    if (!r.isEmpty()) {
      if (std::find(regions.begin(), regions.end(), r) == regions.end()) {
        regions.push_back(r);
      }
    }
  }
  return regions;
}

} // namespace view
} // namespace poponnx
