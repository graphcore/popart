// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <poprithms/logging/timepartitionlogger.hpp>
#include <popart/aliases.hpp>
#include <popart/chains.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/pointercomparators.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {

view::Chains Aliases::getChainsFromTo(Tensor *from, Tensor *to) const {
  if (from == to) {
    return view::Chains::getIdentity(from->info.shape());
  }

  if (aliasChainsFromKey.find(from) == aliasChainsFromKey.end()) {
    return view::Chains();
  }

  auto &allChainsFrom = aliasChainsFromKey.at(from);
  if (allChainsFrom.find(to) == allChainsFrom.end()) {
    return view::Chains();
  }
  return allChainsFrom.at(to);
}

std::unordered_map<Tensor *, view::Chains> Aliases::getAliasChains(
    const std::unordered_map<Tensor *,
                             std::unordered_map<Tensor *, view::Chains>> &fullM,
    Tensor *t) const {
  std::unordered_map<Tensor *, view::Chains> retM{};
  auto found = fullM.find(t);
  if (found != fullM.end()) {
    retM = found->second;
  }
  retM[t] = view::Chains::getIdentity(t->info.shape());
  return retM;
}

std::unordered_map<Tensor *, view::Chains>
Aliases::aliasChainsTo(Tensor *to) const {
  return getAliasChains(aliasChainsToKey, to);
}

std::unordered_map<Tensor *, view::Chains>
Aliases::aliasChainsFrom(Tensor *from) const {
  return getAliasChains(aliasChainsFromKey, from);
}

// Regions in "from" aliased "to"
view::Regions Aliases::getAliasRegions(Tensor *from, Tensor *to) const {
  auto aliasedTensorMap = aliasChainsFrom(from);
  auto it               = aliasedTensorMap.find(to);
  if (it == aliasedTensorMap.end()) {
    return view::Regions({view::Region::getEmpty(to->info.rank())});
  } else {
    return it->second.apply(view::Region::getFull(from->info.shape()));
  }
}

void Aliases::clearAliases() {
  aliasChainsFromKey.clear();
  aliasChainsToKey.clear();
}

// Let the Chains flow through
void Aliases::updateAliases(Tensor *t1,
                            Tensor *t2,
                            view::Regions inRegions,
                            view::RegMap fwdMap,
                            view::RegMap bwdMap,
                            std::string fwdLinkName,
                            std::string bwdLinkName) {

  auto scopedStopwatch = t1->getIr().timePartitionLogger().scopedStopwatch(
      "Aliases::updateAliases");
  // Optimisation: If t1 <--> t2 are already fully aliased, then there is no
  // reason to add more aliasing
  if (getAliasRegions(t1, t2).front() ==
          view::Region::getFull(t2->info.shape()) &&
      getAliasRegions(t2, t1).front() ==
          view::Region::getFull(t1->info.shape())) {
    return;
  }

  std::unordered_map<Tensor *, std::unordered_map<Tensor *, view::Chains>>
      newAliases;

  auto registerChains =
      [&newAliases](Tensor *t0, Tensor *t3, const view::Chains &newChains) {
        if (!newChains.isEmpty()) {
          if (newAliases.find(t0) == newAliases.end()) {
            newAliases[t0] = {};
          }
          if (newAliases.at(t0).find(t3) == newAliases.at(t0).end()) {
            newAliases[t0][t3] = {}; // empty Chains
          }
          // add the new Chains
          newAliases[t0][t3] = newAliases[t0][t3].parallel(newChains);
        }
      };

  for (auto inRegion : inRegions) {
    if (inRegion.isEmpty()) {
      continue;
    }

    view::Regions outRegions = fwdMap(inRegion);

    // if there is an alias between the unique output
    // t2 and the input t1, this opens new Chains
    for (auto outRegion : outRegions) {
      if (outRegion.isEmpty()) {
        continue;
      }

      view::Link fwdLink(inRegion, fwdMap, fwdLinkName);
      view::Link bwdLink(outRegion, bwdMap, bwdLinkName);

      // all chains t0 -> t1 for all t0
      auto allInChains = aliasChainsTo(t1);

      // all chains t2 -> t3 for all t3
      auto allOutChains = aliasChainsFrom(t2);

      for (auto &inwards : allInChains) {
        Tensor *t0 = inwards.first;
        // the chains t0 -> t1
        view::Chains inChains      = inwards.second;
        auto inChainsFwdLinkSeries = inChains.series(fwdLink);

        // the chains t1 -> t0. There are such chains,
        // guaranteed by the existence of chains t0 -> t1
        view::Chains inChainsRev = getChainsFromTo(t1, t0);

        for (auto &outwards : allOutChains) {

          Tensor *t3 = outwards.first;

          // the chains t2 -> t3
          view::Chains outChains = outwards.second;

          // the chains t3 -> t2
          // (which must exist by symmetry of aliasing)
          view::Chains outChainsRev = getChainsFromTo(t3, t2);

          // we now have,
          // t0 -----> t1 -> op -> t2 -----> t3
          // and we want to update aliasChainsToKey[t3][t0]
          // with all new chains that pass through op, as
          // well as aliasChainsToKey[t0][t3]

          auto newFwdChains = inChainsFwdLinkSeries.series(outChains);
          auto newBwdChains = outChainsRev.series(bwdLink).series(inChainsRev);

          bool fwdIsEmpty = newFwdChains.isEmpty();
          bool bwdIsEmpty = newBwdChains.isEmpty();

          if (fwdIsEmpty != bwdIsEmpty) {
            std::ostringstream oss;
            oss << "\n\nnewFwdChains : \n" << newFwdChains << '\n';
            oss << "\ninChains : \n" << inChains << '\n';
            oss << "\nfwdLink : \n" << fwdLink << '\n';
            oss << "\noutChains : \n" << outChains << '\n';
            oss << "\nDetermining if newFwdChains is empty" << '\n';
            oss << "\nConclusion, fwdIsEmpty = : " << fwdIsEmpty << '\n';
            oss << "\n\nnewBwdChains : \n" << newBwdChains << '\n';
            oss << "\noutChainsRev : \n" << outChainsRev << '\n';
            oss << "\nbwdLink : \n" << bwdLink << '\n';
            oss << "\ninChainsRev : \n" << inChainsRev << '\n';
            oss << "\nDetermining if newBwdChains is empty" << '\n';
            oss << "\nConclusion, bwdIsEmpty : " << bwdIsEmpty << '\n';
            throw internal_error(oss.str());
          }

          if (!fwdIsEmpty) {
            registerChains(t3, t0, newFwdChains);
          }

          // same logic for t3 -> t0
          if (!bwdIsEmpty) {
            registerChains(t0, t3, newBwdChains);
          }
        }
      }
    }
  }

  for (auto x : newAliases) {
    auto t0         = x.first;
    auto t3_chain_s = x.second;
    if (aliasChainsToKey.find(t0) == aliasChainsToKey.end()) {
      aliasChainsToKey[t0] = {};
    }
    for (auto t3_chain : t3_chain_s) {
      auto t3    = t3_chain.first;
      auto chain = t3_chain.second;
      if (aliasChainsToKey.at(t0).find(t3) == aliasChainsToKey.at(t0).end()) {
        aliasChainsToKey[t0][t3] = {}; // empty Chains
      }
      // add the new Chains
      aliasChainsToKey[t0][t3] = aliasChainsToKey[t0][t3].parallel(chain);
      // insert the mirror image
      aliasChainsFromKey[t3][t0] = aliasChainsToKey[t0][t3];
    }
  }
}

void Aliases::addAllAliases(const Aliases &other) {
  for (auto &kv0 : other.aliasChainsToKey) {
    for (auto &kv1 : kv0.second) {
      aliasChainsToKey[kv0.first][kv1.first] =
          aliasChainsToKey[kv0.first][kv1.first].parallel(kv1.second);
    }
  }
  for (auto &kv0 : other.aliasChainsFromKey) {
    for (auto &kv1 : kv0.second) {
      aliasChainsFromKey[kv0.first][kv1.first] =
          aliasChainsFromKey[kv0.first][kv1.first].parallel(kv1.second);
    }
  }
}

std::set<Tensor *, PTensorCmp> Aliases::getTensors() const {
  std::set<Tensor *, PTensorCmp> tensors;

  for (auto &kv : aliasChainsFromKey) {
    tensors.insert(kv.first);
  }

  return tensors;
}

} // namespace popart
