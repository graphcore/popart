// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <snap/poputil/TileMapping.hpp>
#include <popart/popx/creatorx.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

namespace popart {
namespace popx {

namespace {

TensorRegions fragment(const view::Region fullRegion,
                       const TensorRegions &tensorRegions) {

  std::vector<std::set<int64_t>> cuts(fullRegion.rank());

  for (auto &tr : tensorRegions) {
    for (int64_t i = 0; i < fullRegion.rank(); ++i) {
      cuts[i].insert(tr.region.getLower()[i]);
      cuts[i].insert(tr.region.getUpper()[i]);
    }
  }

  TensorRegions allTensorRegions;

  for (auto &tr : tensorRegions) {
    view::Regions cutRegions = tr.region.cut(cuts);
    for (auto &r : cutRegions) {
      allTensorRegions.push_back({tr.offset, r, tr.tensor});
    }
  }

  return allTensorRegions;
}

snap::Tensor compose(const TensorRegions &tensorRegions,
                     const view::Region fullRegion,
                     snap::Tensor fullTensor) {
  TensorRegions currentTensorRegions = tensorRegions;

  logging::devicex::trace("[creatorx] Full region {} {}",
                          fullRegion.getLower(),
                          fullRegion.getUpper());

  view::Regions regions;
  for (auto &tensorRegion : tensorRegions) {
    regions.push_back(tensorRegion.region);
    logging::devicex::trace("[creatorx] Tensor region {} {}",
                            regions.back().getLower(),
                            regions.back().getUpper());
  }

  // Regions for which a linear mapping is chosen
  view::Regions linearRegions = fullRegion.sub(regions);

  for (view::Region region : linearRegions) {
    logging::devicex::trace("[creatorx] Adding linear region {} {}",
                            region.getLower(),
                            region.getUpper());
    currentTensorRegions.emplace_back(fullRegion, region, fullTensor);
  }

  // At this point, currentTensorRegions should contain enough regions to
  // compose the tensor fully.

  // Fragment into concat-enable pieces
  currentTensorRegions = fragment(fullRegion, currentTensorRegions);

  // Cut pieces of the tensors that we want to keep for the final tensor
  for (int64_t i = 0; i < currentTensorRegions.size(); ++i) {
    auto offset = currentTensorRegions[i].offset;
    auto region = currentTensorRegions[i].region;
    auto tensor = currentTensorRegions[i].tensor;
    // Tensor can either have the same shape as the region, or the full size
    // if it is the full size, cut down to relevant region size
    if (tensor.numElements() > region.nelms()) {
      std::vector<size_t> l(region.rank());
      std::vector<size_t> u(region.rank());
      std::vector<size_t> o(region.rank());
      l.assign(region.getLower().begin(), region.getLower().end());
      u.assign(region.getUpper().begin(), region.getUpper().end());
      o.assign(offset.getLower().begin(), offset.getLower().end());

      std::transform(
          l.begin(), l.end(), o.begin(), l.begin(), std::minus<size_t>());
      std::transform(
          u.begin(), u.end(), o.begin(), u.begin(), std::minus<size_t>());

      tensor = tensor.slice(poplar::ArrayRef<size_t>(l),
                            poplar::ArrayRef<size_t>(u));
    }

    logging::trace("[creatorx] Tensor shape {} region {} (full region: {})",
                   tensor.shape(),
                   region,
                   fullRegion);

    currentTensorRegions[i] = TensorRegion(offset, region, tensor);
  }

  std::sort(currentTensorRegions.begin(),
            currentTensorRegions.end(),
            [](const TensorRegion &a, const TensorRegion &b) -> bool {
              return a.region.getLower() < b.region.getLower();
            });

  TensorRegions nextTensorRegions;

  // Merge along dimensions
  for (int64_t d = fullRegion.rank() - 1; d >= 0; --d) {
    nonstd::optional<view::Region> r;
    nonstd::optional<snap::Tensor> t;
    for (auto &tensorRegion : currentTensorRegions) {
      std::pair<int64_t, view::Region> rd(-1, view::Region({}, {}));
      if (r.has_value()) {
        rd = r.value().merge(tensorRegion.region);
      }
      if (rd.first != d) {
        // Can't merge regions directly
        if (r.has_value() && t.has_value()) {
          // Push back last region
          nextTensorRegions.emplace_back(fullRegion, r.value(), t.value());
        }
        // Load next region
        r = tensorRegion.region;
        t = tensorRegion.tensor;
      } else {
        // Merge region & concatenate tensor if possible
        if (rd.first == d) {
          t = snap::Tensor{poplar::concat(t->getPoplarTensor(),
                                          tensorRegion.tensor.getPoplarTensor(),
                                          static_cast<uint32_t>(d)),
                           t.value()};
          r = rd.second;
        }
      }
    }
    if (t.has_value() && r.has_value()) {
      nextTensorRegions.emplace_back(fullRegion, r.value(), t.value());
    }
    currentTensorRegions = nextTensorRegions;
    nextTensorRegions.clear();
  }

  return currentTensorRegions.front().tensor;
}

} // namespace

ICreatorCandidate::ICreatorCandidate() {}

bool ICreatorCandidate::greaterThan(ICreatorCandidatePtr icc1,
                                    ICreatorCandidatePtr icc2) {
  return std::tuple<double, int64_t, int64_t>(icc1->getMaxCreatorPriority(),
                                              icc1->getNumElems(),
                                              icc2->getScheduleIndex()) >
         std::tuple<double, int64_t, int64_t>(icc2->getMaxCreatorPriority(),
                                              icc2->getNumElems(),
                                              icc1->getScheduleIndex());
}

InputCreatorCandidate::InputCreatorCandidate(
    InIndex index_,
    const PopOpx *opx_,
    std::vector<OpxInAndOutIndex> pathFromInput_,
    int64_t scheduleIndex_)
    : index(index_), opx(opx_), scheduleIndex(scheduleIndex_) {

  pathFromInput.reserve(pathFromInput_.size());
  for (OpxInAndOutIndex &pathElem : pathFromInput_) {
    if (!pathElem.isDelegate) {
      pathFromInput.push_back(pathElem);
    }
  }
}

double InputCreatorCandidate::getMaxCreatorPriority() {
  return getOpx()->inputCreatorPriority;
}

int64_t InputCreatorCandidate::getNumElems() {
  int64_t n = 0;
  for (auto &r : unwind()) {
    n += r.nelms();
  }
  return n;
}

view::Regions InputCreatorCandidate::unwind() {
  return unwind(view::Region::getFull(opx->inShape(index)));
}

view::Regions InputCreatorCandidate::unwind(popart::view::Region region) {
  auto pathToInput = pathFromInput;
  std::reverse(pathToInput.begin(), pathToInput.end());
  view::Regions rqueue(1, region);
  view::Regions wqueue;
  for (auto &opxOnPath : pathToInput) {
    for (auto &r0 : rqueue) {
      auto regions = opxOnPath.opx->unwindRegion(opxOnPath.inIndex,
                                                 opxOnPath.outIndex)(r0);
      wqueue.insert(wqueue.end(), regions.begin(), regions.end());
    }
    rqueue = wqueue;
    wqueue.clear();
  }
  return rqueue;
}

std::pair<snap::Tensor, ViewChangers>
InputCreatorCandidate::unwindOnPath(const OpxInAndOutIndex &opxOnPath,
                                    const snap::Tensor &outTensor,
                                    const view::Regions &outRegions,
                                    view::Regions &inRegions) {
  logging::devicex::trace("[creatorx] Unwinding at {}",
                          opxOnPath.opx->getOp<Op>().debugName());

  // All efficiently created regions
  for (auto outRegion : outRegions) {
    auto rs = opxOnPath.opx->unwindRegion(opxOnPath.inIndex,
                                          opxOnPath.outIndex)(outRegion);
    for (auto &r : rs) {
      inRegions.push_back(r);
    }
  }

  // Unwound tensor
  auto inTensor = opxOnPath.opx->unwindTensorLayout(
      outTensor, opxOnPath.inIndex, opxOnPath.outIndex);

  if (opxOnPath.opx->hasCreatorViewChangers(opxOnPath.inIndex)) {
    // Early stop unwinding; tensor has view change
    logging::devicex::debug(
        "[creatorx] Early stopping unwinding due to view-changing at Op {}",
        opxOnPath.opx->getOp<Op>().debugName());
    return {inTensor, opxOnPath.opx->getCreatorViewChangers(opxOnPath.inIndex)};
  }

  auto outShape = opxOnPath.opx->getOp<Op>()
                      .output->tensor(opxOnPath.outIndex)
                      ->info.shape();
  auto inShape =
      opxOnPath.opx->getOp<Op>().input->tensor(opxOnPath.inIndex)->info.shape();

  // The offset at which the unwound tensor represents a part of the input
  // region
  auto offsetRegion = opxOnPath.opx->unwindRegion(
      opxOnPath.inIndex, opxOnPath.outIndex)(view::Region::getFull(outShape));
  auto fullRegion = view::Region::getFull(inShape);

  logging::devicex::trace("[creatorx] Expected (in)shape {}", inShape);

  auto inInfo = opxOnPath.opx->getOp<Op>().inInfo(opxOnPath.inIndex);

  auto &graph     = opxOnPath.opx->srcVirtualGraph(opxOnPath.inIndex);
  auto fullTensor = graph.addVariable(popType(inInfo), inInfo.shape_szt(), "");

  // Map it linearly
  snap::poputil::mapTensorLinearly(graph, fullTensor);

  logging::devicex::trace("[creatorx] Tensor shape before compose: {}",
                          inTensor.shape());

  TensorRegions tensorRegions;
  tensorRegions.reserve(inRegions.size());
  for (auto &tRegion : inRegions) {
    tensorRegions.emplace_back(
        view::regionBounds(offsetRegion), tRegion, inTensor);
  }

  // Compose a tensor of fullRegion shape, using as many of the tensorRegions
  // as necessary, and filling in missing pieces by taking them
  // from the linearly created fullTensor.
  inTensor = compose(tensorRegions, fullRegion, fullTensor);

  logging::devicex::trace("[creatorx] Tensor shape after compose {}",
                          inTensor.shape());

  logging::devicex::trace("[creatorx] Tensor shape after unwind: {}",
                          inTensor.shape());
  return {inTensor, ViewChangers()};
}

std::pair<snap::Tensor, ViewChangers>
InputCreatorCandidate::unwind(snap::Tensor input) {

  // Reverse the path,
  // The first element is now the Opx producing a tensor consumed by
  // the candidate.
  // The last element is now the Opx consuming the input we are mapping.

  auto pathToInput = getPathsFromInput().front();
  std::reverse(pathToInput.begin(), pathToInput.end());

  auto region = view::Region::getFull(opx->inShape(index));

  view::Regions outRegions = {region};

  snap::Tensor output;
  for (auto &opxOnPath : pathToInput) {
    view::Regions inRegions;
    auto out = unwindOnPath(opxOnPath, input, outRegions, inRegions);
    if (out.second != ViewChangers()) {
      // Early stop unwinding; tensor has view change
      return out;
    } else {
      // Continue
      input = out.first;
    }
    outRegions = inRegions;
  }

  return {input, ViewChangers()};
}

std::pair<snap::Tensor, ViewChangers>
InputCreatorCandidate::createInput(const poplar::DebugNameAndId &dnai) {
  snap::Tensor t = getOpx()->createInputTensor(getIndex(), dnai);
  if (getOpx()->hasCreatorViewChangers(getIndex())) {
    return {t, getOpx()->getCreatorViewChangers(getIndex())};
  }
  return unwind(t);
}

DnfTensorIds InputCreatorCandidate::mustExistBeforeCreate() {
  return getOpx()->mustExistBeforeCreateDNF(getIndex());
}

std::string InputCreatorCandidate::str() {

  std::string result = getOpx()->op_p->str();

  auto pathToInput = pathFromInput;
  std::reverse(pathToInput.begin(), pathToInput.end());

  result += "(";
  for (auto &i : pathToInput) {
    result += " -> " + i.opx->op_p->str() + " ";
    result += "[" + i.opx->op_p->output->id(i.outIndex);
    result += "->" + i.opx->op_p->input->id(i.inIndex) + "]";
  }
  result += ")";

  return result;
}

InputMultiCreatorCandidate::InputMultiCreatorCandidate()
    : ICreatorCandidate() {}

double InputMultiCreatorCandidate::getMaxCreatorPriority() {
  double priority = std::numeric_limits<double>::lowest();
  for (auto &candidate : candidates) {
    priority = std::max(candidate.first->getMaxCreatorPriority(), priority);
  }
  return priority;
}

view::Regions InputMultiCreatorCandidate::unwind() {
  throw("Not expected to unwind on InputMultiCreatorCandidate");
}

view::Regions InputMultiCreatorCandidate::unwind(popart::view::Region) {
  throw("Not expected to unwind on InputMultiCreatorCandidate");
}

std::pair<snap::Tensor, ViewChangers>
InputMultiCreatorCandidate::unwind(snap::Tensor) {
  throw("Not expected to unwind on InputMultiCreatorCandidate");
}

DnfTensorIds InputMultiCreatorCandidate::mustExistBeforeCreate() {
  DnfTensorIds cumulativeTensorIds;
  for (auto &candidate : candidates) {
    if (!candidate.first->mustExistBeforeCreate().empty()) {
      if (cumulativeTensorIds.empty()) {
        cumulativeTensorIds = candidate.first->mustExistBeforeCreate();
      } else {
        // Distribute DNF over DNF
        DnfTensorIds newCumulativeTensorIds;
        for (auto tensorIds0 : cumulativeTensorIds) {
          for (auto tensorIds1 : candidate.first->mustExistBeforeCreate()) {
            std::set<TensorId> tensorIds2 = tensorIds0;
            tensorIds2.insert(tensorIds1.begin(), tensorIds1.end());
            newCumulativeTensorIds.push_back(tensorIds2);
          }
        }
        cumulativeTensorIds = newCumulativeTensorIds;
      }
    }
  }
  return cumulativeTensorIds;
}

int64_t InputMultiCreatorCandidate::getScheduleIndex() const {
  int64_t index = 0;
  for (auto &candidate : candidates) {
    index = std::max(index, candidate.first->getScheduleIndex());
  }
  return index;
}

int64_t InputMultiCreatorCandidate::getNumElems() {
  int64_t elems = 0;
  // Loop over all candidates
  for (auto &candidate : candidates) {
    // Loop over all accepted regions
    for (auto region : candidate.second) {
      elems += region.nelms();
    }
  }
  return elems;
}

// Create tensor by composing parts created by candidates
std::pair<snap::Tensor, ViewChangers>
InputMultiCreatorCandidate::createInput(const poplar::DebugNameAndId &dnai) {
  auto candidateIdx = 0;

  TensorRegions currentTensorRegions;

  for (auto &candidate : candidates) {
    auto tensorAndView = candidate.first->createInput(
        {dnai, logging::format("fragment/{}", std::to_string(candidateIdx))});
    auto tensor = tensorAndView.first;
    logging::devicex::trace("Accepted candidate regions: {}, tensor shape: {}",
                            candidate.second,
                            tensor.shape());
    for (auto acceptedRegion : candidate.second)
      currentTensorRegions.push_back(
          {view::Region::getFull(
               vector_cast<Shape::value_type>(tensor.shape())),
           acceptedRegion,
           tensor});
    ++candidateIdx;
  }

  // Fallback linearly mapped tensor, inferred from first candidate
  auto popShape = currentTensorRegions.front().tensor.shape();
  std::vector<int64_t> shape(popShape.size());
  shape.assign(popShape.begin(), popShape.end());
  auto fullRegion = view::Region::getFull(shape);
  auto fullTensor = currentTensorRegions.front().tensor;

  return {compose(currentTensorRegions, fullRegion, fullTensor),
          ViewChangers()};
}

std::string InputMultiCreatorCandidate::str() {
  std::stringstream ss;
  ss << "[" << std::endl;
  for (auto candidate : candidates) {
    ss << candidate.first->str() << std::endl;
  }
  ss << "]";
  return ss.str();
}

view::Regions
InputMultiCreatorCandidate::getAcceptedSubregions(view::Region other) {
  view::Regions remainder = {};
  view::Regions stack     = {other};
  for (auto &candidate : candidates) {
    // Loop over accepted regions for each creator
    for (auto region : candidate.second) {
      while (stack.size() > 0) {
        view::Region r0 = stack.back();
        stack.pop_back();
        view::Regions tmp = r0.sub(region);
        remainder.insert(remainder.end(), tmp.begin(), tmp.end());
      }
      stack = remainder;
      remainder.clear();
    }
  }
  return stack;
}

bool InputMultiCreatorCandidate::addCreatorCandidate(
    ICreatorCandidatePtr candidate) {
  view::Regions acceptedRegions;
  view::Regions regions = candidate->unwind();
  for (auto &r : regions) {
    auto acceptedSubregions = getAcceptedSubregions(r);
    acceptedRegions.insert(acceptedRegions.end(),
                           acceptedSubregions.begin(),
                           acceptedSubregions.end());
  }
  if (acceptedRegions.size() > 0) {
    candidates.insert({candidate, acceptedRegions});
    return true;
  }
  return false;
}

std::vector<std::vector<OpxInAndOutIndex>>
InputMultiCreatorCandidate::getPathsFromInput() {
  std::vector<std::vector<OpxInAndOutIndex>> paths;
  for (auto &candidate : candidates) {
    for (auto &path : candidate.first->getPathsFromInput()) {
      paths.push_back(path);
    }
  }
  return paths;
}

} // namespace popx
} // namespace popart
