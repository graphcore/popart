// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/ir.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/poptensors.hpp>

namespace popart {
namespace popx {

PopTensors::PopTensors(const Ir &ir_) : ir(ir_) {}

void PopTensors::verify(TensorId id, const snap::Tensor &pt) {
  auto found             = tensors_.find(id);
  auto foundViewChangers = viewChangers_.find(id);

  if (found != tensors_.end()) {
    throw internal_error("snap::Tensor " + id + " already in map");
  }

  if (!ir.containsTensor(id)) {
    throw internal_error(
        "no tensor named {} in ir, is this a valid snap::Tensor?", id);
  }

  // confirm shapes agree (up to squeezing out the extra 1s)
  auto irTensor = ir.getTensor(id);

  auto shape =
      foundViewChangers == viewChangers_.end()
          ? pt.getPoplarTensor().shape()
          : foundViewChangers->second->apply(pt).getPoplarTensor().shape();

  if (shape != irTensor->info.shape_szt()) {

    // squeeze out extra 1s
    while (!shape.empty() && shape[0] == 1) {
      shape.erase(shape.begin());
    }

    if (shape != irTensor->info.shape_szt()) {
      std::stringstream ss;
      ss << "snap::Tensor " << id << " of unexpected shape. "
         << "Poplar tensor shape: " << shape
         << ". Expected (Ir) tensor shape: " << irTensor->info.shape_szt()
         << ". This for tensor " << irTensor->str();
      throw error(ss.str());
    }
  }

  auto dtype = ir.getTensor(id)->info.dataType();

  if (ir.getSessionOptions().enableSupportedDataTypeCasting) {
    dtype = getCompatibleDataType(dtype);
  }

  // confirm types agree
  auto expectedType = popType(dtype);

  if (pt.getPoplarTensor().elementType() != expectedType) {
    std::stringstream ss;
    ss << "snap::Tensor " << id << " of unexpected Type. "
       << "Poplar tensor type : " << pt.getPoplarTensor().elementType();
    ss << ". Expected (Ir) tensor type : " << expectedType;
    ss << ". This for tensor " << irTensor->str();
    throw error(ss.str());
  }
}

void PopTensors::insert(TensorId id, const snap::Tensor &pt) {
  verify(id, pt);

  tensors_[id] = std::make_shared<snap::Tensor>(pt);

  auto foundViewChangers = viewChangers_.find(id);
  if (foundViewChangers != viewChangers_.end()) {
    views_[id] =
        std::make_shared<snap::Tensor>(foundViewChangers->second->apply(pt));
  }
}

bool PopTensors::canAlias(TensorId id) const {
  return get(id).getPoplarTensor().isParallelWriteable();
}

void PopTensors::insertAliased(TensorId to, TensorId from) {
  std::shared_ptr<snap::Tensor> pt = tensors_.at(from);
  auto foundView                   = views_.find(from);
  if (foundView != views_.end()) {
    views_[to]        = foundView->second;
    viewChangers_[to] = viewChangers_[from];
  }
  verify(to, *pt);
  tensors_[to] = tensors_.at(from);
}

void PopTensors::insertUnsafe(TensorId id, const snap::Tensor &pt) {
  auto found = tensors_.find(id);
  if (found != tensors_.end()) {
    throw internal_error("snap::Tensor " + id + " already in map");
  }

  tensors_[id] = std::make_shared<snap::Tensor>(pt);
}

bool PopTensors::contains(TensorId id) const {
  return tensors_.find(id) != tensors_.end();
}

const snap::Tensor &PopTensors::get(TensorId id) const {
  auto found = tensors_.find(id);
  if (found == tensors_.end()) {
    throw error("no snap::Tensor " + id);
  }
  return *found->second;
}

const snap::Tensor &PopTensors::getView(TensorId id) const {
  auto found = tensors_.find(id);
  if (found == tensors_.end()) {
    throw error("no snap::Tensor " + id);
  }
  auto foundView = views_.find(id);
  if (foundView == views_.end()) {
    return *found->second;
  } else {
    return *foundView->second;
  }
}

bool PopTensors::hasViewChangers(TensorId id) const {
  auto foundViewChangers = viewChangers_.find(id);
  return foundViewChangers != viewChangers_.end();
}

const ViewChangers &PopTensors::getViewChangers(TensorId id) {
  auto foundViewChangers = viewChangers_.find(id);
  if (foundViewChangers == viewChangers_.end()) {
    throw error("no ViewChangers " + id);
  } else {
    return *foundViewChangers->second;
  }
}

void PopTensors::setViewChangers(TensorId id,
                                 const ViewChangers &viewChangers) {
  viewChangers_[id] = std::make_shared<ViewChangers>(viewChangers);
}

const std::map<TensorId, std::shared_ptr<snap::Tensor>> &
PopTensors::getTensors() const {
  return tensors_;
}

} // namespace popx
} // namespace popart
