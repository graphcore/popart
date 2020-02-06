#include <popart/ir.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/poptensors.hpp>

namespace popart {
namespace popx {

PopTensors::PopTensors(const Ir &ir_) : ir(ir_) {}

void PopTensors::insert(TensorId id, const poplar::Tensor &pt) {
  auto found = tensors_.find(id);
  if (found != tensors_.end()) {
    throw internal_error("poplar::Tensor " + id + " already in map");
  }

  if (!ir.containsTensor(id)) {
    throw internal_error(
        "no tensor named {} in ir, is this a valid poplar::Tensor?", id);
  }

  // confirm shapes agree (up to squeezing out the extra 1s)
  auto irTensor = ir.getTensor(id);
  auto shape    = pt.shape();

  if (pt.shape() != irTensor->info.shape_szt()) {

    // squeeze out extra 1s
    while (!shape.empty() && shape[0] == 1) {
      shape.erase(shape.begin());
    }

    if (shape != irTensor->info.shape_szt()) {
      std::stringstream ss;
      ss << "poplar::Tensor " << id << " of unexpected shape. "
         << "Poplar tensor shape: " << pt.shape()
         << ". Expected (Ir) tensor shape: " << irTensor->info.shape_szt()
         << ". This for tensor " << irTensor->str();
      throw error(ss.str());
    }
  }

  // confirm types agree
  auto expectedType = popType(ir.getTensor(id)->info);
  if (pt.elementType() != expectedType) {
    std::stringstream ss;
    ss << "poplar::Tensor " << id << " of unexpected Type. "
       << "Poplar tensor type : " << pt.elementType();
    ss << ". Expected (Ir) tensor type : " << expectedType;
    ss << ". This for tensor " << irTensor->str();
    throw error(ss.str());
  }

  tensors_[id] = pt;
}

void PopTensors::insertUnsafe(TensorId id, const poplar::Tensor &pt) {
  auto found = tensors_.find(id);
  if (found != tensors_.end()) {
    throw internal_error("poplar::Tensor " + id + " already in map");
  }

  tensors_[id] = pt;
}

bool PopTensors::contains(TensorId id) const {
  return tensors_.find(id) != tensors_.end();
}

const poplar::Tensor &PopTensors::get(TensorId id) const {
  auto found = tensors_.find(id);
  if (found == tensors_.end()) {
    throw error("no poplar::Tensor " + id);
  }
  return found->second;
}

const std::map<TensorId, poplar::Tensor> &PopTensors::getTensors() const {
  return tensors_;
}

} // namespace popx
} // namespace popart
