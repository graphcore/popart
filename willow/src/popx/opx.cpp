#include <poponnx/conv.hpp>
#include <poponnx/error.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/opx.hpp>
#include <poponnx/tensor.hpp>

namespace willow {
namespace popx {

Opx::Opx(Op *op_p_, Devicex *dv_p_) : op_p(op_p_), dv_p(dv_p_) {}

Opx::~Opx() = default;

poplar::Tensor Opx::createInput(int) const {
  throw error("Opx for " + op_p->op_type() + " cannot create Input");
}

bool Opx::createsEquiv(int, Opx *, int) const {
  throw error("No check for equivalent tensor create for type " +
              op_p->op_type());
}

std::vector<TensorId> Opx::mustExistBeforeCreate(int index0) const {
  throw error(
      "Opx for " + op_p->op_type() +
      " cannot say which poplar Tensors must exist to create at index " +
      std::to_string(index0));
}

void Opx::grow(poplar::program::Sequence &) const {
  throw error("adding poplar::Tensors not implemented for " + op_p->op_type());
}

bool Opx::canCreateInput(int) const { return false; }

poplar::Graph &Opx::graph() const { return dv_p->graph(); }

const poplar::Tensor &Opx::get(TensorId id) const {
  return dv_p->tensors.get(id);
}

void Opx::insert(TensorId id, const poplar::Tensor &tensor) const {
  dv_p->tensors.insert(id, tensor);
}

TensorId Opx::inId(int index) const { return op_p->input.id(index); }

TensorId Opx::outId(int index) const { return op_p->output.id(index); }

const TensorInfo &Opx::inInfo(int index) const {
  return op_p->input.tensor(index)->info;
}

const std::vector<int64_t> &Opx::inShape(int index) const {
  return inInfo(index).shape();
}

const TensorInfo &Opx::outInfo(int index) const {
  return op_p->output.tensor(index)->info;
}

const std::vector<int64_t> &Opx::outShape(int index) const {
  return outInfo(index).shape();
}

std::string Opx::idStr() const { return std::to_string(op_p->id); }

poplar::Tensor Opx::cloneNcopy(poplar::program::Sequence &prog,
                               TensorId id) const {
  auto outTensor = graph().clone(get(id));
  poplar::program::Copy copyProg(get(id), outTensor);
  prog.add(copyProg);
  return outTensor;
}

} // namespace popx
} // namespace willow
