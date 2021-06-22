// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/remote.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/irlowering.hpp>
#include <popart/popx/op/remotex.hpp>
#include <popart/popx/opxmanager.hpp>

namespace popart {
namespace popx {

RemoteBaseOpx::RemoteBaseOpx(Op *op, Devicex *devicex) : PopOpx(op, devicex) {}

void RemoteBaseOpx::postLoad(poplar::program::Sequence &prog,
                             RemoteBufferId rbid,
                             const poplar::Tensor t) const {
  auto buffer             = dv_p->lowering().getRemoteBuffer(rbid);
  poplar::Tensor rbTensor = buffer.second.value();
  poplar::program::Copy tmp_copy_prog(rbTensor, t, false, debugContext());
  prog.add(tmp_copy_prog);
}

void RemoteBaseOpx::preStore(snap::Graph &sgraph,
                             poplar::program::Sequence &prog,
                             RemoteBufferId rbid,
                             const poplar::Tensor t) const {
  poplar::Tensor rbTensor;
  if (!dv_p->lowering().hasRemoteBuffer(rbid)) {
    rbTensor = sgraph.getPoplarGraph().clone(
        t,
        debugContext(dv_p->lowering().getRemoteBufferName(rbid)),
        poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
    dv_p->lowering().createRemoteBuffer(rbid, rbTensor);
  }
  auto buffer = dv_p->lowering().getRemoteBuffer(rbid);
  rbTensor    = buffer.second.value();
  poplar::program::Copy tmp_copy_prog(t, rbTensor, false, debugContext());
  prog.add(tmp_copy_prog);
}

poplar::Tensor RemoteBaseOpx::makeWritable(snap::Graph &sgraph,
                                           poplar::Tensor t,
                                           RemoteBufferId rbid,
                                           TensorId id) const {
  poplar::Tensor rbTensor;
  if (!dv_p->lowering().hasRemoteBuffer(rbid)) {
    rbTensor = sgraph.getPoplarGraph().clone(
        t,
        debugContext(dv_p->lowering().getRemoteBufferName(rbid)),
        poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
    dv_p->lowering().createRemoteBuffer(rbid, rbTensor);
  }
  auto buffer = dv_p->lowering().getRemoteBuffer(rbid);
  rbTensor    = buffer.second.value();
  if (!t.isParallelWriteable() || t.containsConstant()) {
    logging::opx::warn("Tensor {} is not a writable remote buffer "
                       "copy target, cloning. "
                       "The aliasing properties have changed implicitly.",
                       id);
    poplar::Tensor tw = sgraph.getPoplarGraph().clone(
        rbTensor,
        debugContext(id + "_Writable"),
        poplar::TensorCloneMethod::PRESERVE_ORDER_AND_ALIASES);
    return tw;
  }
  return t;
}

void RemoteBaseOpx::load(snap::Graph &sgraph,
                         poplar::program::Sequence &prog,
                         RemoteBufferId rbid,
                         poplar::Tensor t,
                         poplar::Tensor offset) const {
  poplar::Tensor rbTensor;

  if (!dv_p->lowering().hasRemoteBuffer(rbid)) {
    rbTensor = sgraph.getPoplarGraph().clone(
        t,
        debugContext(dv_p->lowering().getRemoteBufferName(rbid)),
        poplar::TensorCloneMethod::PRESERVE_ORDER_UNLESS_ALIASES);
    dv_p->lowering().createRemoteBuffer(rbid, rbTensor);
  }

  auto buffer = dv_p->lowering().getRemoteBuffer(rbid);
  rbTensor    = buffer.second.value();

  if (offset.valid() && offset.numElements() > 0) {
    poplar::program::Copy copy_prog(
        buffer.first, rbTensor, offset, debugContext());
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(buffer.first, rbTensor, debugContext());
    prog.add(copy_prog);
  }
}

void RemoteBaseOpx::store(poplar::program::Sequence &prog,
                          RemoteBufferId rbid,
                          poplar::Tensor t,
                          poplar::Tensor offset) const {
  auto buffer             = dv_p->lowering().getRemoteBuffer(rbid);
  poplar::Tensor rbTensor = buffer.second.value();
  if (offset.valid() && offset.numElements() > 0) {
    poplar::program::Copy copy_prog(
        rbTensor, buffer.first, offset, debugContext());
    prog.add(copy_prog);
  } else {
    poplar::program::Copy copy_prog(rbTensor, buffer.first, debugContext());
    prog.add(copy_prog);
  }
}

RemoteStoreOpx::RemoteStoreOpx(Op *op, Devicex *devicex)
    : RemoteBaseOpx(op, devicex) {
  verifyOp<RemoteStoreOp>(op, Onnx::CustomOperators::RemoteStore);
}

void RemoteStoreOpx::grow(poplar::program::Sequence &prog) const {
  auto &remoteStoreOp = getOp<RemoteStoreOp>();

  TensorId inTensorId =
      remoteStoreOp.input->tensor(RemoteStoreOp::getLocalTensorInIndex())->id;

  logging::opx::debug(
      "[RemoteStoreOpx] Growing RemoteStore for tensor {}, "
      "using RemoteBuffer {}",
      remoteStoreOp.input->tensor(RemoteStoreOp::getLocalTensorInIndex())->id,
      remoteStoreOp.getRemoteBufferId());

  poplar::Tensor inTensor =
      getInTensor(RemoteStoreOp::getLocalTensorInIndex()).getPoplarTensor();
  poplar::Tensor offset;

  if (remoteStoreOp.input->hasIndex(
          RemoteStoreOp::getRemoteBufferOffsetInIndex())) {
    offset = getInTensor(RemoteStoreOp::getRemoteBufferOffsetInIndex())
                 .getPoplarTensor();
  }

  preStore(graph(), prog, remoteStoreOp.getRemoteBufferId(), inTensor);
  store(prog, remoteStoreOp.getRemoteBufferId(), inTensor, offset);
}

RemoteLoadOpx::RemoteLoadOpx(Op *op, Devicex *devicex)
    : RemoteBaseOpx(op, devicex) {
  verifyOp<RemoteLoadOp>(op, Onnx::CustomOperators::RemoteLoad);
}

void RemoteLoadOpx::grow(poplar::program::Sequence &prog) const {
  auto &remoteLoadOp = getOp<RemoteLoadOp>();

  TensorId outTensorId =
      remoteLoadOp.output->tensor(RemoteLoadOp::getLocalTensorOutIndex())->id;

  // Tensor completely overwritten
  logging::opx::debug(
      "[RemoteLoadOpx] Growing RemoteLoad for tensor {} -> {}, "
      "using RemoteBuffer {}",
      remoteLoadOp.input->tensor(RemoteLoadOp::getLocalTensorInIndex())->id,
      remoteLoadOp.output->tensor(RemoteLoadOp::getLocalTensorOutIndex())->id,
      remoteLoadOp.getRemoteBufferId());

  poplar::Tensor outTensor =
      getInTensor(RemoteLoadOp::getLocalTensorInIndex()).getPoplarTensor();
  poplar::Tensor offset;

  if (remoteLoadOp.input->hasIndex(
          RemoteLoadOp::getRemoteBufferOffsetInIndex())) {
    offset = getInTensor(RemoteLoadOp::getRemoteBufferOffsetInIndex())
                 .getPoplarTensor();
  }

  outTensor = makeWritable(
      graph(), outTensor, remoteLoadOp.getRemoteBufferId(), outTensorId);
  load(graph(), prog, remoteLoadOp.getRemoteBufferId(), outTensor, offset);
  postLoad(prog, remoteLoadOp.getRemoteBufferId(), outTensor);

  if (hasInViewChangers(RemoteLoadOp::getLocalTensorInIndex())) {
    setOutViewChangers(
        RemoteLoadOp::getLocalTensorOutIndex(),
        getInViewChangers(RemoteLoadOp::getLocalTensorInIndex()));
  }
  setOutTensor(RemoteLoadOp::getLocalTensorOutIndex(),
               snap::Tensor{outTensor, graph()});
}

InputCreatorType RemoteLoadOpx::getInputCreatorType(InIndex index) const {
  return index == RemoteLoadOp::getLocalTensorInIndex()
             ? InputCreatorType::CanUnwind
             : PopOpx::getInputCreatorType(index);
}

snap::Tensor RemoteLoadOpx::unwindTensorLayout(snap::Tensor tensor,
                                               InIndex,
                                               OutIndex) const {
  return tensor;
}

view::RegMap RemoteLoadOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

RemoteExchangeOpx::RemoteExchangeOpx(Op *op, Devicex *devicex)
    : RemoteBaseOpx(op, devicex) {
  verifyOp<RemoteLoadOp>(op, Onnx::CustomOperators::RemoteExchange);
}

InputCreatorType RemoteExchangeOpx::getInputCreatorType(InIndex index) const {
  auto &remoteExchangeOp = getOp<RemoteExchangeOp>();
  return index < remoteExchangeOp.numLoads()
             ? InputCreatorType::CanUnwind
             : PopOpx::getInputCreatorType(index);
}

bool RemoteExchangeOpx::canUnwind(InIndex in, OutIndex out) const {
  return in == out;
}

snap::Tensor RemoteExchangeOpx::unwindTensorLayout(snap::Tensor tensor,
                                                   InIndex,
                                                   OutIndex) const {
  return tensor;
}

view::RegMap RemoteExchangeOpx::unwindRegion(InIndex, OutIndex) const {
  return [](const view::Region &r) { return view::Regions(1, r); };
}

void RemoteExchangeOpx::grow(poplar::program::Sequence &prog) const {
  auto &remoteExchangeOp = getOp<RemoteExchangeOp>();
  // RemoteStore
  {
    std::set<RemoteBufferId> usedRemoteBufferIds;

    int lastStoredIndex = 0;
    auto storeUntil     = [this, &prog, &remoteExchangeOp, &lastStoredIndex](
                          int storeUntilIndex) {
      for (; lastStoredIndex < storeUntilIndex; ++lastStoredIndex) {
        RemoteBufferId rbid = remoteExchangeOp.getRemoteBufferId(
            remoteExchangeOp.numLoads() + lastStoredIndex);
        store(prog,
              rbid,
              getInTensor(remoteExchangeOp.numLoads() + lastStoredIndex)
                  .getPoplarTensor(),
              getInTensor(2 * remoteExchangeOp.numLoads() +
                          remoteExchangeOp.numStores() + lastStoredIndex)
                  .getPoplarTensor());
      }
    };

    int storeIndex = 0;
    for (; storeIndex < remoteExchangeOp.numStores(); ++storeIndex) {
      RemoteBufferId rbid = remoteExchangeOp.getRemoteBufferId(
          remoteExchangeOp.numLoads() + storeIndex);
      logging::opx::debug(
          "[RemoteExchangeOpx] Growing RemoteStore for tensor {}, "
          "using RemoteBuffer {}, offset {}",
          remoteExchangeOp.input
              ->tensor(remoteExchangeOp.numLoads() + storeIndex)
              ->id,
          rbid,
          remoteExchangeOp.input
              ->tensor(2 * remoteExchangeOp.numLoads() +
                       remoteExchangeOp.numStores() + storeIndex)
              ->id);
      if (usedRemoteBufferIds.find(rbid) != usedRemoteBufferIds.end()) {
        // Double-use of a remote buffer detected
        storeUntil(storeIndex);
        logging::warn("RemoteBuffer {} used more than once in {}",
                      rbid,
                      remoteExchangeOp.debugName());
      }
      usedRemoteBufferIds.insert(rbid);
      preStore(inGraph(remoteExchangeOp.numLoads() + storeIndex),
               prog,
               rbid,
               getInTensor(remoteExchangeOp.numLoads() + storeIndex)
                   .getPoplarTensor());
    }
    storeUntil(storeIndex);
  }

  // RemoteLoad
  {
    // Prepare output tensors
    for (int loadIndex = 0; loadIndex < remoteExchangeOp.numLoads();
         ++loadIndex) {
      TensorId outTensorId     = remoteExchangeOp.output->tensor(loadIndex)->id;
      poplar::Tensor outTensor = getInTensor(loadIndex).getPoplarTensor();
      outTensor                = makeWritable(outGraph(loadIndex),
                               outTensor,
                               remoteExchangeOp.getRemoteBufferId(loadIndex),
                               outTensorId);
      if (hasInViewChangers(loadIndex)) {
        setOutViewChangers(loadIndex, getInViewChangers(loadIndex));
      }
      setOutTensor(loadIndex, snap::Tensor{outTensor, graph()});
    }

    std::set<RemoteBufferId> usedRemoteBufferIds;
    int lastPostLoadIndex = 0;
    auto postLoadUntil = [this, &prog, &remoteExchangeOp, &lastPostLoadIndex](
                             int postLoadUntilIndex) {
      for (; lastPostLoadIndex < postLoadUntilIndex; ++lastPostLoadIndex) {
        RemoteBufferId rbid =
            remoteExchangeOp.getRemoteBufferId(lastPostLoadIndex);
        poplar::Tensor outTensor =
            getOutTensor(lastPostLoadIndex).getPoplarTensor();
        postLoad(prog, rbid, outTensor);
      }
    };

    int loadIndex = 0;
    for (; loadIndex < remoteExchangeOp.numLoads(); ++loadIndex) {
      InIndex offsetIndex = remoteExchangeOp.numLoads() +
                            remoteExchangeOp.numStores() + loadIndex;
      poplar::Tensor outTensor = getOutTensor(loadIndex).getPoplarTensor();
      RemoteBufferId rbid      = remoteExchangeOp.getRemoteBufferId(loadIndex);
      logging::opx::debug(
          "[RemoteExchangeOpx] Growing RemoteLoad for tensor {} -> {}, "
          "using RemoteBuffer {}, offset {}",
          remoteExchangeOp.input->tensor(loadIndex)->id,
          remoteExchangeOp.output->tensor(loadIndex)->id,
          rbid,
          remoteExchangeOp.input->tensor(offsetIndex)->id);
      if (usedRemoteBufferIds.find(rbid) != usedRemoteBufferIds.end()) {
        // Double-use of a remote buffer detected
        postLoadUntil(loadIndex);
        logging::warn("RemoteBuffer {} used more than once in {}",
                      rbid,
                      remoteExchangeOp.debugName());
      }
      usedRemoteBufferIds.insert(rbid);
      load(outGraph(loadIndex),
           prog,
           rbid,
           outTensor,
           getInTensor(offsetIndex).getPoplarTensor());
    }
    postLoadUntil(loadIndex);
  }
}

snap::Graph &RemoteExchangeOpx::inGraph(InIndex in) const {
  if (op_p->getIr().virtualGraphsEnabled()) {
    auto &remoteExchangeOp = getOp<RemoteExchangeOp>();
    auto vgid = remoteExchangeOp.getIntrospectionInVirtualGraphId(in);
    return dv_p->lowering().getVirtualGraph(vgid.first, vgid.second);
  } else {
    return dv_p->lowering().graph();
  }
}

snap::Graph &RemoteExchangeOpx::outGraph(OutIndex out) const {
  if (op_p->getIr().virtualGraphsEnabled()) {
    auto &remoteExchangeOp = getOp<RemoteExchangeOp>();
    auto vgid = remoteExchangeOp.getIntrospectionInVirtualGraphId(out);
    return dv_p->lowering().getVirtualGraph(vgid.first, vgid.second);
  } else {
    return dv_p->lowering().graph();
  }
}

namespace {
OpxCreator<RemoteStoreOpx>
    remoteStoreOpxCreator(Onnx::CustomOperators::RemoteStore);
OpxCreator<RemoteLoadOpx>
    remoteLoadOpxCreator(Onnx::CustomOperators::RemoteLoad);
OpxCreator<RemoteExchangeOpx>
    remoteExchangeOpxCreator(Onnx::CustomOperators::RemoteExchange);
} // namespace
} // namespace popx
} // namespace popart
