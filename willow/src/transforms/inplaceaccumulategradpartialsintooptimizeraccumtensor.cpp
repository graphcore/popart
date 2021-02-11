#include <popart/transforms/inplaceaccumulategradpartialsintooptimizeraccumtensor.hpp>

#include <popart/graph.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/add.hpp>
#include <popart/op/init.hpp>
#include <popart/topocons.hpp>
#include <popart/vendored/optional.hpp>

#include <algorithm>
#include <tuple>
#include <typeinfo>
#include <vector>

namespace popart {

namespace {

struct InplaceAddTreeIntoOptimiserAccum {
  Tensor *optimiserAccum;
  AccumulateOp *optimiserAccumOp;
  InitOp *treeInitOp;
  Tensor *initialTreeAccum;
  std::vector<Op *> additionTreeOps;
};

/* Prototypes */

std::vector<InplaceAddTreeIntoOptimiserAccum>
findInplaceAddTreesIntoOptimiserAccums(const Graph &graph);

void rewriteAsAccumulateTreeOnOptimiserAccum(
    Graph &graph,
    const InplaceAddTreeIntoOptimiserAccum &tree);

void transferSettingsAndInheritPlacementProperties(const Op *from, Op *to);

bool isDistributiveOverAddition(const AccumulateOp *);

} // namespace

InplaceAccumulateGradPartialsIntoOptimizerAccumTensor::
    InplaceAccumulateGradPartialsIntoOptimizerAccumTensor() {}
InplaceAccumulateGradPartialsIntoOptimizerAccumTensor::
    ~InplaceAccumulateGradPartialsIntoOptimizerAccumTensor() {}

bool InplaceAccumulateGradPartialsIntoOptimizerAccumTensor::apply(
    Graph &graph) const {
  bool modified = false;

  for (const auto &tree : findInplaceAddTreesIntoOptimiserAccums(graph)) {
    rewriteAsAccumulateTreeOnOptimiserAccum(graph, tree);
    modified = true;
  }

  return modified;
}

std::size_t InplaceAccumulateGradPartialsIntoOptimizerAccumTensor::id() {
  return typeid(InplaceAccumulateGradPartialsIntoOptimizerAccumTensor)
      .hash_code();
}

namespace {

bool init = Transform::registerTransform(
    new InplaceAccumulateGradPartialsIntoOptimizerAccumTensor);

std::vector<InplaceAddTreeIntoOptimiserAccum>
findInplaceAddTreesIntoOptimiserAccums(const Graph &graph) {
  std::vector<InplaceAddTreeIntoOptimiserAccum> trees;

  for (const auto &id_op : graph.getOps()) {
    auto op               = id_op.second.get();
    auto optimiserAccumOp = dynamic_cast<AccumulateOp *>(op);
    if (optimiserAccumOp && isDistributiveOverAddition(optimiserAccumOp)) {
      /*
        We have the optimiser AccumulateOp. Try find the final output tensor of
        the inplace addition tree we hope it consumes. Then, iterate backwards
        through the tree, accumulating the inplace add ops we need to return.
      */

      std::vector<Op *> addInplaceOps;

      /* Initialise iterator variables */

      // The accumulation tensor of the addition tree. This will be our primary
      // loop variable.
      Tensor *treeAccum = nullptr;

      // Whether or not ANY of the treeAccums seen so far have had consumers
      // other than the op from the addition tree. As soon as we hit a treeAccum
      // with a consumer other than that, we can short-circuit.
      bool noOtherConsumers = true;

      // Invariants of traversal through addition tree:
      //   - treeAccum is non-null, has no other consumers, and is otherwise
      //     valid.
      // This means, because we are traversing backwards through the tree, we
      // only add an inplace add op to addInplaceOps once treeAccum is its INPUT
      // accum and we have established the invariants on it.

      // Try to find initial treeAccum and establish invariants on it.

      if (!optimiserAccumOp->hasInput(AccumulateOp::getUpdaterInIndex())) {
        continue;
      }
      treeAccum = optimiserAccumOp->inTensor(AccumulateOp::getUpdaterInIndex());

      noOtherConsumers =
          noOtherConsumers && treeAccum->consumers.getTotal() == 1;
      if (!noOtherConsumers) {
        continue;
      }

      // This is a temporary variable local to each loop iteration. We do not
      // establish any invariants on it. Do not use it after the traversal.
      Op *treeOp = nullptr;

      // Traverse backwards through inplace addition tree.
      while (
          /*
            Assume invariants on treeAccum hold. Try to set treeOp to the next
            inplace add op (treeAccum's producer).
          */

          // Update treeOp to treeAccum's producer and check not null.
          (treeOp = treeAccum->getProducerUnsafe()) &&

          // Check treeOp is an AddLhsInplaceOp.
          treeOp->isConvertibleTo<AddLhsInplaceOp>() &&

          // Check treeOp has both inputs (graph is presumably malformed if
          // not, but it is not really our job here to define what counts as a
          // malformed graph or not, so we do not throw an error).
          treeOp->hasInput(AddLhsInplaceOp::getArg1InIndex()) &&
          treeOp->hasInput(AddLhsInplaceOp::getArg0InIndex()) &&

          /*
            We now know treeOp is a valid add inplace op. Update treeAccum to
            its updated-in-place input and establish invariants.

            ONLY IF this new treeAccum is valid, do we know that treeOp forms a
            valid extension of our addition tree.
          */

          // Update treeAccum to treeOp's updated tensor and check not null.
          (treeAccum = treeOp->inTensor(AddLhsInplaceOp::getArg0InIndex())) &&

          // Update noOtherConsumers (check treeAccum has only 1 consumer, which
          // must be treeOp, which we already know consumes it).
          (noOtherConsumers =
               noOtherConsumers && (treeAccum->consumers.getTotal() == 1)) //
      ) {                                                                  //
        addInplaceOps.push_back(treeOp);
      }

      // treeAccum is now the initial accumulation tensor of the addition tree.
      // The invariants still hold.

      // If there was no inplace addition tree, there is nothing to do.
      if (addInplaceOps.empty()) {
        continue;
      }

      // Skip if treeAccum not created by Zero InitOp.
      InitOp *initOp = dynamic_cast<InitOp *>(treeAccum->getProducerUnsafe());
      if (!(initOp && initOp->getInitType() == InitType::Zero)) {
        continue;
      }

      // Reverse addInplaceOps so they are in forward order.
      std::reverse(addInplaceOps.begin(), addInplaceOps.end());

      trees.push_back(
          {optimiserAccumOp->inTensor(AccumulateOp::getVarToUpdateInIndex()),
           optimiserAccumOp,
           initOp,
           treeAccum,
           std::move(addInplaceOps)});
    }
  }

  return trees;
}

void rewriteAsAccumulateTreeOnOptimiserAccum(
    Graph &graph,
    const InplaceAddTreeIntoOptimiserAccum &tree) {
  /*
    The following diagrams correspond to `tree` as so:
      - dw0               = initialTreeAccum
      - Init              = treeInitOp
      - {AddLhsInplace0,
         AddLhsInplace1}  = additionTreeOps
      - Accumulate3       = optimiserAccumOp
      - accum             = optimiserAccum

    Recall:

    Init
     |
    dW0              pW0
      \             /
      AddLhsInPlace0
            |
           dW1              pW1
              \             /
              AddLhsInPlace1
                    |              A
                   dw2   accum ----|
                      \    |
                      Accumulate3
                           |
                         accum'
                           |
                           B

    Becomes:

     A
     |
    accum       pW0
      \         /
      Accumulate
          |
          dW1         pW1
            \         /
            Accumulate
                |
              accum'
                |
                B

    That is, we rewrite N add inplace ops on dW0 + 1 AccumulateOp on accum as N
    AccumulateOps on accum, removing dW0 and its InitOp.

    The inputs to this tree structure are Init/dW0 and accum. Note we have
    preserved their incoming/outgoing topological dependencies, respectively,
    when rewriting the tree.

    We do this in three steps:
      1. Replace dW0 with accum, removing InitOp.
      2. Rewrite all the adds as AccumulateOps.
      3. Remove the no longer needed original optimiser Accumulate Op, and
         connect its output to the last AccumulateOp of the (new) addition tree.

    We must do it in this order, moving forwards through the tree, so the
    `Op::setup()` calls can propagate TensorInfo correctly.

    Regarding the placement attributes and other Op state, we will use
    Op::inheritPlacementAttributes on the new Ops to infer these from the
    incoming Ops. The initial accum and final accum' keep their attributes.
    its existing attributes. The new AccumulateOps are strictly topoligcally
    after accum, so will inherit their attributes from it. Note, the original
    add ops were not strictly ordered relative to accum, so may have had
    different attributes.
  */

  /*
    1. Replace initialTreeAccum tensor with the optimiserAccum tensor, and
    remove initialTreeAccum's InitOp.

      ------------------------------- accum ---- A
      |                                |
      |         pW0                    |
      |         /                      |
      AddLhsInplace0                   |
            |                          |
           dW1           pW1           |
              \         /              |
              AddLhsInplace1           |
                    |                  |
                   dw2     |-----------|
                      \    |
                      Accumulate3
                           |
                          accum'
                           |
                           B
  */
  {
    const auto optimiserAccum   = tree.optimiserAccum;
    const auto treeInitOp       = tree.treeInitOp;
    const auto initialTreeAccum = tree.initialTreeAccum;

    // Already asserted initialTreeAccum has one consumer.
    auto firstAddOp = initialTreeAccum->consumers.getOps().at(0);

    firstAddOp->disconnectInTensor(AddLhsInplaceOp::getArg0InIndex());
    firstAddOp->connectInTensor(AddLhsInplaceOp::getArg0InIndex(),
                                optimiserAccum->id);
    // Will be replaced by AccumulateOp in next step, so we don't call setup().

    treeInitOp->disconnectOutTensor(initialTreeAccum);

    // Move topo cons to firstAddOp, so they get transferred to the first
    // AccumulateOp in the next step and are not lost. Note, the placement
    // attributes of treeInitOp are not needed. We assume firstAddOp's
    // attributes are correct, and just transfer those (in the next step).
    graph.topoCons->transfer(treeInitOp, firstAddOp);

    graph.eraseOp(treeInitOp->id);
    graph.getTensors().remove(initialTreeAccum->id);

    // Because we have kept the original optimiserAccum tensor, if it has a
    // producer, or incoming topo cons, etc; these will be retained when the
    // tensor is reconnected to firstAccumOp's input.
  }

  /*
    2. Rewrite addition tree as AccumulateOp tree.

      ------------------------------- accum ---- A
      |                                |
      |         pW0                    |
      |         /                      |
      Accumulate                       |
            |                          |
           dW1           pW1           |
              \         /              |
              Accumulate               |
                    |                  |
                   dw2     |-----------|
                      \    |
                      Accumulate3
                           |
                         accum'
                           |
                           B
  */
  {
    const auto optimiserAccum   = tree.optimiserAccum;
    const auto optimiserAccumOp = tree.optimiserAccumOp;
    const auto &additionTreeOps = tree.additionTreeOps;

    /* Helper for creating AccumulateOps like optimiserAccumOp. */

    const auto &accumFactor = optimiserAccumOp->getFactor();
    const auto accumType    = optimiserAccumOp->getAccumulationType();

    const auto accumFactorTensorId =
        accumFactor.isConst()
            ? nonstd::nullopt
            : nonstd::optional<TensorId>(
                  optimiserAccumOp->inId(AccumulateOp::getFactorInIndex()));

    const auto mkAccOp =
        [&](const std::string &replacedName) -> AccumulateOp * {
      auto op = std::make_unique<AccumulateOp>(
          accumType,
          accumFactor,
          Op::Settings{graph,
                       "Accumulate__IntoOptimiserAccum__replaced-" +
                           replacedName});

      if (accumFactorTensorId) {
        op->connectInTensor(AccumulateOp::getFactorInIndex(),
                            *accumFactorTensorId);
      }

      const auto rawOp = op.get();
      graph.moveIntoGraph(std::move(op));

      return rawOp;
    };

    /*
       Iterate over additionTreeOps, reconnecting their tensors to new
       AccumulateOps.
     */

    // Recall, in the first step, we swapped initialTreeAccum for
    // optimiserAccum, so optimiserAccum is now the initial tensor in the
    // addition tree.
    auto next           = optimiserAccum;
    decltype(next) curr = nullptr;

    // Assume LhsInplace for now

    for (auto &addOp : additionTreeOps) {
      // curr is the input accum tensor to the current addOp
      curr = next;

      addOp->disconnectInTensor(AddLhsInplaceOp::getArg0InIndex());

      const auto &partialId = addOp->inId(AddLhsInplaceOp::getArg1InIndex());
      addOp->disconnectInTensor(AddLhsInplaceOp::getArg1InIndex());

      next = addOp->outTensor(AddLhsInplaceOp::getOutIndex());
      addOp->disconnectOutTensor(next);

      auto accumOp = mkAccOp(addOp->name());
      accumOp->connectInTensor(AccumulateOp::getVarToUpdateInIndex(), curr->id);
      accumOp->connectInTensor(AccumulateOp::getUpdaterInIndex(), partialId);
      accumOp->connectOutTensor(AccumulateOp::getUpdatedVarOutIndex(),
                                next->id);
      accumOp->setup();

      // Note this looks both forward and backwards to infer the attributes,
      // but the attributes of the Add Op ahead will be transferred to the
      // AccumulateOp it eventually gets replaced with, so this is valid.
      transferSettingsAndInheritPlacementProperties(addOp, accumOp);
      graph.topoCons->transfer(addOp, accumOp);

      graph.eraseOp(addOp->id);
    }
  }

  /*
    3. Remove optimiserAccumOp and reattach output accum to last (new) accum op
    of addition tree.

     A
     |
    accum       pW0
      \         /
      Accumulate
          |
          dW1           pW1
            \         /
            Accumulate
                |
              accum'
                |
                B
  */
  {
    const auto optimiserAccumOp = tree.optimiserAccumOp;

    auto accumOut =
        optimiserAccumOp->outTensor(AccumulateOp::getUpdatedVarOutIndex());
    optimiserAccumOp->disconnectOutTensor(accumOut);

    auto finalTreeAccum =
        optimiserAccumOp->inTensor(AccumulateOp::getUpdaterInIndex());
    auto finalTreeAccumOp = finalTreeAccum->getProducer();

    if (finalTreeAccum->info != accumOut->info) {
      throw internal_error("[InplaceAccumulateGradPartialsIntoOptimiserAccumTen"
                           "sor] Expected TensorInfo of final tree accum and "
                           "optimiser output accum to be the same.");
    }

    optimiserAccumOp->disconnectAllInputs();

    finalTreeAccumOp->disconnectOutTensor(finalTreeAccum);
    finalTreeAccumOp->connectOutTensor(AccumulateOp::getUpdatedVarOutIndex(),
                                       accumOut->id);
    finalTreeAccumOp->setup();

    // finalTreeAccum's properties are inherited from the corresponding add op
    // it replaced, not optimiserAccumOp, but we still transfer
    // optimiserAccumOp's topo cons so they are not lost.
    graph.topoCons->transfer(optimiserAccumOp, finalTreeAccumOp);

    graph.getTensors().remove(finalTreeAccum->id);
    graph.eraseOp(optimiserAccumOp->id);

    // Because we have kept the original accumOut tensor, if it has any
    // consumer ops, these will be retained when the tensor is reconnected to
    // finalTreeAccumOp.
  }
}

// Copied and amended from Pattern::transferBaseProperties
void transferSettingsAndInheritPlacementProperties(const Op *from, Op *to) {
  // Directly transfer base properties from `from`.
  to->settings.scope            = from->settings.scope;
  to->settings.recomputeType    = from->settings.recomputeType;
  to->settings.tensorLocation   = from->settings.tensorLocation;
  to->fromLoss                  = from->fromLoss;
  to->toLoss                    = from->toLoss;
  to->settings.schedulePriority = from->settings.schedulePriority;
  to->settings.debugInfoId      = from->settings.debugInfoId;

  // Inherit placement attributes from surrounding topology.
  to->inheritPlacementAttributes(true);
}

/**
 * It is only mathematically correct to decompose certain AccumulateOps. For
 * example, an AccumulateOp with type MovingAverage cannot be decomposed:
 *
 *   let AccMovingAverage_f(accum, g) := f * accum + (1 - f) * g
 *
 *   AccMovingAverage_f(accum, sum_i^N(pW_i))
 *
 *   =/=
 *
 *   AccMovingAverage_f(pW_N,
 *     AccMovingAverage_f(pW_(N-1),
 *       ...
 *       AccMovingAverage_f(pW0,
 *         accum
 *       )
 *       ...
 *     )
 *   )
 */
bool isDistributiveOverAddition(const AccumulateOp *accOp) {
  const auto type = accOp->getAccumulationType();
  return type == AccumulationType::Add ||
         type == AccumulationType::DampenedAdd ||
         type == AccumulationType::DampenedAddSquare;
}

} // namespace

} // namespace popart
