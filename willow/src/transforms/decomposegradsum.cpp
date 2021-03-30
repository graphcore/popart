// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/add.hpp>
#include <popart/op/init.hpp>
#include <popart/op/sum.hpp>
#include <popart/opmanager.hpp>
#include <popart/poprithmstransitiveclosure.hpp>

#include <transforms/autodiff/gradgrowersumop.hpp>
#include <popart/transforms/decomposegradsum.hpp>

#include <algorithm>
#include <tuple>
#include <vector>

namespace {
using namespace popart;

// Returns the earliest possible schedule index of tensor `t`, according to the
// transitive closure of the graph, `tc`.
uint64_t earliestForTensor(const PoprithmsTransitiveClosure &tc,
                           const Tensor *t);

// This class encapsulates an input tensor of a grad sum and the priority
// with which it should be merged in the decomposed sum's addition tree. That
// is, these priorities determine the order in which the tensors get added
// together.
//
// We say a tensor has higher priority if it has a lower earliest _possible_
// schedule index. This is to reduce the sum-liveness of this operation.
//
// Other Ir factors, like the tensor's pipeline stage, are accounted for.
class TensorMergePriority {
public:
  // Factory func.
  static TensorMergePriority of(Tensor *t, const uint64_t earliest);

  // All default constructors, assignments, destructor;

  TensorMergePriority() = default;

  bool operator<(const TensorMergePriority &other) const {
    return this->priority < other.priority;
  }

  Tensor *tensor() { return t; }

private:
  using ScheduleIdx = uint64_t;
  using Priority    = std::
      tuple<PipelineStage, ExecutionPhase, BatchSerializedPhase, ScheduleIdx>;

  Tensor *t;
  Priority priority;

  TensorMergePriority(Tensor *t, Priority p);
};
} // namespace

namespace popart {

std::size_t DecomposeGradSum::id() {
  return typeid(DecomposeGradSum).hash_code();
}

std::vector<Op *>
DecomposeGradSum::getDecomposableGradSumOps(const Graph &graph) const {
  std::vector<Op *> decomposableGradSumOps;
  // An op in the graph is deemed a decomposable GradSumOp if:
  // 1. it is a SumOp
  // 2. its name contains GradGrowerSumOp::getGradSumOpNamePrefix()
  // 3. it produces a tensor with an id that contains reservedGradientPrefix()
  // 4. it has a path from the loss
  // 5. it consumes >2 ActGrad tensors
  for (auto &id_op : graph.getOps()) {
    Op *op = id_op.second.get();
    // 1.
    if (op->isConvertibleTo<SumOp>()) {
      // 2.
      if (op->settings.name.find(GradGrowerSumOp::getGradSumOpNamePrefix()) !=
          std::string::npos) {
        // 3.
        if (op->outId(SumOp::getOutIndex()).find(reservedGradientPrefix()) !=
            std::string::npos) {
          // 4.
          if (op->outTensor(SumOp::getOutIndex())->fromLoss ==
              PathFromLoss::Yes) {
            auto inputs               = op->input->tensors();
            bool allInputsAreActGrads = true;
            for (Tensor *t : inputs) {
              if (t->tensorType() != TensorType::ActGrad) {
                allInputsAreActGrads = false;
              }
            }
            // 5.
            if (inputs.size() > 2 && allInputsAreActGrads) {
              decomposableGradSumOps.push_back(op);
            }
          }
        }
      }
    }
  }

  return decomposableGradSumOps;
}

// A cycles-for-liveness optimization. Turn gradient summations into
// a schedulable tree of additions such that partial gradients do not
// have to wait until the point of summation of all gradient partials
// to become no longer live.
//
// Consider the model:
// in0 -
//       \
//        Matmul0 - Matmul1 - Matmul2 - loss
//       /          /          /
//  w0 ------------------------
//
// In the backwards pass:
//
// loss_grad
//    |
//    |- Matmul2Grad_rhs ------------- gp0 -    w0 - -
//    |_ Matmul2Grad_lhs                    \         |
//          |                                \        .
//          |- Matmul1Grad_rhs ------- gp1 - Sum -- VarUpdate
//          |_ Matmul1Grad_lhs               /
//                |                         /
//                 - Matmul0Grad_lhs - gp2 -
//
// Observe that the gradient partials, gp0-1, must stay live until gp2
// becomes available so the sum can be computed.
//
// Now observe how the below transform can reduce sum liveness of the
// partial gradient tensors:
//
// loss_grad            InitOp(zero) - gpi -
//    |                                     \
//    |- Matmul2Grad_rhs ------------- gp0 - Add
//    |_ Matmul2Grad_lhs                      \
//          |                                  \
//          |- Matmul1Grad_rhs ------- gp1 ---- Add    w0 - -
//          |_ Matmul1Grad_lhs                   \          |
//                |                               \         .
//                 - Matmul0Grad_lhs - gp2 ------- Add  -- VarUpdate
bool DecomposeGradSum::apply(Graph &graph) const {
  /*
  We use a poprithms transitive closure of the graph's edges to determine
  the earliest possible schedule index of a tensor.

  As we decompose each grad sum in the graph, we are mutating it. However,
  this is not invalidating the transitive closure, for the following reasons:

    (1) We never need to lookup an Op that has been removed by a previous
        decomposition.
    (2) The dependencies recorded in the transitive closure that go "across"
        a previously decomposed grad sum are unchanged.
    (3) If the input tensors of a grad sum are all descendents of THE SAME
        previously decomposed grad sum, then the relative ordering of their
        earliest possible schedule indices are unchanged.

  (1) holds because we are only decomposing grad sums contructed in the backward
  pass due to splits in the network in the forward pass. If we were decomposing
  any sum op, the current implementation would not be correct. More information,
  includng a proof of (1) for grad sums only is given in [comment:proof-1]. Some
  discussion on generalising this transform to all sums is given in
  [comment:all-sums]

  A proof of (2) is given in [comment:proof-2].

  A proof of (3) is given in [comment:proof-3].

  The above properties give us correctness: we can mutate the underlying graph
  without re-computing the transitive closure each time.

  That is except for one edge case. However, we have chosen to use the
  implementation as is. More explanation and motivation for this is given in
  [comment:edge-case].
  */
  const auto tc = PoprithmsTransitiveClosure::fromGraph(graph);

  for (Op *gradSumOp : getDecomposableGradSumOps(graph)) {
    logging::debug("Decomposing gradient sum op '{}' into a tree off additions "
                   "of its inputs",
                   gradSumOp->str());

    // 1) For each input tensor of the sum, get it's merge priority; which takes
    //    into account its earliest possible schedule index in the graph. For
    //    more motivation of this, see [comment:tree-order].

    std::vector<TensorMergePriority> inputMergePriorities;
    inputMergePriorities.reserve(gradSumOp->input->n());

    for (auto *t : gradSumOp->input->tensors()) {
      const auto earliest = earliestForTensor(tc, t);
      inputMergePriorities.emplace_back(TensorMergePriority::of(t, earliest));
    }

    const std::size_t nPartials = inputMergePriorities.size();

    // 2) Set the order in which to sum the grad partials according to their
    //    merge priorities.

    std::vector<Tensor *> partialsSumOrder(nPartials);

    // Note: The priorities array was created in order of
    // `gradSumOp->input->tensors()`, which is deterministic; and `stable_sort`
    // is determinstic; so the resulting `partialsSumOrder` will be
    // deterministic.
    std::stable_sort(inputMergePriorities.begin(), inputMergePriorities.end());

    for (std::size_t i = 0; i < nPartials; i++) {
      partialsSumOrder[i] = inputMergePriorities[i].tensor();
    }

    if (logging::shouldLog(logging::Module::popart, logging::Level::Debug)) {
      logging::debug("  Partials sum order:");
      for (const auto *t : partialsSumOrder) {
        logging::debug("    {}", t->id);
      }
    }

    // 3) Replace the sum op in the graph with a tree of additions, adding the
    //    tensors together according to their order in `partialsSumOrder`.

    // Remove the old Grad Sum op
    Tensor *gradSum = gradSumOp->output->tensor(0);
    gradSumOp->disconnectAllInputs();
    gradSumOp->disconnectAllOutputs();
    graph.eraseOp(gradSumOp->id);

    // Now replace with a series of Adds
    std::vector<Op *> addOps;

    // Create InitOp to produce the initial gradient partial input to the
    // addition tree, gpi. It inherits settings from the producer of the first
    // gradient partial
    auto firstPartialProducer = partialsSumOrder.front()->getProducer();
    Op::Settings initSettings = firstPartialProducer->getOutSettings(
        firstPartialProducer->output->indices(partialsSumOrder.front())
            .front());
    initSettings.name = gradSum->id + "_InitOp";
    auto init         = std::make_unique<InitOp>(Onnx::CustomOperators::Init_1,
                                         partialsSumOrder.front()->info,
                                         TensorType::ActGrad,
                                         InitType::Zero,
                                         initSettings);
    OpId initOpId     = graph.moveIntoGraph(std::move(init));
    Op *initOp        = graph.getOps()[initOpId].get();
    TensorId gradSumInit = gradSum->id + "_init";
    initOp->createAndConnectOutTensor(InitOp::getOutIndex(), gradSumInit);

    // Since initOp needs to be scheduled post-loss,
    // but has no path from loss, we need to force
    // PathToLoss::No, PathFromLoss::Yes
    initOp->toLoss   = PathToLoss::No;
    initOp->fromLoss = PathFromLoss::Yes;
    initOp->setup();
    TensorId addLhsId = gradSumInit;

    // Is this decomposition part of a batch serialisation?
    bool batchSerialized = false;

    for (std::size_t i = 0; i < nPartials; i++) {
      Tensor *t                           = partialsSumOrder[i];
      std::unique_ptr<popart::Op> gradAdd = OpManager::createOp(
          Domain::ai_onnx,
          "Add",
          graph.getIr().getOpSetVersionFromModel(Domain::ai_onnx),
          graph,
          "GradAdd" + std::to_string(i));

      OpId opId = graph.moveIntoGraph(std::move(gradAdd));
      Op *op    = graph.getOps()[opId].get();
      op->connectInTensor(AddOp::getArg0InIndex(), addLhsId);
      op->connectInTensor(AddOp::getArg1InIndex(), t->id);
      if (i == nPartials - 1) {
        // The final summed gradient tensor - it already exists in the Ir
        op->connectOutTensor(AddOp::getOutIndex(), gradSum->id);
      } else {
        TensorId partSummedId = gradSum->id + "_" + std::to_string(i);
        op->createAndConnectOutTensor(AddOp::getOutIndex(), partSummedId);
        // For the next Add
        addLhsId = partSummedId;
      }

      // Gradient accumulator needs the same tensor layout as the gradient.
      // Allow the AddOp to propagate the tensor layout from the gradient
      // to the InitOp output:
      op->settings.inferTensorMappingToFrom.insert(
          {AddOp::getArg0InIndex(), AddOp::getArg1InIndex()});

      op->inheritPlacementAttributes(true);
      op->setup();
      op->toLoss   = PathToLoss::No;
      op->fromLoss = PathFromLoss::Yes;
      batchSerialized |= op->hasBatchSerializedPhase();
    }

    initOp->inheritPlacementAttributes(true);
    if (batchSerialized) {
      initOp->setBatchSerializedPhase(-1);
    }
  }

  return true;
}

} // namespace popart

namespace {
using namespace popart;

bool init = Transform::registerTransform(new DecomposeGradSum());

uint64_t earliestForTensor(const PoprithmsTransitiveClosure &tc,
                           const Tensor *t) {
  // These lines will throw if tensor / producer op / tc are invalid.

  const auto opId  = t->getProducer()->id;
  const auto rOpId = tc.rithmicOpId(opId);

  return tc->earliest(rOpId);
}

TensorMergePriority::TensorMergePriority(Tensor *t, Priority p)
    : t{t}, priority{std::move(p)} {}

TensorMergePriority TensorMergePriority::of(Tensor *t,
                                            const uint64_t earliest) {
  if (t == nullptr) {
    throw internal_error("decomposegradum: TensorMergePriority::of: Provided "
                         "tensor is null.");
  }

  // Is safe.
  Op *op = t->getProducer();

  // Compare factors to determine the optimal order for the addition tree.
  // Consider annotations that imply an order
  // (pipeline stage, pingpong phase, batch serialized phase)
  // If an attribute is not set, assume that the Op comes before any of the
  // Ops that have the attribute set by using the unused stage/phase.

  // TODO(T17524): Abstract inferring operation order so that this
  // transform does not require knowledge of the attributes

  Priority p{
      op->hasPipelineStage() ? op->getPipelineStage() : unusedPipelineStage,
      op->hasExecutionPhase() ? op->getExecutionPhase() : unusedExecutionPhase,
      op->hasBatchSerializedPhase() ? op->getBatchSerializedPhase()
                                    : unusedBatchSerializedPhase,
      earliest};

  // No known guarantees on what valid pipeline stage etc. should be, so no
  // checks here.

  return {t, std::move(p)};
}
} // namespace

/*
  ------------------------------------------------------------------------------
  -- [comment:proof-1] ---------------------------------------------------------

  Recall:
    (1) We never need to lookup an Op that has been removed by a previous
        decomposition.

  First, why do we need to lookup ops? Because the transitive closure (TC) is on
  the Op->Op edges of the graph, so we lookup tensors by their producer op.

  The only reason a lookup could fail is because the Op in question did not
  exist when we created the TC over the graph.

  Recall what this transform does: it removes a SumOp and replaces it with a
  series of new AddOps.

  Thus, the input tensors of the sum will end up with the same producers (these
  weren't changed).

  The new intermediate tensors that are the results of the adds will never have
  their producer looked up in the TC, as they are not the inputs to a grad sum.

  However, the output tensor has had its producer changed from the sum op to the
  final add op, which did not exist at the time the TC was created. So, if it is
  possible that this tensor is the input to another grad sum, (1) does not hold.

  We will now prove it is impossible to construct such a case (1A), and thus (1)
  does hold.

  ------------------------------------------------------------------------------

  The crux of the proof is that this transform applies only to sum operators
  that have been constructed as "grad sums" as part of the backward pass, not
  any arbitrary sum that might have been constructed by the user. See
  `getDecomposableGradSumOps` for the exact definition of what sum ops are
  eligble for decomposition.

  What is it in the forward pass that causes grad sums to be constructed in the
  backwards pass? The answer is a split-merge:

        /---> B --->\
    -> A             D ->
        \---> C --->/

  This results in multiple upstream gradients going into A's grad operator in
  the backward pass. Specifically, the gradients of the inputs of B and C
  that came from A. Due to maths, these gradients are summed before being given
  to A as a single tensor.

  This is the ONLY scenario that induces a grad sum in the backward pass.

  ------------------------------------------------------------------------------

  Back to the problem at hand, we are trying to prove that the output tensor
  of a decomposed grad sum will never be the input to another grad sum.

  In order to have a grad sum going into another grad sum; that is, a nested
  sum; we must have a nested split-merge in the forwad pass. An example is
  given below (showing only the tensors in the network):

    t0 -> t1 -> t2 -> t3 ---
    |     |                 \
    |     |---> t4 -> t5 ------> t8 ----------------> t11
    |     |                 /                    /
    |     |---> t6 -> t7 ---                    /
    |                                          /
    |--------------------------> t9 ----------/
    |                                        /
    |--------------------------> t10 --------

  The nested grad sums will be on t0 and t1, where the splits occur. The exact
  backwards graph that will get constructed, showing just the parts relevant to
  t0 and t1, is:

           ---<-- t0                              ---<-- t1
           |                                      |
    t0' <- T0' <- sum0 <- Sum0 <---------- t1' <- T1' <- sum1 <- Sum1
                          |                                      |
                          |--<-- t9', t10'                       |- t2', t4',
                                                                    t6'

  where the tensors are in lower case and the operators are in capital case.

  You can see that the inner sum operator, Sum1, goes into the actual grad
  operator of the forward op, T1', and not directly into the outer sum
  operator, Sum0. As stated above, the gradients go through a sum op before
  going to the next grad op itself.

  This means that nested split-merges result in grad sums that are not directly
  nested one-into-the-other, so this case is no different from two regular,
  separate grad sums. It is fundamentally impossible to have a grad sum op whose
  output tensor is directly the input of another grad sum op, so that tensor
  NEVER needs to be looked up in the TC!

  Thus, we have proven (1A) and consequently (1) too.

  ------------------------------------------------------------------------------
  -- [comment:all-sums] --------------------------------------------------------

  What if, in the future, we want to generalise this transform to ALL sums, not
  just grad sums?

  This means the user could directly construct a graph like
  `Sum(A, B, Sum(C, D, E))`, and thus the ouput tensor of a sum does feed
  directly into another sum.

  One approach to fix this could be to pre-compute all "earliests" for all
  tensors that are the input to a sum before performing any decompositions on
  the graph. This takes a little more space.

  Another approach could be a pattern that flattens all sums:

    Sum(A, B, Sum(C, D, E)) ---> Sum(A, B, C, D, E)

  as these are semantically equivalent. This completely elides the problem
  altogether, as there is no longer a sum whose output goes directly into
  another sum.

  On a slightly different note, neither of these solutions would happen to solve
  the edge case described in [comment:edge-case].

  ------------------------------------------------------------------------------
  -- [comment:proof-2] ---------------------------------------------------------

  Recall:
    (2) The dependencies recorded in the transitive closure that go "across"
        a previously decomposed grad sum are unchanged.

  Consider a grad sum S and arbitrary vertices B, C.

  If there was a transitive dependency B -> C before the decomposition occurs,
  and S was along the path between B and C, then after S has been
  subsituted for a tree of additions, there is still B -> C. That is:

    B ... S ... C
  becomes
    B ... Add0, Add1, ... C

  So the transitive dependency still holds after the underlying graph has
  been changed.

  Trivially, if S was not along the path between B and C, then the dependence
  is of course unaffected. Similarly, if there was no dependence between B
  and C to start with, then there is no new one introduced.

  Thus, we have proved (2).

  ------------------------------------------------------------------------------
  -- [comment:proof-3] ---------------------------------------------------------

  Recall:
    (3) If the input tensors of a grad sum are all descendents of THE SAME
        previously decomposed grad sum, then the relative ordering of their
        earliest possible schedule indices are unchanged.

  Consider a grad sum S with input tensors T_1, .., T_k; whose earliest possible
  schedule indices are t_1, ..., t_k; respectively.

  Say they are all descendents of another grad sum operator S'.

  Say S' gets decomposed into (j+1) ops (the add ops and the init op).

  After the decomposition, the removed sum S' no longer executes, but it has
  been replaced by j+1 new ops that do. Moreover, just like S', these new ops
  must all execute before the input tensors to S.

  Therefore, ALL of their schedule indices have increase by j. That is, for all
  i in [1, k]:

    t_i (after decomp) = t_i (before decomp) + j

  The key observation is all t_i have increased equally and so their relative
  ordering has not changed. It is this ordering that determines the tree
  decomposition (the order in which the input tensors get added). Thus,
  continuing to look them up by their old absolute values in the TC is fine,
  because the ordering between them hasn't changed.

  So we have proved (3).

  ------------------------------------------------------------------------------
  -- [comment:edge-case] -------------------------------------------------------

  Recall:
    (3) If the input tensors of a grad sum are all descendents of THE SAME
        previously decomposed grad sum, then the relative ordering of their
        earliest possible schedule indices are unchanged.

  However, this says nothing about when the input tensors are descendents of
  DIFFERENT grad sums.

  Recall:
    (2) The dependencies recorded in the transitive closure that go "across"
        a previously decomposed grad sum are unchanged.

  This statement does generalise to the required case, but it only says that the
  dependencies recorded in the TC are still valid. That is, iff B -> C before
  the decomposition, then the dependence hasn't changed after the decomposition.
  It does not say anything about the relative ordering of the "earliests" of the
  inputs tensors to grad sum.

  Thus, we can consider such a case now. That is, we want to generalise (3)
  to:
    (4) If the input tensors of a grad sum are all descendents of ANY (POSSIBLY
        DIFFERING) previously decomposed grad sum, then the relative ordering of
        their earliest possible schedule indices are unchanged.

  ------------------------------------------------------------------------------

  (4) is disproved by the following specific case:

  Consider the backwards graph with a grad sum:

             ----<---- C <--
            /
    <- Sum0 <----<---- H <--
            \
             ----<---- E <--

  Say C has earliest schedule index t_c , H has t_h, and E has t_e.

  Also, let: t_c < t_h < t_e , so the correct tree order is C, H, E.

  Say t_e - t_h = k.

  Now, what if H has an ancestor that is the output of another grad sum Sum1
  that also gets decomposed (before Sum0)?

  Say Sum1 has k' > k input tensors.

  Then, the new earliest schedule index of H is t_h + k'.

  This means, as k' > k, that the new correct tree order for Sum0 is C, E, H;
  but when we look up in the TC what the earliest schedule index of H is, we get
  the original t_h and thus the wrong tree order.

  Therefore, we have disproved (4). Thus far, this is the only case I can think
  of where (4) does not hold.

  ------------------------------------------------------------------------------

  To reaffirm then, the single edge case we cannot handle is as follows:

  Take any two input tensors of a grad sum. These two tensors must have
  differing sets of grad sums they are descended from that are also decomposed
  before this grad sum; and that those two sets of sums must have a cumulative
  difference in arity greater than the the difference between the original
  "earliests" of the two input tensors.

  ------------------------------------------------------------------------------

  We have not observed cases like this to happen often, and the loss of
  optimality should not be too severe. This is of course a statement whose
  validity will change over time; we may need to change the implementation
  eventually.

  If this ever becomes a problem, we will have to see if recomputing the TC for
  every decomposable grad sum becomes too computationally expensive.

  Using the TC solves issues in the tree ordering that the previous
  path-from-loss implementation did not, such as being able to handle skip
  connections. It is a more robust measure of how "early" a tensor will be
  allocated, and thus which one we will want to merge first in order to minimise
  sum liveness.

  Therefore, the change will be landed. See T23402 and its parent tasks.

  ------------------------------------------------------------------------------
  -- [comment:tree-order] ------------------------------------------------------

  Recall, the point of this transform is to reduce the liveness of sum ops.
  A sum with N inputs must have all N inputs allocated before the sum can
  proceed. Splitting the sum up into its elementary adds means we only ever need
  two "partials" live at once and can "merge" them incrementally. This reduces
  best potential liveness (depending on what the scheduler does) from N down to
  a constant factor.

  In other words, we are swapping out a single sum op for a tree of add ops.
  Note then, that we are fixing the order in which the partials get merged, by
  specifying a particular tree.

  That is, we are taking the schedule order of the adds out of the hands of the
  scheduler. Therefore, we must attempt to minimise sum-liveness ourselves.

  Our heuristic is based off the following intuition: whatever tensor was
  allocated first should be deallocated first, as this minmises the time these
  tensors are live, reducing sum-liveness.

  How can we know in a transform in what order the scheduler will choose to
  allocate our input tensors? We don't, but we can use a transitive closure of
  the graph edges to get a concrete answer for the earliest _possible_ schedule
  index at which an op can be scheduled.

  Therefore, we choose a tree order based on the order of the earliest possible
  schedule indices of the input tensor's producer ops.

  This is a reasonably robust heuristic.
*/
