// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <memory>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/add.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/pad.hpp>
#include <popart/op/sum.hpp>
#include <popart/patterns/padsum.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>

#include <boost/numeric/interval.hpp>
#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/numeric.hpp>

#include <algorithm>
#include <iterator>

namespace popart {

bool PadSumPattern::matches(Op *op) const {
  // Is the input a sum/add
  if (!op->isConvertibleTo<AddOp>() && !op->isConvertibleTo<SumOp>()) {
    return false;
  }

  // At least 2 inputs
  if (op->input->n() < 2) {
    return false;
  }

  std::vector<Op *> inputProducers;
  inputProducers.reserve(op->input->n());
  for (int i = 0; i < op->input->n(); ++i) {
    inputProducers.push_back(op->input->tensor(i)->getProducerUnsafe());
  }

  const auto padPredicate = [](Op *producer) -> bool {
    auto pad = dynamic_cast<PadOp *>(producer);

    return pad != nullptr && pad->getPadValue() == 0.0f &&
           pad->getMode() == "constant";
  };

  // Are the producers all pads with constant zero padding
  if (!std::all_of(
          inputProducers.begin(), inputProducers.end(), padPredicate)) {
    return false;
  }

  // Require all pads to have the same dimension
  const auto dimensionPredicate = [&inputProducers](Op *producer) -> bool {
    auto first = dynamic_cast<PadOp *>(inputProducers[0]);
    auto pad   = dynamic_cast<PadOp *>(producer);

    return first->padDimensions() == pad->padDimensions() &&
           pad->padDimensions().size() == 1;
  };

  // Do all of the producers pad on the same dimension
  // TODO (T6771) Support multi-dimension case
  if (!std::all_of(
          inputProducers.begin(), inputProducers.end(), dimensionPredicate)) {
    return false;
  }

  view::Regions regions;
  regions.reserve(inputProducers.size());
  for (auto producer : inputProducers) {
    auto pad = dynamic_cast<PadOp *>(producer);

    regions.push_back(pad->valueRegion());
  }

  // Are all of the non-zero regions non-overlapping
  for (int i = 0; i < regions.size(); ++i) {
    for (int k = i + 1; k < regions.size(); ++k) {
      auto intersection = regions[i].intersect(regions[k]);

      // Found an overlap
      if (!intersection.isEmpty()) {
        return false;
      }
    }
  }

  return true;
}

std::vector<const Tensor *> PadSumPattern::touches(Op *op) const {
  std::vector<const Tensor *> inputs;
  inputs.reserve(op->input->n());

  // We will be touching the tensors between the pads and add/sum
  for (int i = 0; i < op->input->n(); ++i) {
    inputs.push_back(op->input->tensor(i));
  }

  return inputs;
}

static std::vector<int> computeConcatOrder(Op *op) {
  std::vector<int> order(op->input->n());
  boost::iota(order, 0);

  // lexicographical_compare works here because we assume the tensors are padded
  // on the same dimension.
  const auto compPredicate = [&](int i, int k) -> bool {
    auto left  = dynamic_cast<PadOp *>(op->input->tensor(i)->getProducer());
    auto right = dynamic_cast<PadOp *>(op->input->tensor(k)->getProducer());

    return boost::lexicographical_compare(left->getPads(), right->getPads());
  };

  // Decide what order to concatenate the input tensors
  boost::sort(order, compPredicate);

  return order;
}

static std::vector<PadOp *> getProducerOps(Op *op,
                                           const std::vector<int> &order) {
  std::vector<PadOp *> producers;
  producers.reserve(op->input->n());

  for (int i = 0; i < op->input->n(); ++i) {
    producers.push_back(
        dynamic_cast<PadOp *>(op->input->tensor(order[i])->getProducer()));
  }

  return producers;
}

static std::vector<const Tensor *>
getInputTensors(const std::vector<PadOp *> &producers) {
  std::vector<const Tensor *> inputs(producers.size());

  boost::transform(producers, inputs.begin(), [](PadOp *pad) {
    return pad->input->tensor(PadOp::getInIndex());
  });

  return inputs;
}

static int64_t findConcatAxis(const std::vector<PadOp *> &producers) {
  // When actually looking for the concat axis, we look for the first dimension
  // to have a non-zero value. We must look through at least half of the pad ops
  // to find the actual min.
  const auto nonZeroDim = [](int64_t dim) { return dim != 0; };
  const auto findAxis   = [nonZeroDim](int64_t m, PadOp *pad) {
    auto &pads = pad->getPads();
    auto found = boost::find_if<boost::return_begin_found>(pads, nonZeroDim);
    return std::min<int64_t>(m, boost::distance(found));
  };

  // Find the concat axis
  return boost::accumulate(
      producers, std::numeric_limits<int64_t>::max(), findAxis);
}

static std::vector<boost::numeric::interval<int64_t>>
computeInputIntervals(int64_t axis,
                      const std::vector<PadOp *> &producers,
                      const std::vector<const Tensor *> &inputs) {
  std::vector<boost::numeric::interval<int64_t>> intervals(producers.size());

  const auto computeInputInterval =
      [axis](PadOp *pad, const Tensor *t) -> boost::numeric::interval<int64_t> {
    return {pad->getPads()[axis],
            pad->getPads()[axis] + t->info.dim(static_cast<int>(axis))};
  };

  boost::transform(producers, inputs, intervals.begin(), computeInputInterval);

  return intervals;
}

static boost::numeric::interval<int64_t> computeOutputInterval(int64_t axis,
                                                               Tensor *output) {
  return {0, output->info.dim(static_cast<int>(axis))};
}

static std::vector<boost::numeric::interval<int64_t>>
subtractIntervals(boost::numeric::interval<int64_t> a,
                  const std::vector<boost::numeric::interval<int64_t>> &bs) {
  std::vector<boost::numeric::interval<int64_t>> gaps;
  gaps.reserve(bs.size() + 1);

  // Find the regions of a not covered by an element of bs
  if (bs.empty()) {
    gaps.push_back(a);
  } else {
    gaps.emplace_back(a.lower(), bs.front().lower());
    for (int i = 0; i < bs.size() - 1; ++i) {
      gaps.emplace_back(bs[i].upper(), bs[i + 1].lower());
    }
    gaps.emplace_back(bs.back().upper(), a.upper());
  }

  const auto empty = [](boost::numeric::interval<int64_t> gap) {
    return gap.lower() != gap.upper();
  };

  // Remove empty "gaps"
  auto eraseRegion =
      boost::stable_partition<boost::return_found_end>(gaps, empty);

  gaps.erase(eraseRegion.begin(), eraseRegion.end());

  return gaps;
}

static bool customIntervalComp(boost::numeric::interval<int64_t> a,
                               boost::numeric::interval<int64_t> b) {
  return a.lower() < b.lower();
}

static bool notCustomIntervalComp(boost::numeric::interval<int64_t> a,
                                  boost::numeric::interval<int64_t> b) {
  return a.lower() > b.lower();
}

static std::vector<std::unique_ptr<PadOp>> createPadOps(
    int64_t axis,
    const std::vector<const Tensor *> &inputs,
    const std::vector<boost::numeric::interval<int64_t>> &inputIntervals,
    const std::vector<boost::numeric::interval<int64_t>> &paddingIntervals,
    const Op::Settings &settings,
    const TensorId &tensorId) {
  std::vector<std::unique_ptr<PadOp>> result(inputs.size());

  auto subrange = boost::make_iterator_range(paddingIntervals.begin(),
                                             paddingIntervals.end());

  for (int i = 0; i < result.size(); ++i) {
    auto interval = inputIntervals[i];

    auto right = boost::lower_bound(subrange, interval, customIntervalComp);

    auto subrange_r = boost::adaptors::reverse(subrange);
    auto left = boost::upper_bound(subrange_r, interval, notCustomIntervalComp);

    const auto rank = inputs[0]->info.rank();
    std::vector<int64_t> padding(rank * 2, 0);

    // We must take the left padding
    if (left != std::end(subrange_r) && left->upper() == interval.lower()) {
      padding[axis] = left->upper() - left->lower();

      // We should take the right padding with the left, if we can
      if (right != std::end(subrange) && right->lower() == interval.upper()) {
        padding[rank + axis] = right->upper() - right->lower();

        // Advance the subrange past the right
        subrange = boost::make_iterator_range(std::next(right, 1),
                                              paddingIntervals.end());
      }
    } else if (std::distance(right, paddingIntervals.end()) == 1 &&
               right != std::end(subrange) &&
               right->lower() == interval.upper()) {
      // We must take the right alone, if it's the last padding
      padding[rank + axis] = right->upper() - right->lower();

      // Advance the subrange past the right
      subrange = boost::make_iterator_range(std::next(right, 1),
                                            paddingIntervals.end());
    }

    result[i] = std::make_unique<PadOp>(
        Onnx::Operators::Pad_2, padding, 0, "constant", settings);
  }

  for (int i = 0; i < result.size(); ++i) {
    result[i]->connectInTensor(0, inputs[i]->id);
    result[i]->createAndConnectOutTensor(
        0, result[i]->getIr().createIntermediateTensorId(tensorId));
  }

  return result;
}

static std::unique_ptr<ConcatOp>
createConcatOp(int64_t axis,
               const std::vector<std::unique_ptr<PadOp>> &producers,
               const TensorId outTensor,
               const Op::Settings &settings) {
  // Create the concat op
  std::unique_ptr<ConcatOp> concat = std::make_unique<ConcatOp>(
      popart::Onnx::AiOnnx::OpSet9::Concat, axis, settings);

  // Connect the concat op to the input and output tensors
  concat->connectOutTensor(0, outTensor);
  for (int i = 0; i < producers.size(); ++i) {
    concat->connectInTensor(
        i, producers[i]->output->tensor(PadOp::getOutIndex())->id);
  }

  return concat;
}

static void insertPadOps(Graph &graph,
                         std::vector<std::unique_ptr<PadOp>> &padOps) {
  for (auto &pad : padOps) {
    pad->setup();
    graph.moveIntoGraph(std::move(pad));
  }
  padOps.clear();
}

static void insertConcat(Graph &graph, std::unique_ptr<ConcatOp> &concat) {
  concat->setup();
  graph.moveIntoGraph(std::move(concat));
}

static void removeProducers(Graph &graph,
                            const std::vector<PadOp *> &producers) {
  for (auto &producer : producers) {
    producer->disconnectAllInputs();
    producer->disconnectAllOutputs();
    graph.eraseOp(producer->id);
  }
}

static void removeOp(Graph &graph, Op *op) {
  op->disconnectAllInputs();
  op->output->clear();
  graph.eraseOp(op->id);
}

bool PadSumPattern::apply(Op *op) const {
  auto &graph                  = op->getGraph();
  const std::vector<int> order = computeConcatOrder(op);

  // For the purposes of this pattern, we assume that add and sum use the same
  // output tensor index
  if (AddOp::getOutIndex() != SumOp::getOutIndex()) {
    throw error("Logic error in PadSumPattern::apply: AddOp::getOutIndex() and "
                "SumOp::getOutIndex() don't match.");
  }

  auto output = op->outTensor(AddOp::getOutIndex());
  op->disconnectOutTensor(output);

  // Find the input tensors and the intermediate pads
  const auto producers = getProducerOps(op, order);
  const auto inputs    = getInputTensors(producers);

  // Find the concat axis
  const auto axis = findConcatAxis(producers);

  // The 1D intervals of the output tensor axis covered by the input tensors
  const auto inputIntervals = computeInputIntervals(axis, producers, inputs);
  const auto outputInterval = computeOutputInterval(axis, output);
  const auto gaps           = subtractIntervals(outputInterval, inputIntervals);

  // Create new pad ops which don't have overlapping outputs
  auto padTensorOps = createPadOps(
      axis, inputs, inputIntervals, gaps, op->settings, output->id);

  // Create the concat op
  auto concat = createConcatOp(axis, padTensorOps, output->id, op->settings);

  // Insert the new ops into the IR
  insertPadOps(graph, padTensorOps);
  insertConcat(graph, concat);

  // Remove the old ops from the IR
  removeProducers(graph, producers);
  removeOp(graph, op);

  return true;
}

namespace {
static PatternCreator<PadSumPattern> PadSumPattern(PreAliasPatternType::PADSUM,
                                                   "PadSum");
}

} // namespace popart
