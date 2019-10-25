#include <memory>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/concat.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/transpose.hpp>

#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/util.hpp>

#include <popart/transforms/groupmatmuls.hpp>
#include <popart/transforms/transformbuilder.hpp>

#include <boost/any.hpp>
#include <boost/optional/optional_io.hpp>
#include <boost/range/algorithm_ext.hpp>

namespace popart {

std::size_t GroupMatMuls::id() { return typeid(GroupMatMuls).hash_code(); }

std::map<GroupMatMuls::InputShapes, std::vector<MatmulInfo>>
GroupMatMuls::findMatMuls(Graph &graph) const {

  std::map<InputShapes, std::vector<MatmulInfo>> matmuls;

  for (auto &entry : graph.getOps()) {
    Op *op = entry.second.get();

    // Find all matmuls which have input tensors which have rank >= 3
    // TODO : Support 2D and 1D inputs
    if ((op->opid == Onnx::Operators::MatMul_1 ||
         op->opid == Onnx::Operators::MatMul_9) &&
        (op->inRank(MatMulOp::getLhsInIndex()) >= 3 &&
         op->inRank(MatMulOp::getRhsInIndex()) >= 3 &&
         op->inRank(MatMulOp::getLhsInIndex()) ==
             op->inRank(MatMulOp::getRhsInIndex()))) {

      auto insertMatMulInfo =
          [](std::map<InputShapes, std::vector<MatmulInfo>> &matmuls_,
             InputShapes &inputs,
             MatmulInfo info) {
            auto &matmulInfos = matmuls_[inputs];

            // Only add it once. A square tensor will have the same shape when
            // transposed.
            if (std::find_if(matmulInfos.begin(),
                             matmulInfos.end(),
                             [info](const MatmulInfo &i) -> bool {
                               return (i.op == info.op);
                             }) == matmulInfos.end()) {
              matmulInfos.push_back(info);
            }
          };

      // Get the last two dimensions
      /*
      auto matrixDims = [](Shape shape) -> Shape {
        // What about 1-D inputs?
        Shape matrixShape = {shape[shape.size() - 2], shape[shape.size() - 1]};
        return matrixShape;
      };
      */
      // Create the input shape tuple (lhs, rhs)
      /*
      auto inputs =
          std::make_tuple(matrixDims(op->inShape(MatMulOp::getLhsInIndex())),
                          matrixDims(op->inShape(MatMulOp::getRhsInIndex())));
      */
      auto inputs = std::make_tuple(op->inShape(MatMulOp::getLhsInIndex()),
                                    op->inShape(MatMulOp::getRhsInIndex()));

      // add it to the list of potential matmuls to be grouped
      insertMatMulInfo(matmuls, inputs, {op, false});

      auto transposeDim = [](Shape shape) -> Shape {
        // What about 1-D inputs?
        std::swap(shape[shape.size() - 1], shape[shape.size() - 2]);
        return shape;
      };

      // Also add the transpose'ed option. We are using the matrix commutative
      // property AxB = ( B.T x A.T ).T Create the input shape tuple (rhs.T,
      // lhs.T)
      /*
      auto inputsT = std::make_tuple(
          transposeDim(matrixDims(op->inShape(MatMulOp::getRhsInIndex()))),
          transposeDim(matrixDims(op->inShape(MatMulOp::getLhsInIndex()))));
      */
      auto inputsT =
          std::make_tuple(transposeDim(op->inShape(MatMulOp::getRhsInIndex())),
                          transposeDim(op->inShape(MatMulOp::getLhsInIndex())));

      // add it to the list of potential matmuls to be grouped
      insertMatMulInfo(matmuls, inputsT, {op, true});
    }
  }

  return matmuls;
}

std::map<GroupMatMuls::InputShapes,
         std::map<GroupMatMuls::GroupId, std::vector<MatmulInfo>>>
GroupMatMuls::findPotentialGroupedMatMuls(Graph &graph,
                                          GroupId &groupId) const {

  // First get a list of all mat muls & transposed matmuls
  std::map<InputShapes, std::vector<MatmulInfo>> matmuls = findMatMuls(graph);

  /*
  // Code to print all found matmul grouped but input shapes
  for(auto& entry : matmuls) {
    logging::ir::info("Shape {} x {}", std::get<0>(entry.first),
  std::get<1>(entry.first)); for(auto& matmul : entry.second) {
       logging::ir::info(" {}" , matmul.op->str());
    }
  }
  */

  // Find the
  std::map<InputShapes, std::map<GroupId, std::vector<MatmulInfo>>>
      matmulgroups;

  for (auto &entry : matmuls) {

    if (entry.second.size() > 1) {

      for (auto &matmulInfo : entry.second) {

        // Find or create a mat mul group for these input shapes
        auto &groups = matmulgroups[entry.first];

        if (groups.size() == 0) {
          // Add the first matmul for these input shapes
          groups.insert(std::pair<GroupId, std::vector<MatmulInfo>>(
              groupId++, {matmulInfo}));
        } else {

          bool groupFound = false;

          for (auto &group : groups) {

            bool isRelated = false;
            bool sameIpu   = false;

            // Only matmuls on the same virtual graph can be grouped
            if (matmulInfo.op->getOptionalVirtualGraphId() ==
                group.second[0].op->getOptionalVirtualGraphId()) {
              sameIpu = true;

              for (auto &member : group.second) {

                // parent
                {
                  OpsBeforeKey constraint;
                  constraint.insert(std::pair<Op *, std::vector<Op *>>(
                      matmulInfo.op, {member.op}));
                  graph.getIr().isSchedulable(constraint);
                  if (graph.getIr().isSchedulable(constraint) == false)
                    isRelated = true;
                }

                // child
                {
                  OpsBeforeKey constraint;
                  constraint.insert(std::pair<Op *, std::vector<Op *>>(
                      member.op, {matmulInfo.op}));
                  if (graph.getIr().isSchedulable(constraint) == false)
                    isRelated = true;
                }

                // Stop when one member of the group is related.
                if (isRelated) {
                  break;
                }
              }
            }

            // If not related to any members of this group add
            if (sameIpu && isRelated == false) {
              group.second.push_back(matmulInfo);
              groupFound = true;
            }
          }

          // Create new group if needed
          if (groupFound == false) {
            groups.insert(std::pair<GroupId, std::vector<MatmulInfo>>(
                groupId++, {matmulInfo}));
          }
        }
      }
    }
  }

  // Remove any groups of single matmuls
  for (auto &groups : matmulgroups) {
    erase_if(groups.second,
             [](const std::pair<GroupId, std::vector<MatmulInfo>> &i) {
               return i.second.size() < 2;
             });
  }

  erase_if(matmulgroups,
           [](const std::pair<InputShapes,
                              std::map<GroupId, std::vector<MatmulInfo>>> &i) {
             return i.second.empty();
           });

  return matmulgroups;
}

static Shape tranposeDims(Shape s) {
  Shape permutation(s.size());
  boost::iota(permutation, 0);
  std::swap(permutation[s.size() - 2], permutation[s.size() - 1]);
  return permutation;
}

void GroupMatMuls::addGroupedMatMul(Graph &graph,
                                    GroupId groupId,
                                    std::vector<MatmulInfo> &matmulList) const {

  TransformBuilder builder(graph);

  std::string name = "GroupedMatMul_" + std::to_string(groupId);

  // All grouped matmul need to have the same virtual graph id so
  // we can just use the first
  boost::optional<int64_t> virtualGraphId{};
  if (matmulList[0].op->hasVirtualGraphId()) {
    virtualGraphId = matmulList[0].op->getVirtualGraphId();
  }
  boost::optional<PipelineStage> pipelineStage =
      matmulList.at(0).op->getOptionalPipelineStage();

  // For any input that needs first to be transposed
  for (auto &info : matmulList) {
    if (info.transpose) {
      const auto &inputTensorMap = info.op->input->tensorMap();
      auto lhs                   = inputTensorMap.at(MatMulOp::getLhsInIndex());
      auto rhs                   = inputTensorMap.at(MatMulOp::getRhsInIndex());

      Shape lhsTransposeDims = tranposeDims(lhs->info.shape());
      Shape rhsTransposeDims = tranposeDims(rhs->info.shape());

      info.transposeLhsTId =
          builder.transpose(rhs->id,
                            rhsTransposeDims,
                            virtualGraphId,
                            pipelineStage,
                            info.op->name() + "_RhsTranspose",
                            createIntermediateTensorId(rhs->id));

      info.transposeRhsTId =
          builder.transpose(lhs->id,
                            lhsTransposeDims,
                            virtualGraphId,
                            pipelineStage,
                            info.op->name() + "_LhsTranspose",
                            createIntermediateTensorId(lhs->id));
    }
  }

  // Expand input to have a 1' at the front so we can concat and multiple
  // tensors with different group dimensions.
  for (auto &info : matmulList) {

    if (info.transpose) {

      auto lhs = graph.getIr().getTensor(info.transposeLhsTId);
      auto rhs = graph.getIr().getTensor(info.transposeRhsTId);

      auto lhsShape = lhs->info.shape();
      lhsShape.insert(lhsShape.begin(), 1);
      auto rhsShape = rhs->info.shape();
      rhsShape.insert(rhsShape.begin(), 1);

      info.expandedLhsTId =
          builder.reshape(lhs->id,
                          lhsShape,
                          virtualGraphId,
                          pipelineStage,
                          info.op->name() + "_LhsExpand",
                          createIntermediateTensorId(lhs->id));
      info.expandedRhsTId =
          builder.reshape(rhs->id,
                          rhsShape,
                          virtualGraphId,
                          pipelineStage,
                          info.op->name() + "_RhsExpand",
                          createIntermediateTensorId(rhs->id));

    } else {
      const auto &inputTensorMap = info.op->input->tensorMap();
      auto lhs                   = inputTensorMap.at(MatMulOp::getLhsInIndex());
      auto rhs                   = inputTensorMap.at(MatMulOp::getRhsInIndex());

      auto lhsShape = lhs->info.shape();
      lhsShape.insert(lhsShape.begin(), 1);
      auto rhsShape = rhs->info.shape();
      rhsShape.insert(rhsShape.begin(), 1);

      info.expandedLhsTId =
          builder.reshape(lhs->id,
                          lhsShape,
                          virtualGraphId,
                          pipelineStage,
                          info.op->name() + "_LhsExpand",
                          createIntermediateTensorId(lhs->id));

      info.expandedRhsTId =
          builder.reshape(rhs->id,
                          rhsShape,
                          virtualGraphId,
                          pipelineStage,
                          info.op->name() + "_RhsExpand",
                          createIntermediateTensorId(rhs->id));
    }
  }

  // Concat the lhs and rhs inputs
  std::vector<TensorId> lhsTensors, rhsTensors;
  for (auto &info : matmulList) {
    lhsTensors.push_back(info.expandedLhsTId);
    rhsTensors.push_back(info.expandedRhsTId);
  }

  auto lhsConcatId = builder.concat(lhsTensors,
                                    virtualGraphId,
                                    pipelineStage,
                                    name + "_LhsConcat",
                                    builder.getNextId(name + "_LhsConcat"));
  auto rhsConcatId = builder.concat(rhsTensors,
                                    virtualGraphId,
                                    pipelineStage,
                                    name + "_RhsConcat",
                                    builder.getNextId(name + "_RhsConcat"));

  // Need to matmul the grouped lhs & grouped rhs
  auto matmulId = builder.matmul(lhsConcatId,
                                 rhsConcatId,
                                 virtualGraphId,
                                 pipelineStage,
                                 name,
                                 builder.getNextId(name),
                                 {},
                                 matmulList[0].op->opid);

  int groupOffset = 0;
  for (auto &info : matmulList) {
    auto outputTensor =
        info.op->output->tensorMap().at(MatMulOp::getOutIndex());

    info.op->disconnectAllInputs();
    info.op->disconnectAllOutputs();

    Shape starts = {groupOffset};
    Shape ends   = {groupOffset + 1};
    Shape axes   = {0};

    if (info.transpose == false) {

      // Need to un-concat the grouped result

      auto sliceId = builder.slice(matmulId,
                                   starts,
                                   ends,
                                   axes,
                                   virtualGraphId,
                                   pipelineStage,
                                   info.op->name() + "_Slice:",
                                   createIntermediateTensorId(matmulId));

      builder.squeeze(sliceId,
                      {0},
                      outputTensor->id,
                      virtualGraphId,
                      pipelineStage,
                      info.op->name() + "_Squeeze:");

    } else {

      auto sliceId = builder.slice(matmulId,
                                   starts,
                                   ends,
                                   axes,
                                   virtualGraphId,
                                   pipelineStage,
                                   info.op->name() + "_Slice:",
                                   createIntermediateTensorId(matmulId));

      auto squeezeId = builder.squeeze(sliceId,
                                       {0},
                                       virtualGraphId,
                                       pipelineStage,
                                       info.op->name() + "_Squeeze:",
                                       createIntermediateTensorId(sliceId));

      Shape outputTransposeDims =
          tranposeDims(graph.getIr().getTensor(squeezeId)->info.shape());

      builder.transpose(squeezeId,
                        outputTransposeDims,
                        outputTensor->id,
                        virtualGraphId,
                        pipelineStage,
                        info.op->name() + "_Transpose:");
    }

    graph.topoCons->transfer(info.op, outputTensor->getProducer());

    graph.eraseOp(info.op->id);

    groupOffset++;
  }
}

bool GroupMatMuls::apply(Graph &graph) const {

  bool finished   = false;
  GroupId groupId = 0;

  while (finished == false) {

    // Get a list of all potential grouped matmuls
    auto groupedMatmuls = findPotentialGroupedMatMuls(graph, groupId);

    // Pick a grouped matmul
    // 1. Find the group with the most matmuls - maybe we should find the
    // matmuls with the most elements
    GroupId selectedGroupId             = -1;
    std::vector<MatmulInfo> *matmulList = nullptr;

    for (auto &entry : groupedMatmuls) {
      for (auto &groups : entry.second) {

        // Ignore groups of just 1 matmul
        if (groups.second.size() <= 1) {
          continue;
        }

        if (matmulList == nullptr ||
            (matmulList->size() < groups.second.size())) {
          matmulList      = &(groups.second);
          selectedGroupId = groups.first;
        }
      }
    }

    // If no grouped mat mul is selected we are finished.
    if (matmulList == nullptr || matmulList->size() == 0) {
      finished = true;
    } else {
      logging::transform::info("Grouping MatMuls ({})", selectedGroupId);
      for (auto &info : (*matmulList)) {
        logging::transform::info(
            "    {} {}{} transpose:{} ipu:{}",
            info.op->str(),
            info.op->input->tensor(MatMulOp::getLhsInIndex())->info.shape(),
            info.op->input->tensor(MatMulOp::getRhsInIndex())->info.shape(),
            info.transpose,
            info.op->getOptionalVirtualGraphId());
      }

      // Replace the matmuls with the grouped matmul
      addGroupedMatMul(graph, selectedGroupId, *matmulList);

      // Update the graph verticies
      graph.getIr().updateVertices();
    }
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new GroupMatMuls);
}

} // namespace popart
