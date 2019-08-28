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

    // Find all matmuls which have input tensors which have rank 3
    // TODO : Support 2D and 1D inputs
    if ((op->opid == Onnx::Operators::MatMul_1 ||
         op->opid == Onnx::Operators::MatMul_9) &&
        (op->inRank(MatMulOp::getLhsInIndex()) == 3 &&
         op->inRank(MatMulOp::getRhsInIndex()) == 3)) {

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

  std::string name = "groupedMatMul:" + std::to_string(groupId) + "(";
  for (int i = 0; i < matmulList.size(); ++i) {
    if (i != 0) {
      name += ",";
    }
    name += matmulList[i].op->name();
  }
  name += ")";

  // All grouped matmul need to have the same virtual graph id so
  // we can just use the first
  boost::optional<int64_t> virtualGraphId{};
  if (matmulList[0].op->hasVirtualGraphId()) {
    virtualGraphId = matmulList[0].op->getVirtualGraphId();
  }

  // For any input that needs first to be transposed
  for (auto &info : matmulList) {
    if (info.transpose) {
      const auto &inputTensorMap = info.op->input->tensorMap();

      Shape lhsTransposeDims = tranposeDims(
          inputTensorMap.at(MatMulOp::getLhsInIndex())->info.shape());
      Shape rhsTransposeDims = tranposeDims(
          inputTensorMap.at(MatMulOp::getRhsInIndex())->info.shape());

      info.transposeLhsTId =
          builder.transpose(inputTensorMap.at(MatMulOp::getRhsInIndex())->id,
                            rhsTransposeDims,
                            virtualGraphId,
                            name + "/lhs");

      info.transposeRhsTId =
          builder.transpose(inputTensorMap.at(MatMulOp::getLhsInIndex())->id,
                            lhsTransposeDims,
                            virtualGraphId,
                            name + "/rhs");
    }
  }

  // Concat the lhs and rhs inputs
  // Need to assume they are all 3D inputs
  std::vector<TensorId> lhsTensors, rhsTensors;
  for (auto &info : matmulList) {

    if (info.transpose) {
      lhsTensors.push_back(info.transposeLhsTId);
      rhsTensors.push_back(info.transposeRhsTId);
    } else {
      const auto &inputTensorMap = info.op->input->tensorMap();
      lhsTensors.push_back(inputTensorMap.at(MatMulOp::getLhsInIndex())->id);
      rhsTensors.push_back(inputTensorMap.at(MatMulOp::getRhsInIndex())->id);
    }
  }

  auto lhsConcatId = builder.concat(lhsTensors, virtualGraphId, name + "/lhs");
  auto rhsConcatId = builder.concat(rhsTensors, virtualGraphId, name + "/rhs");

  // Need to matmul the grouped lhs & grouped rhs
  auto matmulId =
      builder.matmul(lhsConcatId, rhsConcatId, virtualGraphId, name);

  int sliceCount  = 0;
  int groupOffset = 0;
  for (auto &info : matmulList) {
    auto outputTensor =
        info.op->output->tensorMap().at(MatMulOp::getOutIndex());

    info.op->disconnectAllInputs();
    info.op->disconnectAllOutputs();

    auto numGroups = outputTensor->info.shape()[0];

    Shape starts = {groupOffset};
    Shape ends   = {groupOffset + numGroups};
    Shape axes   = {0};

    if (info.transpose == false) {

      // Need to un-concat the grouped result
      builder.slice(matmulId,
                    starts,
                    ends,
                    axes,
                    outputTensor->id,
                    virtualGraphId,
                    name + std::to_string(sliceCount++));
    } else {

      auto sliceId = builder.slice(matmulId,
                                   starts,
                                   ends,
                                   axes,
                                   virtualGraphId,
                                   name + std::to_string(sliceCount++));

      Shape outputTransposeDims =
          tranposeDims(graph.getIr().getTensor(sliceId)->info.shape());

      builder.transpose(sliceId,
                        outputTransposeDims,
                        outputTensor->id,
                        virtualGraphId,
                        name + "/out");
    }

    graph.topoCons->transfer(info.op, outputTensor->getProducer());

    graph.eraseOp(info.op->id);

    groupOffset += numGroups;
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
