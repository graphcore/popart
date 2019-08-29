#ifndef GUARD_NEURALNET_GROUP_MATMULS_HPP
#define GUARD_NEURALNET_GROUP_MATMULS_HPP

#include <popart/transforms/transform.hpp>

namespace popart {

class MatmulInfo {
public:
  MatmulInfo(Op *op_, bool t_) : op(op_), transpose(t_) {}
  MatmulInfo(MatmulInfo &&other)      = default;
  MatmulInfo(const MatmulInfo &other) = default;
  MatmulInfo &operator=(const MatmulInfo &rhs) = default;
  Op *op                                       = nullptr;
  bool transpose                               = false;

  TensorId transposeLhsTId;
  TensorId transposeRhsTId;
};

class GroupMatMuls : public Transform {
public:
  static std::size_t id();

  using InputShapes = std::tuple<popart::Shape, popart::Shape>;
  using GroupId     = unsigned;

  GroupMatMuls() : Transform() {}
  virtual ~GroupMatMuls() override {}

  virtual bool apply(Graph &graph) const override final;

  virtual std::size_t getId() const override final { return id(); }

  virtual std::string getName() const override final { return "GroupMatMuls"; }

private:
  std::map<InputShapes, std::vector<MatmulInfo>>
  findMatMuls(Graph &graph) const;

  std::map<InputShapes, std::map<GroupId, std::vector<MatmulInfo>>>
  findPotentialGroupedMatMuls(Graph &graph, GroupId &groupId) const;

  void addGroupedMatMul(Graph &graph,
                        GroupId groupId,
                        std::vector<MatmulInfo> &matmuls) const;
};

} // namespace popart

#endif
