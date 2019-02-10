#include <vector>
#include <poponnx/ces/transposece.hpp>
#include <poponnx/ndindices.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

template <typename T> class NDArray {
public:
  NDArray(T *d, const TensorInfo &i) : data(d), info(i), ndindices(info) {}
  T &at(int64_t i) { return data[i]; }
  T &at(const std::vector<int64_t> &indices) {
    return at(ndindices.flatten(indices));
  }
  T *data;
  const TensorInfo &info;
  NDIndices ndindices;
};

class TransposeFunctor {
public:
  // transpose a tensor
  template <typename T>
  std::vector<char> operator()(Tensor *in0, const Shape &perm) {
    Shape shape;
    for (auto d : perm) {
      shape.push_back(in0->info.shape()[d]);
    }

    TensorInfo outInfo(in0->info.data_type(), shape);
    std::vector<char> v_out(outInfo.nbytes());
    NDArray<T> output(reinterpret_cast<T *>(v_out.data()), outInfo);

    NDArray<T> data0(static_cast<T *>(in0->tensorData()->data()), in0->info);

    // in 2-d use a fast blocking algorithm for fewer cache misses
    // (this should be generalised to N-d, see T6847)
    if (perm.size() == 2 && perm[0] == 1 && perm[1] == 0) {
      int64_t BS  = 16;               // block size (in all dimensions)
      int64_t R0  = in0->info.dim(0); // number of   rows, dimension 0
      int64_t R1  = in0->info.dim(1); // number of "rows", dimension 1
      int64_t nB0 = R0 / BS;          // number of blocks, dimension 0
      int64_t nB1 = R1 / BS;          // number of blocks, dimension 1
      int64_t tB0 = R0 % BS;          // tail size, dimension 0
      int64_t tB1 = R1 % BS;          // tail size, dimension 1

      // consider an example, where BS = 4 and the tensor is 5x7.
      // we first process the fully tiled region of the tensor,
      // ****...
      // ****...
      // ****...
      // ****...
      // .......
      for (int64_t b0 = 0; b0 < nB0; ++b0) {
        for (int64_t b1 = 0; b1 < nB1; ++b1) {
          for (int64_t c0 = b0 * BS; c0 < b0 * BS + BS; ++c0) {
            for (int64_t c1 = b1 * BS; c1 < b1 * BS + BS; ++c1) {
              output.at(c1 * R0 + c0) = data0.at(c0 * R1 + c1);
            }
          }
        }
      }

      // we then process the edges without any blocking/tiling
      // .......
      // .......
      // .......
      // .......
      // *******
      for (int64_t c0 = 0; c0 < R0; ++c0) {
        for (int64_t c1 = BS * nB1; c1 < BS * nB1 + tB1; ++c1) {
          output.at(c1 * R0 + c0) = data0.at(c0 * R1 + c1);
        }
      }

      // ....***
      // ....***
      // ....***
      // ....***
      // .......
      for (int64_t c0 = BS * nB0; c0 < BS * nB0 + tB0; ++c0) {
        for (int64_t c1 = 0; c1 < BS * nB1; ++c1) {
          output.at(c1 * R0 + c0) = data0.at(c0 * R1 + c1);
        }
      }
    }

    // the non 2-D case (which should use blocking too T6847)
    else {
      for (int64_t i = 0; i < outInfo.nelms(); ++i) {

        // the N-dimensional indices in the output tensor
        auto indices = data0.ndindices.unflatten(i);

        // re-arrange the indices according to perm
        Shape pindices;
        pindices.reserve(perm.size());

        for (auto d : perm) {
          pindices.push_back(indices[d]);
        }

        // Move the value
        output.at(pindices) = data0.at(indices);
      }
    }

    return v_out;
  }
};

void ConstExprTranspose::insertOutput() {
  std::vector<char> data_;
  Tensor *in0 = atInIndex(0);

  Shape perm;
  nAtts.setIfPresent(perm, "perm");

  if (perm.empty()) {
    // Default is to reverse the input shape
    for (int64_t i = in0->info.rank() - 1; i >= 0; i--) {
      perm.push_back(i);
    }
  }

  // verify that perm is a valid permutation
  std::vector<int> present(in0->info.rank(), 0);
  for (auto &x : perm) {
    if (x >= 0 && x < in0->info.rank()) {
      present[x] = 1;
    }
  }
  if (!std::accumulate(
          present.begin(), present.end(), 1, std::multiplies<int>())) {
    throw error("invalid permutation in ConstExprTranspose");
  }

  // Determine the output shape
  Shape outShape;
  for (auto d : perm) {
    outShape.push_back(in0->info.shape()[d]);
  }

  TensorInfo outInfo(in0->info.data_type(), outShape);

  // Log the Tensor being constexpr'd away
  std::stringstream ss;
  ss << "Adding constexpr transpose output " << atOutIndex0() << " : ";
  outInfo.append(ss);
  ss << " to Ir ";
  logging::ir::debug(ss.str());

  auto data = callOpFunctor<TransposeFunctor>(in0->info.dataType(), in0, perm);
  addConstInitTensor(atOutIndex0(), outInfo, data.data());
}

} // namespace poponnx
