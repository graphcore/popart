#define BOOST_TEST_MODULE MergeCopiesTest

#include <boost/test/unit_test.hpp>
#include <popart/builder.hpp>
#include <popart/dataflow.hpp>
#include <popart/filereader.hpp>
#include <popart/inputshapeinfo.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/op/l1.hpp>
#include <popart/optimizer.hpp>
#include <popart/tensor.hpp>
#include <popart/tensordata.hpp>
#include <popart/tensors.hpp>
#include <popart/testdevice.hpp>

using namespace popart;

template <class T> std::vector<T *> getOpsOfType(Ir &ir) {
  std::vector<T *> ops;
  for (auto &id_op : ir.getMainGraphOps()) {
    auto op = id_op.second.get();
    if (op->isConvertibleTo<T>()) {
      ops.push_back(dynamic_cast<T *>(op));
    }
  }
  return ops;
}

BOOST_AUTO_TEST_CASE(MergeCopies0) {
  // Virtual Graph 0
  // {(i), (c)} -> [Add] -> (a0)
  //
  // Virtual Graph 1
  // {(i[copy]), (a0[copy])} -> [Add] -> ()
  //
  // The copy of i and a0 should be merged to a single IpuCopyOp

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo tinfo{"FLOAT", std::vector<int64_t>{4, 4}};

  float values[4 * 4]       = {0};
  ConstVoidData weight_data = {values, tinfo};

  auto i = builder->addInputTensor(tinfo);
  auto c = builder->addInitializedInputTensor(weight_data);

  auto a0 = aiOnnx.add({i, c});
  builder->virtualGraph(a0, 0);
  auto a1 = aiOnnx.add({i, a0});
  builder->virtualGraph(a1, 1);

  auto out_id = a1;
  builder->addOutputTensor(out_id);

  std::string proto = builder->getModelProto();
  auto model_proto  = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto data_flow = DataFlow(1, {{out_id, AnchorReturnType("ALL")}});

  SessionOptions opts;
  opts.virtualGraphMode = VirtualGraphMode::Manual;

  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({model_proto, {}, data_flow, {}, nullptr, *device, opts, {}});

  // Check the ir
  auto copies = getOpsOfType<IpuCopyOp>(ir);
  // 1) there should only be 1 IpuCopyOp
  BOOST_CHECK(copies.size() == 1);
  // 2) it should have 2 outputs
  BOOST_CHECK(copies.front()->output->n() == 2);
}

BOOST_AUTO_TEST_CASE(MergeCopies1) {
  // Virtual Graph 0
  // {(i), (c1)} -> [Add] -> (a0)
  //
  // Virtual Graph 1
  // {(i_copy), (c2)} -> [Add] -> (a1)
  // {(i_copy), (a0_copy)} -> [Add] -> (a2)
  // {(a1), (a2)} -> [Add] -> (out)
  //
  // No copies should be merged as (i_copy) can be consumed by:
  //   {(i_copy), (c2)} -> [Add] -> (a1)
  // before (a0_copy) is ready

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo tinfo{"FLOAT", std::vector<int64_t>{4, 4}};

  float values[4 * 4]       = {0};
  ConstVoidData weight_data = {values, tinfo};

  auto i  = builder->addInputTensor(tinfo);
  auto c1 = builder->addInitializedInputTensor(weight_data);
  auto c2 = builder->addInitializedInputTensor(weight_data);

  auto a0 = aiOnnx.add({i, c1});
  builder->virtualGraph(a0, 0);

  auto a1 = aiOnnx.add({i, c2});
  builder->virtualGraph(a1, 1);

  // Note that add({i, a0}) is not enough for this test, as we need to ensure
  // that a1 is generated before a2. Without topo-con exposure to public API, we
  // replace add with sum
  auto a2 = aiOnnx.sum({i, a0, a1});
  builder->virtualGraph(a2, 1);

  auto out = aiOnnx.add({a1, a2});
  builder->virtualGraph(out, 1);

  builder->addOutputTensor(out);

  std::string proto = builder->getModelProto();
  auto model_proto  = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto data_flow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

  SessionOptions opts;
  opts.virtualGraphMode = VirtualGraphMode::Manual;

  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({model_proto, {}, data_flow, {}, nullptr, *device, opts, {}});

  // Check the ir
  auto copies = getOpsOfType<IpuCopyOp>(ir);
  // 1) there should only be 1 IpuCopyOp
  BOOST_CHECK(copies.size() == 2);
  // 2) each should have 1 output
  for (auto c : copies) {
    BOOST_CHECK(c->output->n() == 1);
  }
}

BOOST_AUTO_TEST_CASE(MergeCopies2) {
  // Virtual Graph 0
  // {(i), (c1)} -> [Add] -> (a0)
  //
  // Virtual Graph 1
  // {(i_copy), (a0_copy)} -> [Add] -> (a1)
  // {(a1), (a0_copy)} -> [Add] -> (a2)
  //
  // Copy of (i_copy) and (a0_copy) should be merged

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo tinfo{"FLOAT", std::vector<int64_t>{4, 4}};

  float values[4 * 4]       = {0};
  ConstVoidData weight_data = {values, tinfo};

  auto i  = builder->addInputTensor(tinfo);
  auto c1 = builder->addInitializedInputTensor(weight_data);

  auto a0 = aiOnnx.add({i, c1});
  builder->virtualGraph(a0, 0);

  auto a1 = aiOnnx.add({i, a0});
  builder->virtualGraph(a1, 1);

  auto out = aiOnnx.add({a0, a1});
  builder->virtualGraph(out, 1);

  builder->addOutputTensor(out);

  std::string proto = builder->getModelProto();
  auto model_proto  = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto data_flow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

  SessionOptions opts;
  opts.virtualGraphMode = VirtualGraphMode::Manual;

  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({model_proto, {}, data_flow, {}, nullptr, *device, opts, {}});

  // Check the ir
  auto copies = getOpsOfType<IpuCopyOp>(ir);
  // 1) there should only be 1 IpuCopyOp
  BOOST_CHECK(copies.size() == 1);
  // 2) each should have 1 output
  BOOST_CHECK(copies.front()->output->n() == 2);
}

BOOST_AUTO_TEST_CASE(MergeCopies3) {
  // Virtual Graph 0
  // {(i1), (i2)} -> [concat] -> (a0)
  //
  // Virtual Graph 1
  // {(i1_copy), (a0_copy), (c1)} -> [concat] -> (out)
  //
  // Merge (i1_copy) and (a0_copy) but don't error that c1 isn't a copy.

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo tinfo{"FLOAT", std::vector<int64_t>{4, 4}};

  float values[4 * 4]       = {0};
  ConstVoidData weight_data = {values, tinfo};

  auto i1 = builder->addInputTensor(tinfo, "i1");
  auto i2 = builder->addInputTensor(tinfo, "i2");
  auto c1 = builder->addInitializedInputTensor(weight_data, "c1");

  auto a0 = aiOnnx.concat({i1, i2}, 0);
  builder->virtualGraph(a0, 0);

  auto out = aiOnnx.concat({i1, a0, c1}, 0);
  builder->virtualGraph(out, 1);

  builder->addOutputTensor(out);

  std::string proto = builder->getModelProto();
  auto model_proto  = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto data_flow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

  SessionOptions opts;
  opts.virtualGraphMode = VirtualGraphMode::Manual;

  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({model_proto, {}, data_flow, {}, nullptr, *device, opts, {}});

  // Check the ir
  auto copies = getOpsOfType<IpuCopyOp>(ir);
  // 1) there should only be 1 IpuCopyOp
  BOOST_CHECK(copies.size() == 1);
  // 2) and it should have 2 outputs
  BOOST_CHECK(copies.front()->output->n() == 2);
}

BOOST_AUTO_TEST_CASE(MergeCopies4) {
  // Virtual Graph 0
  // {(i0), (i1)} -> [Add] -> (a0)
  //
  // Virtual Graph 1
  // {(i0), (i1)} -> [Add] -> (a1)
  //
  // Virtual Graph 2
  // {(a0[copy]), (a1[copy])} -> [Add] -> ()
  //
  // The copy of a0 and a1 should be merged to a single IpuCopyOp

  auto builder = Builder::create();
  auto aiOnnx  = builder->aiOnnxOpset9();

  TensorInfo tinfo{"FLOAT", std::vector<int64_t>{4, 4}};

  auto i0 = builder->addInputTensor(tinfo, "i0");
  auto i1 = builder->addInputTensor(tinfo, "i1");

  auto a0 = aiOnnx.add({i0, i1});
  builder->virtualGraph(a0, 0);
  auto a1 = aiOnnx.add({i0, i1});
  builder->virtualGraph(a1, 1);
  auto a2 = aiOnnx.add({a0, a1});
  builder->virtualGraph(a2, 2);

  auto out = a2;
  builder->addOutputTensor(out);

  std::string proto = builder->getModelProto();
  auto model_proto  = io::getModelFromString(proto);

  // Create the IR, adding outId as an anchor
  auto data_flow = DataFlow(1, {{out, AnchorReturnType("ALL")}});

  SessionOptions opts;
  opts.virtualGraphMode = VirtualGraphMode::Manual;

  auto device = createTestDevice(TEST_TARGET);

  Ir ir;
  ir.prepare({model_proto, {}, data_flow, {}, nullptr, *device, opts, {}});

  // Check the ir
  auto copies = getOpsOfType<IpuCopyOp>(ir);
  // 1) there should only be 2 IpuCopyOps,
  //    one to copy i0 and i1 from virtual graph 0 to virtual graph 1,
  //    and one to copy a0 and a1 from virtual graph 0 and 1 to virtual graph 2.
  BOOST_CHECK(copies.size() == 2);
  // 2) and they should both have 2 inputs and outputs
  for (auto op : copies) {
    BOOST_CHECK(op->input->n() == 2);
    BOOST_CHECK(op->output->n() == 2);
  }
}
