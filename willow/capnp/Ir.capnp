# Copyright (c) 2020 Graphcore Ltd. All rights reserved.
@0xdf4b02ad4598ec3d;  # unique file ID, generated by `capnp id`

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("popart::cap");

using TensorId = Text;

enum TensorType{
  actGrad @0;
  constant @1;
  stream @2;
  unknown @3;
  variable @4;
  n @5;
}


enum DataType {
  uint8 @0;
  int8 @1;
  uint16 @2;
  int16 @3;
  int32 @4;
  int64 @5;
  uint32 @6;
  uint64 @7;
  bool @8;
  float @9;
  float16 @10;
  bfloat16 @11;
  double @12;
  complex64 @13;
  complex128 @14;
  string @15;
  undefined @16;
}

struct TensorInfo{
  using Shape = List(Int64);
  struct DataTypeInfo{
    dataType @0: DataType;
    nbytes @1: Int32;
    isFixedPoint @2: Bool;
    name @3: Text;
    lCaseName @4: Text;
  }

  dataTypeInfo @0: DataTypeInfo;
  shape @1: Shape;
}

struct TensorLocationInfo{
    struct RemoteBufferInfo{
    id @0: Int64;
    index @1: Int64;
  }

  remote @0: Bool;
  sharded @1: Bool;
  remoteBufferInfo @2: RemoteBufferInfo;
}

enum CommGroupType {
  all @0;
  consecutive @1;
  orthogonal @2;
  none @3;
  n @4;
}

struct CommGroup {
  type @0: CommGroupType;
  replicaGroupSize @1: Int32;
}

enum VariableRetrievalMode {
  onePerGroup @0;
  allReduceReplicas @1;
  allReplicas @2;
}

struct VariableSettings {
  useCommGroup @0: Bool;
  commGroupType @1: CommGroupType;
  stride @2: Int32;
  groupSize @3: Int32;
  retrievalMode @4: VariableRetrievalMode;
}

struct Tensor {
  id @0 :TensorId;
  tensorType @1 :TensorType;
  tensorInfo @2 :TensorInfo;
  tensorLocationInfo @3: TensorLocationInfo;
  variableSettings @4: VariableSettings;
}

enum SyntheticDataMode{
  off @0;
  zeros @1;
  randomNormal @2;
  n @3;
}

struct SessionOptions{
  struct PoplarOptions{
    options @0: List(Option);
    struct Option{
      key @0: Text;
      value @1: Text;
    }
  }
}

enum AnchorReturnTypeId{
  final @0;
  everyN @1;
  all @2;
  sum @3;
}

struct AnchorReturnType{
  artStr @0: Text;
  artId @1: AnchorReturnTypeId;
  returnPeriod @2: Int32;
}

struct DataFlow{
  struct AnchorMap{
    anchors @0: List(ArtEntry);
    struct ArtEntry{
      id @0: TensorId;
      returnType @1: AnchorReturnType;
    }
  }

  batchesPerStep @0: Int32;
  anchorMap @1: AnchorMap;
}

struct Ir {
  enum ExecutionMode{
    inference @0;
    training @1;
  }
  dataFlow @0: DataFlow;
  sessionOptions @1: SessionOptions;
  requiresRandomSeed @2: Bool;
  executionMode @3: ExecutionMode;
  additionalModelProtoTensors @4: List(TensorId);
}
