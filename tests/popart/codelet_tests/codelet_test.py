import numpy as np
import pytest
import popart
import os

# `import test_util` requires adding to sys.path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import test_util as tu

cpp_str = """
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>

using namespace poplar;

static constexpr auto ONE_PTR = VectorLayout::ONE_PTR;
static constexpr auto SPAN = VectorLayout::SPAN;

namespace testops {

template <typename type> class TestVertex : public Vertex
{
public:
    Input<Vector<type, ONE_PTR>> params;
    Input<Vector<unsigned, ONE_PTR>> indices;
    Output<Vector<type, ONE_PTR>> out;
    int indicesSize;

    bool compute()
    {
        for (int index = 0; index < indicesSize; ++index)
        {
            out[index] = static_cast<type>(1);
        }
        return true;
    }
};

} // end namespace testops
"""


def create_cpp_file(string, filename):
    with open(filename, "w") as f:
        f.write(string)


@tu.requires_ipu_model
def test_codelet():
    """Write a cpp file using the string above, then attempt to load it as
    a custom codelet. We assume poplar will handle invalid codelet code, so
    as long as poplar loads it, we are good. We skip this test if we have
    trouble writing to the filesystem to avoid unessecary buildbot failiures.
    """

    fname = "test_vertex.cpp"
    try:
        create_cpp_file(cpp_str, fname)
    except IOError:
        pytest.skip(f"Could not write to {fname}, skipping test")

    builder = popart.Builder()

    shape = popart.TensorInfo("FLOAT16", [2])

    i1 = builder.addInputTensor(shape)
    i2 = builder.addInputTensor(shape)
    o = builder.aiOnnx.add([i1, i2])
    builder.addOutputTensor(o)

    proto = builder.getModelProto()
    opts = popart.SessionOptions()
    opts.customCodelets = ["test_vertex.cpp"]

    dataFlow = popart.DataFlow(1, {o: popart.AnchorReturnType("ALL")})

    session = popart.InferenceSession(fnModel=proto,
                                      dataFeed=dataFlow,
                                      userOptions=opts,
                                      deviceInfo=tu.create_test_device())

    session.prepareDevice()

    anchors = session.initAnchorArrays()

    inputs = {
        i1: np.array([1., 3.], dtype=np.float16),
        i2: np.array([7., 8.], dtype=np.float16)
    }
    stepio = popart.PyStepIO(inputs, anchors)

    session.run(stepio)

    try:
        os.remove("test_vertex.cpp")
        print("File sucessfully removed")
    except IOError:
        print("Could not find file... Exiting")
