# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import popart
import popxl
from popxl import ops


class TestWriteVariablesData:
    # Test that write_variable_data updates variable data on host and
    # that its weight is updated on device when using POPXL_CACHE_DIR, [A].
    # This test was failing, see ~T60847~. But was passing when POPXL_CACHE_DIR
    # was not used, [B].
    # The reason was that for [B] weights are populated by IR and we were
    # updating IR when calling write_variable_data.
    # For [A] the weights were not updated as in this case
    # they are not populated by IR but during second executablex constructor - deserialized
    # version. The issue was fixed by updating weights when calling write_variable_data.
    def test_write_variables_data_cache(self, tmp_path, monkeypatch):
        popart.getLogger().setLevel("DEBUG")
        monkeypatch.setenv("POPXL_CACHE_DIR", str(tmp_path / "cache"))
        ir = popxl.Ir()
        with ir.main_graph:
            v = popxl.variable(1, popxl.float32)
            d2h = popxl.d2h_stream(v.shape, v.dtype)
            ops.host_store(d2h, v)

        with popxl.Session(ir, "ipu_hw") as sess:
            assert sess.run()[d2h] == 1
            sess.write_variable_data(v, 2)
            assert sess.run()[d2h] == 2
            sess.write_variable_data(v, 3)
            assert sess.run()[d2h] == 3
