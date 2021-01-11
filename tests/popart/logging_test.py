import popart


def test_setLevel(capfd):
    def check_loglevel(level):
        popart.getLogger().setLevel(level)

        builder = popart.Builder()
        i1 = builder.addInputTensor(popart.TensorInfo("FLOAT", [1, 2, 3]))
        o = builder.aiOnnx.abs([i1])

        _, err = capfd.readouterr()
        return err

    # Change the log level from trace to off and then back to trace again.
    assert 'Adding ai.onnx.Abs' in check_loglevel('TRACE')
    assert check_loglevel('OFF') == ''
    assert 'Adding ai.onnx.Abs' in check_loglevel('TRACE')
