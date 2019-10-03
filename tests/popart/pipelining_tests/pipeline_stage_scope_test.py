import popart
import pytest


def test_get_pipeline_stage():
    builder = popart.Builder()

    # If the scope hasn't been set, hasPipelineStage will return False
    assert (builder.hasPipelineStage() == False)

    # If the scope hasn't been set, getPipelineStage will throw a runtime error
    with pytest.raises(popart.popart_exception) as excinfo:
        stage = builder.getPipelineStage()

    assert ("Pipeline stage not set in current scope." in str(excinfo.value))

    # When we enter a pipelineStage context, the builder should return that stage
    with builder.pipelineStage(0):
        assert (builder.hasPipelineStage() == True)
        assert (builder.getPipelineStage() == 0)

    with builder.pipelineStage(1):
        assert (builder.hasPipelineStage() == True)
        assert (builder.getPipelineStage() == 1)

    # Nested pipeline scopes are not currently supported. See below:
    with builder.pipelineStage(0):
        assert (builder.hasPipelineStage() == True)
        assert (builder.getPipelineStage() == 0)

        # Pipeline stage isn't stacked, so this has no effect, but...
        with builder.pipelineStage(1):
            assert (builder.hasPipelineStage() == True)
            assert (builder.getPipelineStage() == 0)

        # ...since we've now exited the context, pipeline stage will have been unset
        assert (builder.hasPipelineStage() == False)
