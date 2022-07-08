# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
import popart
import json

# Example how customer make instrument their own python code

# Unfortunately python default arguments are evaluated once when the function
# is defined, not when it is called, so we can not use the default argument
# trick to set the debugContext.


# bar method takes an argument and auto creates a debug context if not supplied
def bar(arg, debugContext):
    # Create a debug info for the `unit_test` layer and attach to the debug context
    di = popart.DebugInfo(debugContext, "unit_test")
    # Add fields to the debug info
    di.setValue("arg", str(arg))


# foo method takes an argument and auto creates a debug context if not supplied
def foo(arg, debugContext):

    # Create a debug info for the `unit_test` layer and attach to the debug context
    di = popart.DebugInfo(debugContext, "unit_test")
    # Add fields to the debug info
    di.setValue("arg", str(arg))

    # Call bar
    # propagate the debug info to the method and name it `bar`
    bar(123, debugContext=popart.DebugContext(di, "bar"))


def test_basic(tmpdir):
    filename = str(tmpdir) + "/debug.json"
    popart.initializePoplarDebugInfo(filename, "json")

    # Calling method
    foo(42, popart.DebugContext())

    popart.closePoplarDebugInfo()

    with open(filename, encoding="utf-8") as json_file:
        data = json.load(json_file)
        print(data)

        # Expect 2 contexts one for foo and one for bar
        assert len(data["contexts"]) == 2

        # Written out backwards
        barContext = data["contexts"][0]
        fooContext = data["contexts"][1]

        assert data["stringTable"][int(fooContext["name"])] == ""
        assert data["stringTable"][int(barContext["name"])] == "bar"

        assert barContext["parentId"] == 1

        assert barContext["layer"] == "unit_test"
        assert fooContext["layer"] == "unit_test"

        assert barContext["arg"] == "123"
        assert fooContext["arg"] == "42"

        # Verify the location inforamation
        barLocation = barContext["location"]
        fooLocation = fooContext["location"]

        assert data["stringTable"][int(fooLocation["fileName"])].endswith(
            "debug_info_test.py"
        )
        assert data["stringTable"][int(barLocation["fileName"])].endswith(
            "debug_info_test.py"
        )

        assert data["stringTable"][int(fooLocation["functionName"])] == "test_basic"
        assert data["stringTable"][int(barLocation["functionName"])] == "foo"

        assert fooLocation["lineNumber"] == 38
        assert barLocation["lineNumber"] == 30
