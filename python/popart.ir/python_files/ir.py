# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Definition of a class that represents the PopART IR."""
import inspect
from weakref import WeakValueDictionary
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Iterable, Union
from collections import OrderedDict
from pathlib import Path
import os

import popart
import popart._internal.ir as _ir
from popart.ir.graph import Graph
from popart.ir.module import Module
from popart.ir.tensor import Tensor, TensorByRef, subgraph_input, subgraph_output, TensorSpec

if TYPE_CHECKING:
    IrCache = WeakValueDictionary[int, 'Ir']


class Ir:
    """
    Class that represents the PopART IR.

    This class contains a main graph. Furthermore, it defines methods and
    decorators for creating additional graphs from Python functions.
    """

    def __init__(self):
        """Initialises a new IR."""
        self._pb_ir = _ir.Ir()
        # Set better defaults for popart.ir programs.
        # Some parts of graph construction use the session options to make decisions,
        # such as inheritPlacementAttributes and scheduling priorities. So we must set
        # the options when we construct the _internal IR before the user starts constructing
        # the graphs.
        opts = self._pb_ir.getSessionOptions()
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        opts.useHostCopyOps = True
        opts.aliasZeroCopy = True
        opts.enableExplicitMainLoops = True
        opts.explicitRecomputation = True
        opts.enableInplaceAmbiguityChecking = True

        Ir._ir_cache[self.id] = self
        self._graph_cache: Dict[str, Graph] = {}

    _ir_cache: 'IrCache' = WeakValueDictionary()

    @classmethod
    def _from_pb(
            cls,
            pb_ir: '_ir.Ir',
    ) -> 'Ir':
        """Factory method to return popart.ir.Ir for a pybind _ir.Ir.

        Args:
            pb_ir (_ir.Ir):
                An instance of the low-level pybind11 `Ir`.

        Raises:
            RuntimeError: If the pb_ir does not already have a popart.ir Ir in-memory.

        Returns:
            Ir:
                A popart.ir.Ir that represents the passed pb_ir.
        """
        _id = pb_ir.getId()
        if _id not in Ir._ir_cache:
            raise RuntimeError(
                "Constructing a new Ir with _from_pb is unexpected. "
                "This implies the Ir was garbage collected. "
                "Is a popart.ir class missing a back reference to Ir?")
        return Ir._ir_cache[_id]

    def main_graph(self) -> 'Graph':
        """Every IR is initialised with a main graph. This method returns this
        graph.

        Returns:
            Graph:
                The main graph of the IR.
        """
        return Graph._from_pb(self._pb_ir.getMainGraph())

    def create_graph(
            self,
            fn: Union[Callable[..., Union[None, Tensor, Iterable[Tensor]]],
                      Module],
            *args: Any,
            **kwargs: Any,
    ) -> 'Graph':
        """
        Create a subgraph from a Python callable `fn` or the build method of a `Module`.
        The graph inputs are determined using the signature of the function `fn`
        and the supplied arguments `args` and `kwargs`. Tensors or TensorSpecs passed via the
        arguments are used to determine the shape and dtype of the graph inputs (the
        tensors are not actually passed to the graph). The graph outputs are
        determined using the outputs of the function when called.

        The order of inputs in the returned subgraph will be the same as the
        order of the tensor inputs in the function signature and the
        order of kwargs. This determines the order in which you pass the parent
        tensors as inputs at the callsite.

        The function `fn` can take any arguments. Any Tensor arguments are
        automatically detected. Any Tensor arguments inside a tuple, list,
        `*arg` or `**kwargs` are also detected. `*args`, `**kwargs`, lists
        cannot contain a mixture of tensors and other types. Nested lists
        or dicts of tensors are not supported.

        If an input is type hinted with `TensorByRef` or `List[TensorByRef]`
        where appropriate in the signature of `fn` then the corresponding inputs
        will be passed by reference instead of by value when the graph is called.

        The output of `fn` must be either None, a Tensor or an iterable of Tensors.

        Args:
            fn (Callable[..., Any]):
                The Python function that defines the graph. The signature of
                `fn` with its arguments is used to determine the
                inputs of the graph.
            args (Any):
                Arguments passed to the Python function that defines the graph
                that can be a mixture of tensors and other types. Tensors are
                used to determine the tensor info of the inputs.
            kwargs (Any):
                Keyword arguments passed to the Python function that defines the
                graph that can be a mixture of tensors and other types. Tensors
                are used to determine the tensor info of the inputs.

        Returns:
            Graph:
                A graph that corresponds to the input Python function.
        """
        if isinstance(fn, Module):
            qualname = fn.__class__.__qualname__
            func = fn.build
        else:
            # Note all Python functions will have __qualname__.
            if not callable(fn) or not hasattr(fn, '__qualname__'):
                raise TypeError(
                    "Callable `fn` must be either a function or a class that "
                    "extends popart.ir.Module")
            else:
                qualname = fn.__qualname__
                func = fn

        signature = inspect.signature(func, follow_wrapped=True)
        try:
            bound_args = signature.bind(*args, **kwargs)
        except TypeError as e:
            raise ValueError(
                "The arguments, `args` and `kwargs`, do not match the signature"
                " of the function `fn`.") from e
        bound_args.apply_defaults()
        arguments = bound_args.arguments

        name = self._create_name(qualname)
        _pb_subgraph = self._pb_ir.createGraph(name)
        subgraph = Graph._from_pb(_pb_subgraph)

        with subgraph:
            for name, arg in bound_args.arguments.items():
                type_hint = signature.parameters[name].annotation

                if isinstance(arg, (Tensor, TensorSpec)):
                    by_ref = type_hint is TensorByRef
                    arguments[name] = subgraph_input(arg.shape,
                                                     arg.dtype,
                                                     name,
                                                     by_ref=by_ref,
                                                     meta_shape=arg.meta_shape)

                # Supported args:
                # 1. Argument that is a list `def func(x: List[Tensor])`
                # 2. Variable length argument `def func(*x)`
                # 3. Variable keyword arguments `def func(**x)`
                elif isinstance(arg, (tuple, list, dict)):
                    by_ref = False
                    signature_kind = signature.parameters[name].kind

                    if (signature_kind is inspect.Parameter.VAR_POSITIONAL or
                            signature_kind is inspect.Parameter.VAR_KEYWORD):
                        # Variable length argument (*arg)
                        # or variable keyword argument (**kwarg)
                        by_ref = type_hint is TensorByRef
                    elif isinstance(arg, (tuple, list)):
                        # Argument that is a list
                        by_ref = type_hint is List[
                            TensorByRef] or type_hint is Tuple[TensorByRef]
                    elif isinstance(arg, dict):
                        # Argument that is a dict
                        continue

                    if isinstance(arg, dict):
                        items = arg.items()
                    else:
                        items = zip([name + f'_{i}' for i in range(len(arg))],
                                    arg)

                    in_args_sub = OrderedDict()
                    contains_tensor = False

                    for i, (subarg_name, subarg) in enumerate(items):
                        if i > 0 and (isinstance(subarg, (Tensor, TensorSpec))
                                      != contains_tensor):
                            raise TypeError(
                                f"A {type(arg)} argument can't contain a "
                                f"mixture of Tensors and other types. Arg name:"
                                f" {name}. Value: {arg}")
                        if isinstance(subarg, Tensor):
                            contains_tensor = True
                            in_args_sub[subarg_name] = subgraph_input(
                                subarg.shape,
                                subarg.dtype,
                                subarg_name,
                                by_ref=by_ref,
                                meta_shape=subarg.meta_shape)
                    if contains_tensor and isinstance(arg, dict):
                        arguments[name] = in_args_sub
                    elif contains_tensor and isinstance(arg, (tuple, list)):
                        arguments[name] = list(in_args_sub.values())

            bounds_args_new = inspect.BoundArguments(signature, arguments)
            outputs = fn(*bounds_args_new.args, **bounds_args_new.kwargs)

            if not (outputs is None or isinstance(outputs, Tensor) or
                    (isinstance(outputs, Iterable)
                     and all(isinstance(e, Tensor) for e in outputs))):
                raise ValueError(
                    "Output of subgraph must be None, a Tensor or an iterable of Tensors."
                    f" Output type: {type(outputs)}. Value {outputs}")

            if outputs is None:
                outputs = []

            if isinstance(outputs, Tensor):
                outputs = (outputs, )

            for out in outputs:
                subgraph_output(out)

        return subgraph

    def create_empty_graph(self, name: Optional[str] = None) -> 'Graph':
        """Create a new graph.

        Args:
            name (Optional[str]): Name of the graph. Defaults to "graph".

        Returns:
            Graph: An empty graph.
        """
        name = self._create_name(name or "graph")
        _pb_subgraph = self._pb_ir.createGraph(
            name)  # type: ignore GraphId != str
        return Graph._from_pb(_pb_subgraph)

    def dot_checkpoint(self,
                       check: str,
                       save_dir: Optional[Union[Path, str]] = None) -> None:
        """Output a graphical representation of the graph in Graphviz DOT format.

        Checkpoints can be activated by either setting the `dotChecks` option in session
        options or the `POPART_DOT_CHECKS` environmental variable. These should be set to
        the list of the checks to be activated.
        Note that if either `dotChecks` or `POPART_DOT_CHECKS` is set to `ALL`, all checkpoints
        will be activated.
        See the `PopART User Guide <https://docs.graphcore.ai/projects/popart-user-guide/en/latest/env_vars.html#generating-dot-files>`__ for more information.

        If no checkpoints are activated, this function will activate them all by setting the `dotChecks` option to `ALL`.

        Args:
            check (str): Name of this checkpoint.
            save_dir (Optional[Union[Path, str]]): Directory to store the dot files in.
              Note that this will set the save directory for all dot checkpoints in the graph.
        """
        opts = self._pb_ir.getSessionOptions()
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            opts.logDir = str(save_dir)

        if "POPART_DOT_CHECKS" not in os.environ:
            if len(opts.dotChecks) == 0:
                opts.dotChecks = {
                    "ALL",
                }

        self._pb_ir.dotCheckpoint(self._pb_ir, check)

    def _create_name(self, name: str) -> str:
        """Generate a graph name based on the qualified name of the Python
        function that created it.

        Note: occurrences of ".<locals>" in the name are removed.

        Example:
            Suppose a graph function:
                >>> class Foo:
                ...     def bar():
                ...         # Graph definition...
            Creating the following graphs:
                >>> ir.create_graph(Foo.bar)
                >>> ir.create_graph(Foo.bar)
            will result in graph names `Foo.bar_0` and `Foo.bar_1`.

        Args:
            name (str):
                The `__qualname__` attribute of the Python function.

        Returns:
            str:
                The name of the graph.
        """
        name = name.replace(".<locals>", "")
        name = self._pb_ir.createUniqueSubgraphId(name)
        return name

    @property
    def id(self) -> int:
        return self._pb_ir.getId()

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, value: 'Ir') -> bool:
        if not isinstance(value, Ir):
            raise TypeError(
                f"Value must be of type pir.Ir. Value: {value}. Type: {type(value)}"
            )
        return self.id == value.id

    def __repr__(self) -> str:
        return f"Ir[id={self.id}]"
