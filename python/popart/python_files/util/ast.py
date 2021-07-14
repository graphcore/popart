import ast
import inspect
from typing import Any, Iterable, Optional, Tuple, Union


def get_call_id(func) -> Optional[str]:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def get_op_id_to_ast_op(func_id):
    if func_id == "__matmul__":
        return ast.MatMult
    if func_id in ["__add__", "__iadd__"]:
        return ast.Add
    if func_id in ["__itruediv__"]:
        return ast.Div
    raise RuntimeError(f"Unknown ast.BinOp func_id {func_id}")


class DebugNameVisitor(ast.NodeVisitor):
    def __init__(self, function_id: str):
        super().__init__()
        self.id = function_id
        self.result: Optional[Tuple[str]] = None
        self.previous_assign_name = None
        self.try_higher = False

    def visit_Assign(self, node: ast.Assign) -> Any:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            self.previous_assign_name = (target.id, )
        elif isinstance(target, ast.Tuple):
            self.previous_assign_name = tuple(map(lambda t: t.id, target.elts))
        return super().generic_visit(node)

    def visit_AugAssign(self, node: ast.Assign) -> Any:
        func_op = get_op_id_to_ast_op(self.id)
        if isinstance(node.op, func_op):
            self.result = (node.target.id, )
        return super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        func_id = get_call_id(node.func)
        if func_id is not None and func_id == self.id:
            self.result = self.previous_assign_name
        return super().generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        func_op = get_op_id_to_ast_op(self.id)
        if isinstance(node.op, func_op):
            self.result = self.previous_assign_name
        return super().generic_visit(node)

    def visit_Return(self, node: ast.Return) -> Any:
        self.try_higher = True
        return super().generic_visit(node)


def get_class_name_from_frame(fr):
    """ from https://stackoverflow.com/a/2220759"""
    args, _, _, value_dict = inspect.getargvalues(fr)
    # we check the first parameter for the frame function is
    # named 'self'
    if len(args) and args[0] == 'self':
        # in that case, 'self' will be referenced in value_dict
        instance = value_dict.get('self', None)
        if instance:
            # return its class name
            return getattr(getattr(instance, '__class__', None), '__name__',
                           None)
    # return None otherwise
    return None


def tidy_code(code: Iterable[str]) -> str:
    def strip(s: str) -> str:
        return s.strip()

    code = map(strip, code)
    return ''.join(code)


AST_DEBUG = False


def parse_result(stack, frame_number):
    fn_frame = stack[frame_number]
    fn_name = fn_frame.function
    if fn_name == "__init__":
        fn_name = get_class_name_from_frame(fn_frame.frame)

    caller_frame = stack[frame_number + 1]
    tree = ast.parse(tidy_code(caller_frame.code_context))

    global AST_DEBUG
    if AST_DEBUG:
        print(frame_number)
        print(fn_name)
        print(ast.dump(tree))

    visitor = DebugNameVisitor(fn_name)
    visitor.visit(tree)
    return visitor


def parse_mutli_assign_from_ast(
        frame_number: int = 1) -> Union[Tuple[str], None]:
    try:
        stack = inspect.getouterframes(inspect.currentframe())
        visitor = parse_result(stack, frame_number)

        while visitor.result is None and visitor.try_higher:
            frame_number += 1
            visitor = parse_result(stack, frame_number)

        return visitor.result
    except Exception as e:
        if AST_DEBUG:
            print(e)
        pass
    return None


def parse_assign_from_ast(frame_number: int = 1) -> Union[str, None]:
    result = parse_mutli_assign_from_ast(frame_number + 1)
    if result:
        return result[0]
    return result
