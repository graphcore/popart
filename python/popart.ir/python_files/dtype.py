class dtype:
    @property
    def is_complex(self) -> bool:
        return self._is_complex

    @property
    def is_floating_point(self) -> bool:
        return self._is_floating_point

    @property
    def is_signed(self) -> bool:
        return self._is_signed

    def __repr__(self) -> str:
        return self._name


def _register_dtype(name: str, is_complex: bool, is_floating_point: bool,
                    is_signed: bool) -> dtype:
    t = dtype()
    t._is_complex = is_complex
    t._is_floating_point = is_floating_point
    t._is_signed = is_signed
    t._name = name
    return t


uint8 = _register_dtype(
    name='uint8',
    is_complex=False,
    is_floating_point=False,
    is_signed=False,
)
int8 = _register_dtype(
    name='int8',
    is_complex=False,
    is_floating_point=False,
    is_signed=True,
)
uint16 = _register_dtype(
    name='uint16',
    is_complex=False,
    is_floating_point=False,
    is_signed=False,
)
int16 = _register_dtype(
    name='int16',
    is_complex=False,
    is_floating_point=False,
    is_signed=True,
)
uint32 = _register_dtype(
    name='uint32',
    is_complex=False,
    is_floating_point=False,
    is_signed=False,
)
int32 = _register_dtype(
    name='int32',
    is_complex=False,
    is_floating_point=False,
    is_signed=True,
)
uint64 = _register_dtype(
    name='uint64',
    is_complex=False,
    is_floating_point=False,
    is_signed=False,
)
int64 = _register_dtype(
    name='int64',
    is_complex=False,
    is_floating_point=False,
    is_signed=True,
)
bool = _register_dtype(
    name='bool',
    is_complex=False,
    is_floating_point=False,
    is_signed=False,
)
float16 = _register_dtype(
    name='float16',
    is_complex=False,
    is_floating_point=True,
    is_signed=True,
)
float32 = _register_dtype(
    name='float32',
    is_complex=False,
    is_floating_point=True,
    is_signed=True,
)
float64 = _register_dtype(
    name='float64',
    is_complex=False,
    is_floating_point=True,
    is_signed=True,
)

__all__ = [name for name in dir() if name[0] != '_']