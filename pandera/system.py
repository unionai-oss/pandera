"""Global variables relating to OS."""

import numpy as np

# Windows and Mac M1 don't support floats of this precision:
# https://github.com/pandera-dev/pandera/issues/623
FLOAT_128_AVAILABLE = hasattr(np, "float128")
