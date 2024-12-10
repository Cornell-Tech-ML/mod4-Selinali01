# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any, Dict

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Dict[str, Any]) -> Fn:
    """Create a device-specific JIT-compiled function.

    Args:
    ----
        fn: Function to compile.
        **kwargs: Additional arguments for JIT compilation.

    Returns:
    -------
        JIT-compiled function for CUDA device.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Callable, **kwargs: Dict[str, Any]) -> FakeCUDAKernel:
    """Create a JIT-compiled CUDA kernel.

    Args:
    ----
        fn: Function to compile into a CUDA kernel.
        **kwargs: Additional arguments for JIT compilation.

    Returns:
    -------
        Compiled CUDA kernel.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    """CUDA implementation of tensor operations."""

    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """Applies a binary function to corresponding elements of two tensors.

        Args:
        ----
            fn: Binary function to apply element-wise.

        Returns:
        -------
            Function that performs element-wise operation on two tensors.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Creates a reduction function along a specific dimension.

        Args:
        ----
            fn: Binary reduction function.
            start: Initial value for reduction.

        Returns:
        -------
            Function that performs reduction along specified dimension.

        """
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Performs matrix multiplication of two tensors.

        Args:
        ----
            a: First input tensor.
            b: Second input tensor.

        Returns:
        -------
            Result of matrix multiplication.

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        # TODO: Implement for Task 3.3.
        # Out of bounds
        if i >= out_size:
            return

        # Convert position to indices
        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, in_shape, in_index)

        # Apply function and store result
        out[i] = fn(in_storage[index_to_position(in_index, in_strides)])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if i >= out_size:
            return

        # Convert position to indices
        to_index(i, out_shape, out_index)
        broadcast_index(out_index, out_shape, a_shape, a_index)
        broadcast_index(out_index, out_shape, b_shape, b_index)

        # Apply function and store result
        a_val = a_storage[index_to_position(a_index, a_strides)]
        b_val = b_storage[index_to_position(b_index, b_strides)]
        out[i] = fn(a_val, b_val)

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Practice sum kernel to prepare for reduce operation.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out: Storage for output tensor.
        a: Storage for input tensor.
        size: Length of input storage.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    # TODO: Implement for Task 3.3.
    # Load data into shared memory
    cache[pos] = a[i] if i < size else 0.0
    cuda.syncthreads()

    # Parallel reduction in shared memory
    stride = BLOCK_DIM // 2
    while stride > 0:
        if pos < stride:
            cache[pos] += cache[pos + stride]
        cuda.syncthreads()
        stride //= 2

    # Write result for this block
    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Performs a practice sum reduction on a tensor using CUDA.

    Args:
    ----
        a: Input tensor.

    Returns:
    -------
        Reduced tensor data.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        # TODO: Implement for Task 3.3.
        if out_pos >= out_size:
            return

        # Convert output position to index
        to_index(out_pos, out_shape, out_index)

        # Initialize shared memory
        cache[pos] = reduce_value

        # Loop over reduce dimension
        for j in range(pos, a_shape[reduce_dim], BLOCK_DIM):
            # Copy out_index for the reduced dimension
            out_index[reduce_dim] = j
            # Load and reduce
            val = a_storage[index_to_position(out_index, a_strides)]
            cache[pos] = fn(cache[pos], val)

        cuda.syncthreads()

        # Parallel reduction in shared memory
        stride = BLOCK_DIM // 2
        while stride > 0:
            if pos < stride:
                cache[pos] = fn(cache[pos], cache[pos + stride])
            cuda.syncthreads()
            stride //= 2

        # Write final reduced value
        if pos == 0:
            out[out_pos] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    r"""Practice square matrix multiplication kernel.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:
        * All data must be first moved to shared memory.
        * Only read each cell in `a` and `b` once.
        * Only write to global memory once per kernel.

    Args:
    ----
        out: Storage for output tensor.
        a: Storage for first input tensor.
        b: Storage for second input tensor.
        size: Size of the square matrices.

    """
    BLOCK_DIM = 32
    # TODO: Implement for Task 3.3.
    # Create shared memory arrays for a and b matrices
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Get thread indices
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y

    # Initialize accumulator for dot product
    accumulator = 0.0

    # Load data into shared memory
    if i < size and j < size:
        # Load elements from matrix a and b into shared memory
        a_shared[i, j] = a[i * size + j]  # Using stride [size, 1]
        b_shared[i, j] = b[i * size + j]  # Using stride [size, 1]

    # Ensure all threads have loaded their data
    cuda.syncthreads()

    # Compute matrix multiplication
    if i < size and j < size:
        # Compute the dot product for this thread's element
        for k in range(size):
            accumulator += a_shared[i, k] * b_shared[k, j]

        # Write result to global memory
        out[i * size + j] = accumulator


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice matrix multiplication using CUDA.

    Args:
    ----
        a: First input tensor.
        b: Second input tensor.

    Returns:
    -------
        Result of matrix multiplication.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0
    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    BLOCK_DIM = 32
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    # The local position in the block.
    pi = cuda.threadIdx.x
    pj = cuda.threadIdx.y

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]
    # TODO: Implement for Task 3.4.
    # Initialize accumulator for dot product
    MAX_BLOCKS = a_shape[2]
    accumalated_sum = 0.0
    for block_offset in range(0, MAX_BLOCKS, BLOCK_DIM):  # iterate over each block
        local_j_offset = block_offset + pj  # offset of local j index
        local_i_offset = block_offset + pi  # offset of local i index
        if i < a_shape[1] and local_j_offset < MAX_BLOCKS:
            a_index = (
                a_batch_stride * batch
                + a_strides[1] * i
                + a_strides[2] * local_j_offset
            )  # Getting position of a_storage
            a_shared[pi, pj] = a_storage[a_index]  # Copying into shared memory

        # b) Copy into shared memory for b matrix
        # Load elements of tensor 'a' into shared memory
        if local_i_offset < b_shape[1] and j < b_shape[2]:
            b_index = (
                b_batch_stride * batch
                + b_strides[1] * local_i_offset
                + b_strides[2] * j
            )  # Getting position of a_storage
            b_shared[pi, pj] = b_storage[b_index]  # Copying into shared memory

        cuda.syncthreads()  # Synchronize threads in the block to ensure shared memory is fully populated

        # Compute dot product for elements in the shared block and sums over the relevant row and column for a and b respectively
        for k in range(BLOCK_DIM):
            if (k + block_offset) < MAX_BLOCKS:
                accumalated_sum += a_shared[pi, k] * b_shared[k, pj]

    if i < out_shape[1] and j < out_shape[2]:
        out_loc = out_strides[0] * batch + out_strides[1] * i + out_strides[2] * j
        out[out_loc] = accumalated_sum


tensor_matrix_multiply = jit(_tensor_matrix_multiply)
