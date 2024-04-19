
# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# Source Nodes: [add, add_1, contrast, max_1, min_1, sub], Original ATen: [aten.add, aten.div, aten.max, aten.min, aten.sub]
# add => add_2
# add_1 => add_3
# contrast => div
# max_1 => max_1
# min_1 => min_1
# sub => sub
triton_per_fused_add_div_max_min_sub_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_max_min_sub_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'c1d84c71ed30354d85375ee7980e62864c69d0c6c9b5d255fd82e100eab6423d'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + ((256*(tl.minimum(255, tl.maximum(0, (-1) + (r1 // 3) + (x0 // 256))))) + (tl.minimum(255, tl.maximum(0, (-1) + (r1 % 3) + (x0 % 256))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp6 = tl.where(rmask, tmp1, float("inf"))
    tmp7 = triton_helpers.min2(tmp6, 1)[:, None]
    tmp8 = tmp4 - tmp7
    tmp9 = tmp4 + tmp7
    tmp10 = 1e-10
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 / tmp11
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp12, None)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# Source Nodes: [bin_edges, cumsum_2, mul, weight1], Original ATen: [aten.cumsum, aten.linspace, aten.mul]
# bin_edges => add_4, convert_element_type, convert_element_type_1, iota_6, lt, mul, mul_1, sub_1, sub_2, where
# cumsum_2 => cumsum_2
# mul => mul_2
# weight1 => cumsum
triton_per_fused_cumsum_linspace_mul_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints=[1, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {3: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_linspace_mul_1', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': 'c1d84c71ed30354d85375ee7980e62864c69d0c6c9b5d255fd82e100eab6423d'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3, = tl.associative_scan((tmp2,), 0, _triton_helper_fn_add0)
    tmp4 = r0
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 128.0
    tmp7 = tmp5 < tmp6
    tmp8 = 0.00392156862745098
    tmp9 = tmp5 * tmp8
    tmp10 = 0.0
    tmp11 = tmp9 + tmp10
    tmp12 = 255 + ((-1)*r0)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp8
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tl.where(tmp7, tmp11, tmp16)
    tmp18 = tmp0 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp21, = tl.associative_scan((tmp20,), 0, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp21, rmask)
''', device_str='cuda')


# Source Nodes: [bin_edges, cumsum_1, cumsum_3, flip, flip_2, mul_1], Original ATen: [aten.cumsum, aten.flip, aten.linspace, aten.mul]
# bin_edges => add_4, convert_element_type, convert_element_type_1, iota_6, lt, mul, mul_1, sub_1, sub_2, where
# cumsum_1 => cumsum_1
# cumsum_3 => cumsum_3
# flip => rev
# flip_2 => rev_2
# mul_1 => mul_3
triton_per_fused_cumsum_flip_linspace_mul_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton.jit
def _triton_helper_fn_add0(arg0_0, arg1_0):
    tmp0 = arg0_0 + arg1_0
    return tmp0

@triton_heuristics.persistent_reduction(
    size_hints=[1, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {3: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_flip_linspace_mul_2', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': 'c1d84c71ed30354d85375ee7980e62864c69d0c6c9b5d255fd82e100eab6423d'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, rnumel):
    xnumel = 1
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    roffset = 0
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (255 + ((-1)*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp3, = tl.associative_scan((tmp2,), 0, _triton_helper_fn_add0)
    tmp4 = 255 + ((-1)*r0)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = 128.0
    tmp7 = tmp5 < tmp6
    tmp8 = 0.00392156862745098
    tmp9 = tmp5 * tmp8
    tmp10 = 0.0
    tmp11 = tmp9 + tmp10
    tmp12 = r0
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp8
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tl.where(tmp7, tmp11, tmp16)
    tmp18 = tmp0 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp21, = tl.associative_scan((tmp20,), 0, _triton_helper_fn_add0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp3, rmask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp21, rmask)
''', device_str='cuda')


# Source Nodes: [idx, mul_2, pow_1, sub_1, variance12], Original ATen: [aten.argmax, aten.mul, aten.pow, aten.sub]
# idx => argmax
# mul_2 => mul_4
# pow_1 => pow_1
# sub_1 => sub_3
# variance12 => mul_5
triton_red_fused_argmax_mul_pow_sub_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.reduction(
    size_hints=[1, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {5: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_argmax_mul_pow_sub_3', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'c1d84c71ed30354d85375ee7980e62864c69d0c6c9b5d255fd82e100eab6423d'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 255
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp11 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    _tmp11_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (254 + ((-1)*r0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (254 + ((-1)*r0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 * tmp1
        tmp4 = tmp3 / tmp0
        tmp6 = tmp5 / tmp1
        tmp7 = tmp4 - tmp6
        tmp8 = tmp7 * tmp7
        tmp9 = tmp2 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        _tmp11_next, _tmp11_index_next = triton_helpers.maximum_with_index(
            _tmp11, _tmp11_index, tmp10, rindex
        )
        _tmp11 = tl.where(rmask, _tmp11_next, _tmp11)
        _tmp11_index = tl.where(rmask, _tmp11_index_next, _tmp11_index)
    _, tmp11_tmp = triton_helpers.max_with_index(_tmp11, _tmp11_index, 1)
    tmp11 = tmp11_tmp[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp11, None)
''', device_str='cuda')


# Source Nodes: [e_sum, hi_contrast_count, mul_4], Original ATen: [aten.mul, aten.sum]
# e_sum => sum_2
# hi_contrast_count => sum_1
# mul_4 => mul_6
triton_per_fused_mul_sum_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mul_sum_4', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'c1d84c71ed30354d85375ee7980e62864c69d0c6c9b5d255fd82e100eab6423d'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + ((256*(tl.minimum(255, tl.maximum(0, (-1) + (r1 // 3) + (x0 // 256))))) + (tl.minimum(255, tl.maximum(0, (-1) + (r1 % 3) + (x0 % 256))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp14 = tl.load(in_ptr2 + ((256*(tl.minimum(255, tl.maximum(0, (-1) + (r1 // 3) + (x0 // 256))))) + (tl.minimum(255, tl.maximum(0, (-1) + (r1 % 3) + (x0 % 256))))), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.00390625
    tmp5 = tmp3 * tmp4
    tmp6 = tmp0 < tmp5
    tmp7 = 0.0
    tmp8 = 1.0
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = tl.sum(tmp12, 1)[:, None]
    tmp15 = tmp14 * tmp9
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, None)
    tl.store(out_ptr1 + (x0), tmp19, None)
''', device_str='cuda')


# Source Nodes: [e_mean, e_mean_1, isnan, mul_5, sub_2], Original ATen: [aten.div, aten.isnan, aten.mul, aten.scalar_tensor, aten.sub, aten.where]
# e_mean => div_4
# e_mean_1 => full_default_2, where_2
# isnan => isnan
# mul_5 => mul_7
# sub_2 => sub_4
triton_poi_fused_div_isnan_mul_scalar_tensor_sub_where_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.pointwise(
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_isnan_mul_scalar_tensor_sub_where_5', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'c1d84c71ed30354d85375ee7980e62864c69d0c6c9b5d255fd82e100eab6423d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 65536
    x1 = (xindex // 65536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((256*(tl.minimum(255, tl.maximum(0, (-1) + (x0 // 256) + (x1 // 3))))) + (tl.minimum(255, tl.maximum(0, (-1) + (x0 % 256) + (x1 % 3))))), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + ((256*(tl.minimum(255, tl.maximum(0, (-1) + (x0 // 256) + (x1 // 3))))) + (tl.minimum(255, tl.maximum(0, (-1) + (x0 % 256) + (x1 % 3))))), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp3 = tmp1 / tmp2
    tmp4 = libdevice.isnan(tmp3).to(tl.int1)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tmp7 = tmp0 - tmp6
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 0.00390625
    tmp13 = tmp11 * tmp12
    tmp14 = tmp8 < tmp13
    tmp15 = 1.0
    tmp16 = tl.where(tmp14, tmp5, tmp15)
    tmp17 = tmp7 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# Source Nodes: [e_std, result, setitem, square], Original ATen: [aten.index_put, aten.lift_fresh, aten.mean, aten.pow, aten.zeros_like]
# e_std => mean
# result => full_default_4
# setitem => full_default_5, index_put
# square => pow_2
triton_per_fused_index_put_lift_fresh_mean_pow_zeros_like_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor import triton_helpers, triton_heuristics
from torch._inductor.ir import ReductionHint, TileHint
from torch._inductor.triton_helpers import libdevice, math as tl_math
from torch._inductor.triton_heuristics import AutotuneHint
from torch._inductor.utils import instance_descriptor

@triton_heuristics.persistent_reduction(
    size_hints=[65536, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_index_put_lift_fresh_mean_pow_zeros_like_6', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'c1d84c71ed30354d85375ee7980e62864c69d0c6c9b5d255fd82e100eab6423d'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 65536
    rnumel = 9
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (65536*r1)), rmask, other=0.0)
    tmp6 = tl.load(in_out_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp7 = 3.0
    tmp8 = tmp6 >= tmp7
    tmp11 = tmp10 / tmp6
    tmp12 = libdevice.isnan(tmp11).to(tl.int1)
    tmp13 = 0.0
    tmp14 = tl.where(tmp12, tmp13, tmp11)
    tmp15 = 9.0
    tmp16 = tmp5 / tmp15
    tmp17 = libdevice.sqrt(tmp16)
    tmp18 = libdevice.isnan(tmp17).to(tl.int1)
    tmp19 = tl.where(tmp18, tmp13, tmp17)
    tmp20 = 0.5
    tmp21 = tmp19 * tmp20
    tmp22 = tmp14 + tmp21
    tmp23 = tmp9 <= tmp22
    tmp24 = tmp8 & tmp23
    tmp25 = 1.0
    tmp26 = tl.where(tmp24, tmp25, tmp13)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp26, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 256, 256), (65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((65536, ), (1, ), torch.float32)
        buf4 = buf0; del buf0  # reuse
        # Source Nodes: [add, add_1, contrast, max_1, min_1, sub], Original ATen: [aten.add, aten.div, aten.max, aten.min, aten.sub]
        stream0 = get_raw_stream(0)
        triton_per_fused_add_div_max_min_sub_0.run(buf4, arg0_1, 65536, 9, grid=grid(65536), stream=stream0)
        # Source Nodes: [counts_1], Original ATen: [aten.histc]
        buf5 = aten.histc.default(buf4, 256, 0, 1)
        buf6 = buf5
        del buf5
        buf7 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf9 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [bin_edges, cumsum_2, mul, weight1], Original ATen: [aten.cumsum, aten.linspace, aten.mul]
        triton_per_fused_cumsum_linspace_mul_1.run(buf6, buf7, buf9, 1, 256, grid=grid(1), stream=stream0)
        buf8 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf10 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [bin_edges, cumsum_1, cumsum_3, flip, flip_2, mul_1], Original ATen: [aten.cumsum, aten.flip, aten.linspace, aten.mul]
        triton_per_fused_cumsum_flip_linspace_mul_2.run(buf6, buf8, buf10, 1, 256, grid=grid(1), stream=stream0)
        del buf6
        buf11 = empty_strided_cuda((), (), torch.int64)
        # Source Nodes: [idx, mul_2, pow_1, sub_1, variance12], Original ATen: [aten.argmax, aten.mul, aten.pow, aten.sub]
        triton_red_fused_argmax_mul_pow_sub_3.run(buf7, buf8, buf9, buf10, buf11, 1, 255, grid=grid(1), stream=stream0)
        del buf10
        del buf7
        del buf8
        del buf9
        buf12 = empty_strided_cuda((65536, ), (1, ), torch.float32)
        buf13 = empty_strided_cuda((65536, ), (1, ), torch.float32)
        # Source Nodes: [e_sum, hi_contrast_count, mul_4], Original ATen: [aten.mul, aten.sum]
        triton_per_fused_mul_sum_4.run(buf4, buf11, arg0_1, buf12, buf13, 65536, 9, grid=grid(65536), stream=stream0)
        buf14 = empty_strided_cuda((9, 65536), (65536, 1), torch.float32)
        # Source Nodes: [e_mean, e_mean_1, isnan, mul_5, sub_2], Original ATen: [aten.div, aten.isnan, aten.mul, aten.scalar_tensor, aten.sub, aten.where]
        triton_poi_fused_div_isnan_mul_scalar_tensor_sub_where_5.run(arg0_1, buf13, buf12, buf4, buf11, buf14, 589824, grid=grid(589824), stream=stream0)
        del buf11
        del buf4
        buf16 = reinterpret_tensor(buf12, (1, 256, 256), (65536, 256, 1), 0); del buf12  # reuse
        # Source Nodes: [e_std, result, setitem, square], Original ATen: [aten.index_put, aten.lift_fresh, aten.mean, aten.pow, aten.zeros_like]
        triton_per_fused_index_put_lift_fresh_mean_pow_zeros_like_6.run(buf16, buf14, arg0_1, buf13, 65536, 9, grid=grid(65536), stream=stream0)
        del arg0_1
        del buf13
        del buf14
    return (buf16, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 256, 256), (65536, 256, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([arg0_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
