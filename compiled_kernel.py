
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
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# Source Nodes: [bin_edges], Original ATen: [aten.linspace]
# bin_edges => add_12, convert_element_type, convert_element_type_1, iota_6, lt, mul_1, mul_2, sub_13, sub_14, where
triton_poi_fused_linspace_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_linspace_0', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'cc5e07134fe44736250e1e295655d5c8951a39baf1ba36eb13cc712d6e9b780d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 128.0
    tmp3 = tmp1 < tmp2
    tmp4 = 0.00392156862745098
    tmp5 = tmp1 * tmp4
    tmp6 = 0.0
    tmp7 = tmp5 + tmp6
    tmp8 = 255 + ((-1)*x0)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp9 * tmp4
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = tl.where(tmp3, tmp7, tmp12)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
''', device_str='cuda')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, split_scan_grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream


# Source Nodes: [add, add_1, bin_edges, contrast, counts, idx, sub, weights, zeros], Original ATen: [aten.add, aten.bucketize, aten.div, aten.linspace, aten.ones_like, aten.scatter_reduce, aten.sub, aten.zeros]
# add => add_10
# add_1 => add_11
# bin_edges => add_12, convert_element_type, convert_element_type_1, iota_6, lt, mul_1, mul_2, sub_13, sub_14, where
# contrast => div
# counts => scatter_reduce
# idx => bucketize
# sub => sub_12
# weights => full
# zeros => full_default
triton_poi_fused_add_bucketize_div_linspace_ones_like_scatter_reduce_sub_zeros_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(), divisible_by_8=(1,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_bucketize_div_linspace_ones_like_scatter_reduce_sub_zeros_1', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'cc5e07134fe44736250e1e295655d5c8951a39baf1ba36eb13cc712d6e9b780d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# Source Nodes: [add, add_1, bin_edges, contrast, counts, idx, max_1, min_1, sub, weights, zeros], Original ATen: [aten.add, aten.bucketize, aten.div, aten.linspace, aten.max, aten.min, aten.ones_like, aten.scatter_reduce, aten.sub, aten.zeros]
# add => add_10
# add_1 => add_11
# bin_edges => add_12, convert_element_type, convert_element_type_1, iota_6, lt, mul_1, mul_2, sub_13, sub_14, where
# contrast => div
# counts => scatter_reduce
# idx => bucketize
# max_1 => max_1
# min_1 => min_1
# sub => sub_12
# weights => full
# zeros => full_default
triton_per_fused_add_bucketize_div_linspace_max_min_ones_like_scatter_reduce_sub_zeros_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[1048576, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': {AutotuneHint.ELEMENTS_PER_WARP_32}, 'kernel_name': 'triton_per_fused_add_bucketize_div_linspace_max_min_ones_like_scatter_reduce_sub_zeros_2', 'mutated_arg_names': ['out_ptr2'], 'no_x_dim': False, 'backend_hash': 'cc5e07134fe44736250e1e295655d5c8951a39baf1ba36eb13cc712d6e9b780d'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + ((ks1*(tl.math.min((-1) + ks0, tl.math.max(0, (-1) + (r1 // 3) + (x0 // ks1), tl.PropagateNan.NONE), tl.PropagateNan.NONE))) + (tl.math.min((-1) + ks1, tl.math.max(0, (-1) + (r1 % 3) + (x0 % ks1), tl.PropagateNan.NONE), tl.PropagateNan.NONE))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tmp6 = tl.where(rmask & xmask, tmp1, float("inf"))
    tmp7 = triton_helpers.min2(tmp6, 1)[:, None]
    tmp8 = tmp4 - tmp7
    tmp9 = tmp4 + tmp7
    tmp10 = 1e-10
    tmp11 = tmp9 + tmp10
    tmp12 = tmp8 / tmp11
    tmp13 = triton_helpers.bucketize_binary_search(tmp12, in_ptr1, tl.int64, False, 256, [XBLOCK, 1])
    tmp14 = tmp13 + 256
    tmp15 = tmp13 < 0
    tmp16 = tl.where(tmp15, tmp14, tmp13)
    tl.device_assert((0 <= tmp16) & (tmp16 < 256), "index out of bounds: 0 <= tmp16 < 256")
    tmp17 = 1.0
    tl.atomic_add(out_ptr2 + (tl.broadcast_to(tmp16, [XBLOCK, 1])), tmp17, xmask)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
''', device_str='cuda')


# Source Nodes: [cumsum_2, mul, weight1], Original ATen: [aten.cumsum, aten.mul]
# cumsum_2 => cumsum_2
# mul => mul_3
# weight1 => cumsum
triton_per_fused_cumsum_mul_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@triton.jit
def _triton_helper_fn0(arg0, arg1):
    tmp0 = arg0 + arg1
    return tmp0

@persistent_reduction(
    size_hints=[1, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {3: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(3,), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_mul_3', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': 'cc5e07134fe44736250e1e295655d5c8951a39baf1ba36eb13cc712d6e9b780d'}
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
    tmp2 = tl.full([1], 0, tl.float32)
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, tmp2)
    tmp4 = tl.associative_scan(tmp3, 0, _triton_helper_fn0)
    tmp5 = r0
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 128.0
    tmp8 = tmp6 < tmp7
    tmp9 = 0.00392156862745098
    tmp10 = tmp6 * tmp9
    tmp11 = 0.0
    tmp12 = tmp10 + tmp11
    tmp13 = 255 + ((-1)*r0)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp9
    tmp16 = 1.0
    tmp17 = tmp16 - tmp15
    tmp18 = tl.where(tmp8, tmp12, tmp17)
    tmp19 = tmp0 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp21 = tl.where(rmask, tmp20, tmp2)
    tmp22 = tl.associative_scan(tmp21, 0, _triton_helper_fn0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp4, rmask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp22, rmask)
''', device_str='cuda')


# Source Nodes: [cumsum_1, cumsum_3, flip, flip_2, mul_1], Original ATen: [aten.cumsum, aten.flip, aten.mul]
# cumsum_1 => cumsum_1
# cumsum_3 => cumsum_3
# flip => rev
# flip_2 => rev_2
# mul_1 => mul_4
triton_per_fused_cumsum_flip_mul_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@triton.jit
def _triton_helper_fn0(arg0, arg1):
    tmp0 = arg0 + arg1
    return tmp0

@persistent_reduction(
    size_hints=[1, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {3: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(3,), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cumsum_flip_mul_4', 'mutated_arg_names': [], 'no_x_dim': True, 'backend_hash': 'cc5e07134fe44736250e1e295655d5c8951a39baf1ba36eb13cc712d6e9b780d'}
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
    tmp2 = tl.full([1], 0, tl.float32)
    tmp0 = tl.load(in_ptr0 + (255 + ((-1)*r0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, tmp2)
    tmp4 = tl.associative_scan(tmp3, 0, _triton_helper_fn0)
    tmp5 = 255 + ((-1)*r0)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 128.0
    tmp8 = tmp6 < tmp7
    tmp9 = 0.00392156862745098
    tmp10 = tmp6 * tmp9
    tmp11 = 0.0
    tmp12 = tmp10 + tmp11
    tmp13 = r0
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp14 * tmp9
    tmp16 = 1.0
    tmp17 = tmp16 - tmp15
    tmp18 = tl.where(tmp8, tmp12, tmp17)
    tmp19 = tmp0 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp21 = tl.where(rmask, tmp20, tmp2)
    tmp22 = tl.associative_scan(tmp21, 0, _triton_helper_fn0)
    tl.store(out_ptr0 + (tl.broadcast_to(r0, [RBLOCK])), tmp4, rmask)
    tl.store(out_ptr1 + (tl.broadcast_to(r0, [RBLOCK])), tmp22, rmask)
''', device_str='cuda')


# Source Nodes: [idx_1, mul_2, pow_1, sub_1, variance12], Original ATen: [aten.argmax, aten.mul, aten.pow, aten.sub]
# idx_1 => argmax
# mul_2 => mul_5
# pow_1 => pow_1
# sub_1 => sub_15
# variance12 => mul_6
triton_red_fused_argmax_mul_pow_sub_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@reduction(
    size_hints=[1, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {5: 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(5,), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_argmax_mul_pow_sub_5', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'cc5e07134fe44736250e1e295655d5c8951a39baf1ba36eb13cc712d6e9b780d'}
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


# Source Nodes: [hi_contrast_count], Original ATen: [aten.sum]
# hi_contrast_count => sum_1
triton_per_fused_sum_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[1048576, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_6', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'cc5e07134fe44736250e1e295655d5c8951a39baf1ba36eb13cc712d6e9b780d'}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + ((ks1*(tl.math.min((-1) + ks0, tl.math.max(0, (-1) + (r1 // 3) + (x0 // ks1), tl.PropagateNan.NONE), tl.PropagateNan.NONE))) + (tl.math.min((-1) + ks1, tl.math.max(0, (-1) + (r1 % 3) + (x0 % ks1), tl.PropagateNan.NONE), tl.PropagateNan.NONE))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + ((ks1*(tl.math.min((-1) + ks0, tl.math.max(0, (-1) + (r1 // 3) + (x0 // ks1), tl.PropagateNan.NONE), tl.PropagateNan.NONE))) + (tl.math.min((-1) + ks1, tl.math.max(0, (-1) + (r1 % 3) + (x0 % ks1), tl.PropagateNan.NONE), tl.PropagateNan.NONE))), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp2 = tmp0 - tmp1
    tmp3 = tmp0 + tmp1
    tmp4 = 1e-10
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 / tmp5
    tmp9 = tmp8.to(tl.float32)
    tmp10 = 256.0
    tmp11 = tmp9 / tmp10
    tmp12 = tmp6 < tmp11
    tmp13 = 0.0
    tmp14 = 1.0
    tmp15 = tl.where(tmp12, tmp13, tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp19, xmask)
''', device_str='cuda')


# Source Nodes: [mul_4], Original ATen: [aten.mul]
# mul_4 => mul_8
triton_poi_fused_mul_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i64', 4: '*fp32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_7', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'cc5e07134fe44736250e1e295655d5c8951a39baf1ba36eb13cc712d6e9b780d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % ks0
    x1 = (xindex // ks0)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((ks2*(tl.math.min((-1) + ks1, tl.math.max(0, (-1) + (x0 // ks2) + (x1 // 3), tl.PropagateNan.NONE), tl.PropagateNan.NONE))) + (tl.math.min((-1) + ks2, tl.math.max(0, (-1) + (x0 % ks2) + (x1 % 3), tl.PropagateNan.NONE), tl.PropagateNan.NONE))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((ks2*(tl.math.min((-1) + ks1, tl.math.max(0, (-1) + (x0 // ks2) + (x1 // 3), tl.PropagateNan.NONE), tl.PropagateNan.NONE))) + (tl.math.min((-1) + ks2, tl.math.max(0, (-1) + (x0 % ks2) + (x1 % 3), tl.PropagateNan.NONE), tl.PropagateNan.NONE))), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + ((ks2*(tl.math.min((-1) + ks1, tl.math.max(0, (-1) + (x0 // ks2) + (x1 // 3), tl.PropagateNan.NONE), tl.PropagateNan.NONE))) + (tl.math.min((-1) + ks2, tl.math.max(0, (-1) + (x0 % ks2) + (x1 % 3), tl.PropagateNan.NONE), tl.PropagateNan.NONE))), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + (0))
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp3 = tmp1 - tmp2
    tmp4 = tmp1 + tmp2
    tmp5 = 1e-10
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 / tmp6
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 256.0
    tmp12 = tmp10 / tmp11
    tmp13 = tmp7 < tmp12
    tmp14 = 0.0
    tmp15 = 1.0
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 * tmp16
    tl.store(out_ptr0 + (x2), tmp17, xmask)
''', device_str='cuda')


# Source Nodes: [e_sum], Original ATen: [aten.sum]
# e_sum => sum_2
triton_per_fused_sum_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[1048576, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_sum_8', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'cc5e07134fe44736250e1e295655d5c8951a39baf1ba36eb13cc712d6e9b780d'}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (ks0*ks1*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# Source Nodes: [e_mean, e_mean_1, isnan, mul_5, sub_2], Original ATen: [aten.div, aten.isnan, aten.mul, aten.scalar_tensor, aten.sub, aten.where]
# e_mean => div_4
# e_mean_1 => full_default_3, where_2
# isnan => isnan
# mul_5 => mul_9
# sub_2 => sub_28
triton_poi_fused_div_isnan_mul_scalar_tensor_sub_where_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*i64', 6: '*fp32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_isnan_mul_scalar_tensor_sub_where_9', 'mutated_arg_names': [], 'no_x_dim': False, 'backend_hash': 'cc5e07134fe44736250e1e295655d5c8951a39baf1ba36eb13cc712d6e9b780d'},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % ks0
    x1 = (xindex // ks0)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((ks2*(tl.math.min((-1) + ks1, tl.math.max(0, (-1) + (x0 // ks2) + (x1 // 3), tl.PropagateNan.NONE), tl.PropagateNan.NONE))) + (tl.math.min((-1) + ks2, tl.math.max(0, (-1) + (x0 % ks2) + (x1 % 3), tl.PropagateNan.NONE), tl.PropagateNan.NONE))), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr3 + ((ks2*(tl.math.min((-1) + ks1, tl.math.max(0, (-1) + (x0 // ks2) + (x1 // 3), tl.PropagateNan.NONE), tl.PropagateNan.NONE))) + (tl.math.min((-1) + ks2, tl.math.max(0, (-1) + (x0 % ks2) + (x1 % 3), tl.PropagateNan.NONE), tl.PropagateNan.NONE))), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr4 + ((ks2*(tl.math.min((-1) + ks1, tl.math.max(0, (-1) + (x0 // ks2) + (x1 // 3), tl.PropagateNan.NONE), tl.PropagateNan.NONE))) + (tl.math.min((-1) + ks2, tl.math.max(0, (-1) + (x0 % ks2) + (x1 % 3), tl.PropagateNan.NONE), tl.PropagateNan.NONE))), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (0))
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK])
    tmp3 = tmp1 / tmp2
    tmp4 = tl.math.isnan(tmp3).to(tl.int1)
    tmp5 = 0.0
    tmp6 = tl.where(tmp4, tmp5, tmp3)
    tmp7 = tmp0 - tmp6
    tmp10 = tmp8 - tmp9
    tmp11 = tmp8 + tmp9
    tmp12 = 1e-10
    tmp13 = tmp11 + tmp12
    tmp14 = tmp10 / tmp13
    tmp17 = tmp16.to(tl.float32)
    tmp18 = 256.0
    tmp19 = tmp17 / tmp18
    tmp20 = tmp14 < tmp19
    tmp21 = 1.0
    tmp22 = tl.where(tmp20, tmp5, tmp21)
    tmp23 = tmp7 * tmp22
    tl.store(out_ptr0 + (x2), tmp23, xmask)
''', device_str='cuda')


# Source Nodes: [e_std, result, setitem, square], Original ATen: [aten.index_put, aten.lift_fresh, aten.mean, aten.pow, aten.zeros_like]
# e_std => mean
# result => full_2
# setitem => full_default_5, index_put
# square => pow_2
triton_per_fused_index_put_lift_fresh_mean_pow_zeros_like_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers
from triton.compiler.compiler import AttrsDescriptor

@persistent_reduction(
    size_hints=[1048576, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_index_put_lift_fresh_mean_pow_zeros_like_10', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'backend_hash': 'cc5e07134fe44736250e1e295655d5c8951a39baf1ba36eb13cc712d6e9b780d'}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ks0, ks1, ks2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + (ks0*ks1*r1)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_out_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tmp0 * tmp0
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.where(rmask & xmask, tmp2, 0)
    tmp5 = tl.sum(tmp4, 1)[:, None]
    tmp7 = ks2
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 >= tmp8
    tmp12 = tmp11 / tmp6
    tmp13 = tl.math.isnan(tmp12).to(tl.int1)
    tmp14 = 0.0
    tmp15 = tl.where(tmp13, tmp14, tmp12)
    tmp16 = 9.0
    tmp17 = tmp5 / tmp16
    tmp18 = tl.sqrt(tmp17)
    tmp19 = tl.math.isnan(tmp18).to(tl.int1)
    tmp20 = tl.where(tmp19, tmp14, tmp18)
    tmp21 = 2.0
    tmp22 = tmp20 / tmp21
    tmp23 = tmp15 + tmp22
    tmp24 = tmp10 <= tmp23
    tmp25 = tmp9 & tmp24
    tmp26 = 1.0
    tmp27 = tl.where(tmp25, tmp26, tmp14)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp27, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s3 = arg4_1
    assert_size_stride(arg2_1, (1, s0, s1), (s0*s1, s1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf4 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [bin_edges], Original ATen: [aten.linspace]
        stream0 = get_raw_stream(0)
        triton_poi_fused_linspace_0.run(buf4, 256, grid=grid(256), stream=stream0)
        buf5 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [add, add_1, bin_edges, contrast, counts, idx, sub, weights, zeros], Original ATen: [aten.add, aten.bucketize, aten.div, aten.linspace, aten.ones_like, aten.scatter_reduce, aten.sub, aten.zeros]
        triton_poi_fused_add_bucketize_div_linspace_ones_like_scatter_reduce_sub_zeros_1.run(buf5, 256, grid=grid(256), stream=stream0)
        buf0 = empty_strided_cuda((s0*s1, ), (1, ), torch.float32)
        buf2 = empty_strided_cuda((s0*s1, ), (1, ), torch.float32)
        # Source Nodes: [add, add_1, bin_edges, contrast, counts, idx, max_1, min_1, sub, weights, zeros], Original ATen: [aten.add, aten.bucketize, aten.div, aten.linspace, aten.max, aten.min, aten.ones_like, aten.scatter_reduce, aten.sub, aten.zeros]
        triton_per_fused_add_bucketize_div_linspace_max_min_ones_like_scatter_reduce_sub_zeros_2_xnumel = s0*s1
        triton_per_fused_add_bucketize_div_linspace_max_min_ones_like_scatter_reduce_sub_zeros_2.run(arg2_1, buf4, buf0, buf2, buf5, s0, s1, triton_per_fused_add_bucketize_div_linspace_max_min_ones_like_scatter_reduce_sub_zeros_2_xnumel, 9, grid=grid(triton_per_fused_add_bucketize_div_linspace_max_min_ones_like_scatter_reduce_sub_zeros_2_xnumel), stream=stream0)
        buf7 = buf4; del buf4  # reuse
        buf9 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [cumsum_2, mul, weight1], Original ATen: [aten.cumsum, aten.mul]
        triton_per_fused_cumsum_mul_3.run(buf5, buf7, buf9, 1, 256, grid=grid(1), stream=stream0)
        buf8 = empty_strided_cuda((256, ), (1, ), torch.float32)
        buf10 = empty_strided_cuda((256, ), (1, ), torch.float32)
        # Source Nodes: [cumsum_1, cumsum_3, flip, flip_2, mul_1], Original ATen: [aten.cumsum, aten.flip, aten.mul]
        triton_per_fused_cumsum_flip_mul_4.run(buf5, buf8, buf10, 1, 256, grid=grid(1), stream=stream0)
        del buf5
        buf11 = empty_strided_cuda((), (), torch.int64)
        # Source Nodes: [idx_1, mul_2, pow_1, sub_1, variance12], Original ATen: [aten.argmax, aten.mul, aten.pow, aten.sub]
        triton_red_fused_argmax_mul_pow_sub_5.run(buf7, buf8, buf9, buf10, buf11, 1, 255, grid=grid(1), stream=stream0)
        del buf10
        del buf7
        del buf8
        del buf9
        buf12 = empty_strided_cuda((s0*s1, ), (1, ), torch.float32)
        # Source Nodes: [hi_contrast_count], Original ATen: [aten.sum]
        triton_per_fused_sum_6_xnumel = s0*s1
        triton_per_fused_sum_6.run(buf0, buf2, buf11, buf12, s0, s1, triton_per_fused_sum_6_xnumel, 9, grid=grid(triton_per_fused_sum_6_xnumel), stream=stream0)
        ps0 = s0*s1
        buf13 = empty_strided_cuda((9, s0*s1), (s0*s1, 1), torch.float32)
        # Source Nodes: [mul_4], Original ATen: [aten.mul]
        triton_poi_fused_mul_7_xnumel = 9*s0*s1
        triton_poi_fused_mul_7.run(arg2_1, buf0, buf2, buf11, buf13, ps0, s0, s1, triton_poi_fused_mul_7_xnumel, grid=grid(triton_poi_fused_mul_7_xnumel), stream=stream0)
        buf14 = empty_strided_cuda((s0*s1, ), (1, ), torch.float32)
        # Source Nodes: [e_sum], Original ATen: [aten.sum]
        triton_per_fused_sum_8_xnumel = s0*s1
        triton_per_fused_sum_8.run(buf13, buf14, s0, s1, triton_per_fused_sum_8_xnumel, 9, grid=grid(triton_per_fused_sum_8_xnumel), stream=stream0)
        buf15 = buf13; del buf13  # reuse
        # Source Nodes: [e_mean, e_mean_1, isnan, mul_5, sub_2], Original ATen: [aten.div, aten.isnan, aten.mul, aten.scalar_tensor, aten.sub, aten.where]
        triton_poi_fused_div_isnan_mul_scalar_tensor_sub_where_9_xnumel = 9*s0*s1
        triton_poi_fused_div_isnan_mul_scalar_tensor_sub_where_9.run(arg2_1, buf14, buf12, buf0, buf2, buf11, buf15, ps0, s0, s1, triton_poi_fused_div_isnan_mul_scalar_tensor_sub_where_9_xnumel, grid=grid(triton_poi_fused_div_isnan_mul_scalar_tensor_sub_where_9_xnumel), stream=stream0)
        del buf0
        del buf11
        del buf2
        buf17 = reinterpret_tensor(buf12, (1, s0, s1), (s0*s1, s1, 1), 0); del buf12  # reuse
        # Source Nodes: [e_std, result, setitem, square], Original ATen: [aten.index_put, aten.lift_fresh, aten.mean, aten.pow, aten.zeros_like]
        triton_per_fused_index_put_lift_fresh_mean_pow_zeros_like_10_xnumel = s0*s1
        triton_per_fused_index_put_lift_fresh_mean_pow_zeros_like_10.run(buf17, buf15, arg2_1, buf14, s0, s1, s3, triton_per_fused_index_put_lift_fresh_mean_pow_zeros_like_10_xnumel, 9, grid=grid(triton_per_fused_index_put_lift_fresh_mean_pow_zeros_like_10_xnumel), stream=stream0)
        del arg2_1
        del buf14
        del buf15
    return (buf17, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 581
    arg1_1 = 1091
    arg2_1 = rand_strided((1, 581, 1091), (633871, 1091, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = 3
    arg4_1 = 3
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
