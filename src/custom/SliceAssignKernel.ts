/**
 * GPU kernel for in-place slice assignment
 * TODO(PAPR): currently supports tensors up to 5D 
 */

import { KernelSpec } from '../kernel';

export const sliceAssignKernel: KernelSpec = {
  name: "slice_assign",
  config: [],
  parameters: [
    { name: "ndim", shaderType: "u32" },

    { name: "destD0", shaderType: "u32" },
    { name: "destD1", shaderType: "u32" },
    { name: "destD2", shaderType: "u32" },
    { name: "destD3", shaderType: "u32" },
    { name: "destD4", shaderType: "u32" },

    { name: "srcD0", shaderType: "u32" },
    { name: "srcD1", shaderType: "u32" },
    { name: "srcD2", shaderType: "u32" },
    { name: "srcD3", shaderType: "u32" },
    { name: "srcD4", shaderType: "u32" },

    { name: "start0", shaderType: "u32" },
    { name: "start1", shaderType: "u32" },
    { name: "start2", shaderType: "u32" },
    { name: "start3", shaderType: "u32" },
    { name: "start4", shaderType: "u32" },

    // Pre-computed strides for faster indexing
    { name: "destStride0", shaderType: "u32" },
    { name: "destStride1", shaderType: "u32" },
    { name: "destStride2", shaderType: "u32" },
    { name: "destStride3", shaderType: "u32" },
    { name: "srcStride0", shaderType: "u32" },
    { name: "srcStride1", shaderType: "u32" },
    { name: "srcStride2", shaderType: "u32" },
    { name: "srcStride3", shaderType: "u32" },
  ],
  inputs: [
    { name: "src", shaderType: "array<f32>" }
  ],
  outputs: [
    { name: "tensor", shaderType: "array<f32>", size: "destD0 * destD1 * destD2 * destD3 * destD4" }
  ],
  workgroupSize: [256, 1, 1],
  workgroupCount: [
    // OPTIMIZATION: Each thread processes 4 elements, so divide by 4
    "((srcD0 * srcD1 * srcD2 * srcD3 * srcD4 + 3) / 4 + 255) / 256",
    1,
    1
  ],
  workgroupVariables: [],
  shader: `
    // OPTIMIZATION: Process 4 elements per thread for better bandwidth utilization
    let thread_idx = global_id.x;
    let total_elements = parameters.srcD0 * parameters.srcD1 * parameters.srcD2 *
                         parameters.srcD3 * parameters.srcD4;

    let base_idx = thread_idx * 4u;

    // OPTIMIZATION: Loop unrolling - process 4 elements per thread
    for (var i = 0u; i < 4u; i = i + 1u) {
      let linear_idx = base_idx + i;

      if (linear_idx >= total_elements) {
        return;
      }

      // Decompose linear index to 5D source coordinates
      var temp = linear_idx;
      let s4 = temp % parameters.srcD4;
      temp = temp / parameters.srcD4;
      let s3 = temp % parameters.srcD3;
      temp = temp / parameters.srcD3;
      let s2 = temp % parameters.srcD2;
      temp = temp / parameters.srcD2;
      let s1 = temp % parameters.srcD1;
      let s0 = temp / parameters.srcD1;

      // Compute destination coordinates by adding offset
      let d0 = s0 + parameters.start0;
      let d1 = s1 + parameters.start1;
      let d2 = s2 + parameters.start2;
      let d3 = s3 + parameters.start3;
      let d4 = s4 + parameters.start4;

      // Direct index computation using strides
      let dst_idx = d0 * parameters.destStride0 +
                    d1 * parameters.destStride1 +
                    d2 * parameters.destStride2 +
                    d3 * parameters.destStride3 +
                    d4;

      // Direct copy with memory coalescing
      tensor[dst_idx] = src[linear_idx];
    }
  `
};
