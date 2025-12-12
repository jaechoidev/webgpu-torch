/**
 * GPU-based slice implementation for WebGPU
 * PyTorch reference: torch/csrc/autograd/FunctionsManual.cpp (SliceBackward)
 * PyTorch uses storage_offset for zero-copy views, but we copy data on GPU instead.
 * TODO(PAPR): currently supports tensors up to 5D 
 */

import type { KernelSpec } from '../kernel';

/**
 * Generic slice kernel for 1D-5D tensors
 */
export const sliceKernel: KernelSpec = {
  name: "slice",
  config: [],
  parameters: [
    { name: "ndim", shaderType: "u32" },
    { name: "inputD0", shaderType: "u32" },
    { name: "inputD1", shaderType: "u32" },
    { name: "inputD2", shaderType: "u32" },
    { name: "inputD3", shaderType: "u32" },
    { name: "inputD4", shaderType: "u32" },
    { name: "outputD0", shaderType: "u32" },
    { name: "outputD1", shaderType: "u32" },
    { name: "outputD2", shaderType: "u32" },
    { name: "outputD3", shaderType: "u32" },
    { name: "outputD4", shaderType: "u32" },
    { name: "offset0", shaderType: "u32" },
    { name: "offset1", shaderType: "u32" },
    { name: "offset2", shaderType: "u32" },
    { name: "offset3", shaderType: "u32" },
    { name: "offset4", shaderType: "u32" },
    // Pre-computed strides for faster indexing
    { name: "inputStride0", shaderType: "u32" },
    { name: "inputStride1", shaderType: "u32" },
    { name: "inputStride2", shaderType: "u32" },
    { name: "inputStride3", shaderType: "u32" },
    { name: "outputStride0", shaderType: "u32" },
    { name: "outputStride1", shaderType: "u32" },
    { name: "outputStride2", shaderType: "u32" },
    { name: "outputStride3", shaderType: "u32" },
  ],
  inputs: [
    {
      name: "input",
      shaderType: "array<f32>",
    },
  ],
  outputs: [
    {
      name: "output",
      shaderType: "array<f32>",
      size: "outputD0 * outputD1 * outputD2 * outputD3 * outputD4",
    },
  ],
  workgroupSize: [256, 1, 1],
  workgroupCount: ["((outputD0 * outputD1 * outputD2 * outputD3 * outputD4 + 3) / 4 + 255) / 256", 1, 1],
  workgroupVariables: [],
  shader: `
    // OPTIMIZATION: Process 4 elements per thread
    let thread_idx = global_id.x;
    let total_elements = parameters.outputD0 * parameters.outputD1 * parameters.outputD2 *
                         parameters.outputD3 * parameters.outputD4;

    let base_idx = thread_idx * 4u;

    // OPTIMIZATION: Unrolled loop for 4 elements
    for (var i = 0u; i < 4u; i = i + 1u) {
      let output_idx = base_idx + i;

      if (output_idx >= total_elements) {
        return;
      }

      // Fast index decomposition
      var temp = output_idx;
      let o4 = temp % parameters.outputD4;
      temp = temp / parameters.outputD4;
      let o3 = temp % parameters.outputD3;
      temp = temp / parameters.outputD3;
      let o2 = temp % parameters.outputD2;
      temp = temp / parameters.outputD2;
      let o1 = temp % parameters.outputD1;
      let o0 = temp / parameters.outputD1;

      // Direct input index calculation
      let input_idx = (o0 + parameters.offset0) * parameters.inputStride0 +
                      (o1 + parameters.offset1) * parameters.inputStride1 +
                      (o2 + parameters.offset2) * parameters.inputStride2 +
                      (o3 + parameters.offset3) * parameters.inputStride3 +
                      (o4 + parameters.offset4);

      // Memory coalescing
      output[output_idx] = input[input_idx];
    }
  `,
};
