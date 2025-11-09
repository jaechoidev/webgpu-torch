/**
 * GPU-based slice implementation for WebGPU
 * PyTorch reference: torch/csrc/autograd/FunctionsManual.cpp (SliceBackward)
 * PyTorch uses storage_offset for zero-copy views, but we copy data on GPU instead.
 * TODO(PAPR): currently supports tensors up to 5D 
 */

import type { KernelSpec } from 'webgpu-torch';

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
  workgroupCount: ["65535", "((outputD0 * outputD1 * outputD2 * outputD3 * outputD4 + 255) / 256 + 65534) / 65535", 1],
  workgroupVariables: [],
  shader: `
    let output_idx = global_id.x + global_id.y * 65535u * 256u;
    let total_elements = parameters.outputD0 * parameters.outputD1 * parameters.outputD2 *
                         parameters.outputD3 * parameters.outputD4;

    if (output_idx >= total_elements) {
        return;
    }

    var temp_idx = output_idx;
    let output_d4 = temp_idx % parameters.outputD4;
    temp_idx = temp_idx / parameters.outputD4;
    let output_d3 = temp_idx % parameters.outputD3;
    temp_idx = temp_idx / parameters.outputD3;
    let output_d2 = temp_idx % parameters.outputD2;
    temp_idx = temp_idx / parameters.outputD2;
    let output_d1 = temp_idx % parameters.outputD1;
    let output_d0 = temp_idx / parameters.outputD1;

    let input_d0 = output_d0 + parameters.offset0;
    let input_d1 = output_d1 + parameters.offset1;
    let input_d2 = output_d2 + parameters.offset2;
    let input_d3 = output_d3 + parameters.offset3;
    let input_d4 = output_d4 + parameters.offset4;

    let input_idx = input_d4 +
                    input_d3 * parameters.inputD4 +
                    input_d2 * parameters.inputD3 * parameters.inputD4 +
                    input_d1 * parameters.inputD2 * parameters.inputD3 * parameters.inputD4 +
                    input_d0 * parameters.inputD1 * parameters.inputD2 * parameters.inputD3 * parameters.inputD4;

    output[output_idx] = input[input_idx];
  `,
};
