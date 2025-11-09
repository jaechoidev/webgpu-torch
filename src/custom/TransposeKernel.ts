/**
 * GPU kernel for transposing last two dimensions of N-D tensors
 * Supports 2D-5D tensors by padding dimensions to 5D for uniform processing.
 */

import { KernelSpec } from '../kernel';

/**
 * GPU kernel for transposing last two dimensions
 */
export const transposeKernel: KernelSpec = {
  name: "transpose_last2",
  config: [],
  parameters: [
    // Number of actual dimensions (2-5)
    { name: "ndim", shaderType: "u32" },

    // Input shape (padded to 5D)
    { name: "inputD0", shaderType: "u32" },
    { name: "inputD1", shaderType: "u32" },
    { name: "inputD2", shaderType: "u32" },
    { name: "inputD3", shaderType: "u32" },
    { name: "inputD4", shaderType: "u32" },

    // Output shape (last two dims swapped, padded to 5D)
    { name: "outputD0", shaderType: "u32" },
    { name: "outputD1", shaderType: "u32" },
    { name: "outputD2", shaderType: "u32" },
    { name: "outputD3", shaderType: "u32" },
    { name: "outputD4", shaderType: "u32" },
  ],
  inputs: [
    { name: "input", shaderType: "array<f32>" }
  ],
  outputs: [
    {
      name: "output",
      shaderType: "array<f32>",
      size: "outputD0 * outputD1 * outputD2 * outputD3 * outputD4"
    }
  ],
  workgroupSize: [256, 1, 1],
  workgroupCount: [
    "65535",
    "((outputD0 * outputD1 * outputD2 * outputD3 * outputD4 + 255) / 256 + 65534) / 65535",
    1
  ],
  workgroupVariables: [],
  shader: `
    let outputSize = parameters.outputD0 * parameters.outputD1 * parameters.outputD2 * parameters.outputD3 * parameters.outputD4;
    let workgroup_1d = (global_id.x / 256u) + global_id.y * 65535u;
    let i = workgroup_1d * 256u + (global_id.x % 256u);

    if (i >= outputSize) {
      return;
    }

    var idx = i;
    let d4 = idx % parameters.outputD4;
    idx = idx / parameters.outputD4;
    let d3 = idx % parameters.outputD3;
    idx = idx / parameters.outputD3;
    let d2 = idx % parameters.outputD2;
    idx = idx / parameters.outputD2;
    let d1 = idx % parameters.outputD1;
    let d0 = idx / parameters.outputD1;

    let in_d4 = d3;  
    let in_d3 = d4;  

    let src_idx = ((d0 * parameters.inputD1 + d1) * parameters.inputD2 + d2) * parameters.inputD3 * parameters.inputD4
                  + in_d3 * parameters.inputD4
                  + in_d4;

    output[i] = input[src_idx];
  `
};
