/**
 * GPU kernel for tensor permutation (dimension reordering)
 *
 * PyTorch reference: torch/csrc/autograd/FunctionsManual.cpp (PermuteBackward)
 * PyTorch uses storage_offset for zero-copy views, but we copy data on GPU.
 * TODO(PAPR): currently supports tensors up to 5D
 */

import { KernelSpec } from '../kernel';

/**
 * Permute kernel for 1D-5D tensors
 */
export const permuteKernel: KernelSpec = {
  name: "permute",
  config: [],
  parameters: [
    { name: "ndim", shaderType: "u32" },

    { name: "perm0", shaderType: "u32" },
    { name: "perm1", shaderType: "u32" },
    { name: "perm2", shaderType: "u32" },
    { name: "perm3", shaderType: "u32" },
    { name: "perm4", shaderType: "u32" },

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
    let output_idx = global_id.x + global_id.y * 65535u * 256u;
    let total_elements = parameters.outputD0 * parameters.outputD1 * parameters.outputD2 *
                         parameters.outputD3 * parameters.outputD4;

    if (output_idx >= total_elements) {
      return;
    }

    var temp_idx = output_idx;
    let out_d4 = temp_idx % parameters.outputD4;
    temp_idx = temp_idx / parameters.outputD4;
    let out_d3 = temp_idx % parameters.outputD3;
    temp_idx = temp_idx / parameters.outputD3;
    let out_d2 = temp_idx % parameters.outputD2;
    temp_idx = temp_idx / parameters.outputD2;
    let out_d1 = temp_idx % parameters.outputD1;
    let out_d0 = temp_idx / parameters.outputD1;

    var out_coords: array<u32, 5>;
    out_coords[0] = out_d0;
    out_coords[1] = out_d1;
    out_coords[2] = out_d2;
    out_coords[3] = out_d3;
    out_coords[4] = out_d4;

    var in_coords: array<u32, 5>;
    in_coords[parameters.perm0] = out_coords[0];
    in_coords[parameters.perm1] = out_coords[1];
    in_coords[parameters.perm2] = out_coords[2];
    in_coords[parameters.perm3] = out_coords[3];
    in_coords[parameters.perm4] = out_coords[4];

    let input_idx = in_coords[0] * parameters.inputD1 * parameters.inputD2 * parameters.inputD3 * parameters.inputD4 +
                    in_coords[1] * parameters.inputD2 * parameters.inputD3 * parameters.inputD4 +
                    in_coords[2] * parameters.inputD3 * parameters.inputD4 +
                    in_coords[3] * parameters.inputD4 +
                    in_coords[4];

    output[output_idx] = input[input_idx];
  `
};
