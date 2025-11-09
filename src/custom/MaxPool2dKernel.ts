/**
 * MaxPool2d GPU Kernel for WebGPU
 */

import type { KernelSpec } from 'webgpu-torch';

/**
 * MaxPool2d kernel: [N, C, H, W] -> [N, C, H/2, W/2]
 */
export const maxpool2dKernel: KernelSpec = {
  name: "maxpool2d",
  config: [],
  parameters: [
    { name: "N", shaderType: "u32" },
    { name: "C", shaderType: "u32" },
    { name: "H_in", shaderType: "u32" },
    { name: "W_in", shaderType: "u32" },
    { name: "H_out", shaderType: "u32" },
    { name: "W_out", shaderType: "u32" },
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
      size: "N * C * H_out * W_out",
    },
  ],
  workgroupSize: [16, 16, 1],
  workgroupCount: [
    "(W_out + 15) / 16",
    "(H_out + 15) / 16",
    "N * C"
  ],
  workgroupVariables: [],
  shader: `
    let w_out = global_id.x;
    let h_out = global_id.y;
    let nc = global_id.z;

    if (w_out >= parameters.W_out || h_out >= parameters.H_out || nc >= parameters.N * parameters.C) {
      return;
    }

    let n = nc / parameters.C;
    let c = nc % parameters.C;

    let h_in = h_out * 2u;
    let w_in = w_out * 2u;

    // Input layout: [N, C, H, W]
    let base_idx = ((n * parameters.C + c) * parameters.H_in * parameters.W_in) + (h_in * parameters.W_in + w_in);

    let val00 = input[base_idx];                                      // [h, w]
    let val01 = input[base_idx + 1u];                                 // [h, w+1]
    let val10 = input[base_idx + parameters.W_in];                    // [h+1, w]
    let val11 = input[base_idx + parameters.W_in + 1u];               // [h+1, w+1]

    var max_val = val00;
    max_val = max(max_val, val01);
    max_val = max(max_val, val10);
    max_val = max(max_val, val11);

    let out_idx = ((n * parameters.C + c) * parameters.H_out + h_out) * parameters.W_out + w_out;
    output[out_idx] = max_val;
  `
};
