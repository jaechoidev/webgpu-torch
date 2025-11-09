/**
 * GPU Kernel for clamping tensor values
 * Clamps all elements in a tensor to [min, max] range
 */

import type { KernelSpec } from '../kernel';

export const clampKernel: KernelSpec = {
  name: "clamp",
  config: [],
  parameters: [
    { name: "size", shaderType: "u32" },
    { name: "min_val", shaderType: "f32" },
    { name: "max_val", shaderType: "f32" },
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
      size: "size",
    },
  ],
  workgroupSize: [256, 1, 1],
  workgroupCount: [
    "65535",
    "((size + 255) / 256 + 65534) / 65535",
    1
  ],
  workgroupVariables: [],
  shader: `
    // 2D dispatch to avoid workgroup count limit (max 65535 per dimension)
    // X dimension: up to 65535 workgroups
    // Y dimension: remaining workgroups distributed
    // Total capacity: 65535 * 65535 * 256 = 1.1 trillion elements
    let idx = global_id.x + global_id.y * 65535u * 256u;

    if (idx >= parameters.size) {
      return;
    }

    var val = input[idx];
    val = max(val, parameters.min_val);
    val = min(val, parameters.max_val);
    output[idx] = val;
  `
};
