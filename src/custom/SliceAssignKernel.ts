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
  ],
  inputs: [
    { name: "src", shaderType: "array<f32>" }
  ],
  outputs: [
    { name: "tensor", shaderType: "array<f32>", size: "destD0 * destD1 * destD2 * destD3 * destD4" } 
  ],
  workgroupSize: [16, 16, 1],
  workgroupCount: [
    "(destD2 + 15) / 16", 
    "(destD1 + 15) / 16", 
    "destD0 * destD3 * destD4" 
  ],
  workgroupVariables: [],
  shader: `
    let d2 = global_id.x;  // Width
    let d1 = global_id.y;  // Height

    var remaining = global_id.z;
    let d0 = remaining / (parameters.destD3 * parameters.destD4);
    remaining = remaining % (parameters.destD3 * parameters.destD4);
    let d3 = remaining / parameters.destD4;
    let d4 = remaining % parameters.destD4;

    if (d0 >= parameters.destD0 || d1 >= parameters.destD1 || d2 >= parameters.destD2 ||
        d3 >= parameters.destD3 || d4 >= parameters.destD4) {
      return;
    }

    let tensor_idx = d0 * parameters.destD1 * parameters.destD2 * parameters.destD3 * parameters.destD4 +
                    d1 * parameters.destD2 * parameters.destD3 * parameters.destD4 +
                    d2 * parameters.destD3 * parameters.destD4 +
                    d3 * parameters.destD4 +
                    d4;

    var in_slice = true;
    if (d0 < parameters.start0 || d0 >= parameters.start0 + parameters.srcD0) {
      in_slice = false;
    }
    if (d1 < parameters.start1 || d1 >= parameters.start1 + parameters.srcD1) {
      in_slice = false;
    }
    if (d2 < parameters.start2 || d2 >= parameters.start2 + parameters.srcD2) {
      in_slice = false;
    }
    if (d3 < parameters.start3 || d3 >= parameters.start3 + parameters.srcD3) {
      in_slice = false;
    }
    if (d4 < parameters.start4 || d4 >= parameters.start4 + parameters.srcD4) {
      in_slice = false;
    }

    if (in_slice) {
      let s0 = d0 - parameters.start0;
      let s1 = d1 - parameters.start1;
      let s2 = d2 - parameters.start2;
      let s3 = d3 - parameters.start3;
      let s4 = d4 - parameters.start4;

      let src_idx = s0 * parameters.srcD1 * parameters.srcD2 * parameters.srcD3 * parameters.srcD4 +
                    s1 * parameters.srcD2 * parameters.srcD3 * parameters.srcD4 +
                    s2 * parameters.srcD3 * parameters.srcD4 +
                    s3 * parameters.srcD4 +
                    s4;

      tensor[tensor_idx] = src[src_idx];
    } else {
      // No change to tensor outside the slice region
    }
  `
};
