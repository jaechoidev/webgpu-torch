/**
 * GPU kernel for softmax normalization along any dimension
 *   softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
 */

import { KernelNode } from '../graph';
import { KernelSpec } from '../kernel';

/**
 * Fused softmax kernel for any dimension
 */

export const softmaxKernel: KernelSpec = {
  name: "softmax_dim",
  config: [
    { name: "numThreads" }
  ],
  parameters: [
    { name: "ndim", shaderType: "u32" },
    { name: "reduceDim", shaderType: "u32" },
    { name: "D0", shaderType: "u32" },
    { name: "D1", shaderType: "u32" },
    { name: "D2", shaderType: "u32" },
    { name: "D3", shaderType: "u32" },
    { name: "D4", shaderType: "u32" },
  ],
  inputs: [
    { name: "input", shaderType: "array<f32>" }
  ],
  outputs: [
    {
      name: "output",
      shaderType: "array<f32>",
      size: "D0 * D1 * D2 * D3 * D4"
    }
  ],
  workgroupSize: [256, 1, 1],
  workgroupCount: [
    "(numThreads + 255) / 256",
    1,
    1
  ],
  workgroupVariables: [],
  shader: `
    let stride4 = 1u;
    let stride3 = parameters.D4;
    let stride2 = parameters.D3 * parameters.D4;
    let stride1 = parameters.D2 * parameters.D3 * parameters.D4;
    let stride0 = parameters.D1 * parameters.D2 * parameters.D3 * parameters.D4;

    var reduceSize: u32;
    if (parameters.reduceDim == 0u) {
      reduceSize = parameters.D0;
    } else if (parameters.reduceDim == 1u) {
      reduceSize = parameters.D1;
    } else if (parameters.reduceDim == 2u) {
      reduceSize = parameters.D2;
    } else if (parameters.reduceDim == 3u) {
      reduceSize = parameters.D3;
    } else {
      reduceSize = parameters.D4;
    }

    let totalSize = parameters.D0 * parameters.D1 * parameters.D2 * parameters.D3 * parameters.D4;
    let numSlices = totalSize / reduceSize;

    let sliceIdx = global_id.x;

    if (sliceIdx >= numSlices) {
      return;
    }

    var temp = sliceIdx;
    var d0 = 0u;
    var d1 = 0u;
    var d2 = 0u;
    var d3 = 0u;
    var d4 = 0u;

    if (parameters.reduceDim == 4u) {
      d4 = 0u;
      d3 = temp % parameters.D3; temp = temp / parameters.D3;
      d2 = temp % parameters.D2; temp = temp / parameters.D2;
      d1 = temp % parameters.D1; temp = temp / parameters.D1;
      d0 = temp;
    } else if (parameters.reduceDim == 3u) {
      d3 = 0u;
      d4 = temp % parameters.D4; temp = temp / parameters.D4;
      d2 = temp % parameters.D2; temp = temp / parameters.D2;
      d1 = temp % parameters.D1; temp = temp / parameters.D1;
      d0 = temp;
    } else if (parameters.reduceDim == 2u) {
      d2 = 0u;
      d4 = temp % parameters.D4; temp = temp / parameters.D4;
      d3 = temp % parameters.D3; temp = temp / parameters.D3;
      d1 = temp % parameters.D1; temp = temp / parameters.D1;
      d0 = temp;
    } else if (parameters.reduceDim == 1u) {
      d1 = 0u;
      d4 = temp % parameters.D4; temp = temp / parameters.D4;
      d3 = temp % parameters.D3; temp = temp / parameters.D3;
      d2 = temp % parameters.D2; temp = temp / parameters.D2;
      d0 = temp;
    } else {  // parameters.reduceDim == 0
      d0 = 0u;
      d4 = temp % parameters.D4; temp = temp / parameters.D4;
      d3 = temp % parameters.D3; temp = temp / parameters.D3;
      d2 = temp % parameters.D2; temp = temp / parameters.D2;
      d1 = temp;
    }

    // Find max value along reduction dimension
    var maxVal = -3.402823e+38;  // -FLT_MAX
    for (var i = 0u; i < reduceSize; i = i + 1u) {
      // Set the reduction dimension coordinate
      var idx_d0 = d0;
      var idx_d1 = d1;
      var idx_d2 = d2;
      var idx_d3 = d3;
      var idx_d4 = d4;

      if (parameters.reduceDim == 0u) { idx_d0 = i; }
      else if (parameters.reduceDim == 1u) { idx_d1 = i; }
      else if (parameters.reduceDim == 2u) { idx_d2 = i; }
      else if (parameters.reduceDim == 3u) { idx_d3 = i; }
      else { idx_d4 = i; }

      let flatIdx = idx_d0 * stride0 + idx_d1 * stride1 + idx_d2 * stride2 + idx_d3 * stride3 + idx_d4 * stride4;
      let val = input[flatIdx];
      if (val > maxVal) {
        maxVal = val;
      }
    }

    // Compute sum of exp(x - max)
    var sumExp = 0.0;
    for (var i = 0u; i < reduceSize; i = i + 1u) {
      var idx_d0 = d0;
      var idx_d1 = d1;
      var idx_d2 = d2;
      var idx_d3 = d3;
      var idx_d4 = d4;

      if (parameters.reduceDim == 0u) { idx_d0 = i; }
      else if (parameters.reduceDim == 1u) { idx_d1 = i; }
      else if (parameters.reduceDim == 2u) { idx_d2 = i; }
      else if (parameters.reduceDim == 3u) { idx_d3 = i; }
      else { idx_d4 = i; }

      let flatIdx = idx_d0 * stride0 + idx_d1 * stride1 + idx_d2 * stride2 + idx_d3 * stride3 + idx_d4 * stride4;
      sumExp = sumExp + exp(input[flatIdx] - maxVal);
    }

    // Compute and write output: exp(x - max) / sumExp
    for (var i = 0u; i < reduceSize; i = i + 1u) {
      var idx_d0 = d0;
      var idx_d1 = d1;
      var idx_d2 = d2;
      var idx_d3 = d3;
      var idx_d4 = d4;

      if (parameters.reduceDim == 0u) { idx_d0 = i; }
      else if (parameters.reduceDim == 1u) { idx_d1 = i; }
      else if (parameters.reduceDim == 2u) { idx_d2 = i; }
      else if (parameters.reduceDim == 3u) { idx_d3 = i; }
      else { idx_d4 = i; }

      let flatIdx = idx_d0 * stride0 + idx_d1 * stride1 + idx_d2 * stride2 + idx_d3 * stride3 + idx_d4 * stride4;
      output[flatIdx] = exp(input[flatIdx] - maxVal) / sumExp;
    }
  `
};

export const softmax_fast_kernel: KernelSpec = {
  name: "softmax_fast",
  config: [
    { name: "numThreads" }
  ],
  parameters: [
    { name: "packedCols", shaderType: "i32" },
    { name: "rows", shaderType: "i32" },
    { name: "components", shaderType: "i32" }
  ],
  inputs: [
    { name: "x", shaderType: "array<f32>" }
  ],
  outputs: [
    {
      name: "result", 
      shaderType: "array<f32>",
      size: "rows * packedCols * components"
    }
  ],
  workgroupSize: [64, 1, 1], // Default to 64, but you might want to make this configurable
  workgroupCount: [
    "rows", // One workgroup per row
    1,
    1
  ],
  workgroupVariables: [
    { name: "rowMaxShared", shaderType: "f32" },
    { name: "rowSumShared", shaderType: "f32" },
    { name: "threadShared", shaderType: ["array<f32>", 64] }
  ],
  shader: `
    var<workgroup> rowMaxShared: f32;
    var<workgroup> rowSumShared: f32;
    var<workgroup> threadShared: array<f32, 64>;
    
    let wg = 64u;
    let lindex = local_id.x;
    let row = workgroup_id.x;
    let cols = parameters.packedCols;
    let row_stride = parameters.packedCols;

    // ---------------------------------------------------------------------
    // PASS 1: find row max (parallel reduction)
    // ---------------------------------------------------------------------
    
    var threadMax: f32 = -3.402823e+38;
    var col = lindex;
    
    while (col < cols) {
      let index = row * row_stride + col;
      let value = x[index];
      threadMax = max(threadMax, value);
      col += wg;
    }
    
    if (lindex < cols) {
      threadShared[lindex] = threadMax;
    }
    workgroupBarrier();

    // Parallel reduction for max
    var reduceSize = min(cols, wg);
    var currSize = reduceSize >> 1;
    
    while (currSize > 0) {
      reduceSize = currSize + (reduceSize & 1);
      if (lindex < currSize) {
        threadShared[lindex] = max(threadShared[lindex], threadShared[lindex + reduceSize]);
      }
      workgroupBarrier();
      currSize = reduceSize >> 1;
    }
    
    if (lindex == 0) {
      rowMaxShared = threadShared[0];
    }
    workgroupBarrier();

    // ---------------------------------------------------------------------
    // PASS 2: find row sum of exp(x - max) (parallel reduction)
    // ---------------------------------------------------------------------
    
    var threadSum: f32 = 0.0;
    col = lindex;
    
    while (col < cols) {
      let index = row * row_stride + col;
      let subExp = exp(x[index] - rowMaxShared);
      threadSum += subExp;
      col += wg;
    }
    
    threadShared[lindex] = threadSum;
    workgroupBarrier();

    // Parallel reduction for sum
    currSize = wg >> 1;
    while (currSize > 0) {
      if (lindex < currSize) {
        threadShared[lindex] += threadShared[lindex + currSize];
      }
      workgroupBarrier();
      currSize = currSize >> 1;
    }
    
    if (lindex == 0) {
      rowSumShared = threadShared[0];
    }
    workgroupBarrier();

    // ---------------------------------------------------------------------
    // PASS 3: calculate final values
    // ---------------------------------------------------------------------
    
    col = lindex;
    while (col < cols) {
      let index = row * row_stride + col;
      var value = exp(x[index] - rowMaxShared) / rowSumShared;
      // max operation protects against NaN since all values should be >=0
      value = max(value, 0.0);
      result[index] = value;
      col += wg;
    }
  `
};


export const squaremaxKernel: KernelSpec = {
  name: "squaremax_dim",
  config: [
    { name: "numThreads" }
  ],
  parameters: [
    { name: "ndim", shaderType: "u32" },
    { name: "reduceDim", shaderType: "u32" },
    { name: "D0", shaderType: "u32" },
    { name: "D1", shaderType: "u32" },
    { name: "D2", shaderType: "u32" },
    { name: "D3", shaderType: "u32" },
    { name: "D4", shaderType: "u32" },
  ],
  inputs: [
    { name: "input", shaderType: "array<f32>" }
  ],
  outputs: [
    {
      name: "output",
      shaderType: "array<f32>",
      size: "D0 * D1 * D2 * D3 * D4"
    }
  ],
  workgroupSize: [256, 1, 1],
  workgroupCount: [
    "(numThreads + 255) / 256",
    1,
    1
  ],
  workgroupVariables: [],
  shader: `
    let stride4 = 1u;
    let stride3 = parameters.D4;
    let stride2 = parameters.D3 * parameters.D4;
    let stride1 = parameters.D2 * parameters.D3 * parameters.D4;
    let stride0 = parameters.D1 * parameters.D2 * parameters.D3 * parameters.D4;

    var reduceSize: u32;
    if (parameters.reduceDim == 0u) {
      reduceSize = parameters.D0;
    } else if (parameters.reduceDim == 1u) {
      reduceSize = parameters.D1;
    } else if (parameters.reduceDim == 2u) {
      reduceSize = parameters.D2;
    } else if (parameters.reduceDim == 3u) {
      reduceSize = parameters.D3;
    } else {
      reduceSize = parameters.D4;
    }

    let totalSize = parameters.D0 * parameters.D1 * parameters.D2 * parameters.D3 * parameters.D4;
    let numSlices = totalSize / reduceSize;

    let sliceIdx = global_id.x;

    if (sliceIdx >= numSlices) {
      return;
    }

    var temp = sliceIdx;
    var d0 = 0u;
    var d1 = 0u;
    var d2 = 0u;
    var d3 = 0u;
    var d4 = 0u;

    if (parameters.reduceDim == 4u) {
      d4 = 0u;
      d3 = temp % parameters.D3; temp = temp / parameters.D3;
      d2 = temp % parameters.D2; temp = temp / parameters.D2;
      d1 = temp % parameters.D1; temp = temp / parameters.D1;
      d0 = temp;
    } else if (parameters.reduceDim == 3u) {
      d3 = 0u;
      d4 = temp % parameters.D4; temp = temp / parameters.D4;
      d2 = temp % parameters.D2; temp = temp / parameters.D2;
      d1 = temp % parameters.D1; temp = temp / parameters.D1;
      d0 = temp;
    } else if (parameters.reduceDim == 2u) {
      d2 = 0u;
      d4 = temp % parameters.D4; temp = temp / parameters.D4;
      d3 = temp % parameters.D3; temp = temp / parameters.D3;
      d1 = temp % parameters.D1; temp = temp / parameters.D1;
      d0 = temp;
    } else if (parameters.reduceDim == 1u) {
      d1 = 0u;
      d4 = temp % parameters.D4; temp = temp / parameters.D4;
      d3 = temp % parameters.D3; temp = temp / parameters.D3;
      d2 = temp % parameters.D2; temp = temp / parameters.D2;
      d0 = temp;
    } else {  // parameters.reduceDim == 0
      d0 = 0u;
      d4 = temp % parameters.D4; temp = temp / parameters.D4;
      d3 = temp % parameters.D3; temp = temp / parameters.D3;
      d2 = temp % parameters.D2; temp = temp / parameters.D2;
      d1 = temp;
    }

    // 1. Find max absolute value for numerical stability
    // We scale by maxAbs to prevent overflow when squaring large numbers
    var maxAbs = 1e-12; // Epsilon to prevent division by zero
    for (var i = 0u; i < reduceSize; i = i + 1u) {
      var idx_d0 = d0;
      var idx_d1 = d1;
      var idx_d2 = d2;
      var idx_d3 = d3;
      var idx_d4 = d4;

      if (parameters.reduceDim == 0u) { idx_d0 = i; }
      else if (parameters.reduceDim == 1u) { idx_d1 = i; }
      else if (parameters.reduceDim == 2u) { idx_d2 = i; }
      else if (parameters.reduceDim == 3u) { idx_d3 = i; }
      else { idx_d4 = i; }

      let flatIdx = idx_d0 * stride0 + idx_d1 * stride1 + idx_d2 * stride2 + idx_d3 * stride3 + idx_d4 * stride4;
      let val = input[flatIdx];
      let absVal = abs(val);
      if (absVal > maxAbs) {
        maxAbs = absVal;
      }
    }

    // 2. Compute sum of squares (normalized)
    // sum = sum( (x/maxAbs)^2 )
    var sumSq = 0.0;
    for (var i = 0u; i < reduceSize; i = i + 1u) {
      var idx_d0 = d0;
      var idx_d1 = d1;
      var idx_d2 = d2;
      var idx_d3 = d3;
      var idx_d4 = d4;

      if (parameters.reduceDim == 0u) { idx_d0 = i; }
      else if (parameters.reduceDim == 1u) { idx_d1 = i; }
      else if (parameters.reduceDim == 2u) { idx_d2 = i; }
      else if (parameters.reduceDim == 3u) { idx_d3 = i; }
      else { idx_d4 = i; }

      let flatIdx = idx_d0 * stride0 + idx_d1 * stride1 + idx_d2 * stride2 + idx_d3 * stride3 + idx_d4 * stride4;
      let val = input[flatIdx];
      let norm = val / maxAbs; 
      sumSq = sumSq + (norm * norm);
    }

    // 3. Compute and write output: (x/maxAbs)^2 / sumSq
    for (var i = 0u; i < reduceSize; i = i + 1u) {
      var idx_d0 = d0;
      var idx_d1 = d1;
      var idx_d2 = d2;
      var idx_d3 = d3;
      var idx_d4 = d4;

      if (parameters.reduceDim == 0u) { idx_d0 = i; }
      else if (parameters.reduceDim == 1u) { idx_d1 = i; }
      else if (parameters.reduceDim == 2u) { idx_d2 = i; }
      else if (parameters.reduceDim == 3u) { idx_d3 = i; }
      else { idx_d4 = i; }

      let flatIdx = idx_d0 * stride0 + idx_d1 * stride1 + idx_d2 * stride2 + idx_d3 * stride3 + idx_d4 * stride4;
      let val = input[flatIdx];
      let norm = val / maxAbs;
      
      // If sumSq is effectively zero (inputs were all zero), output 0 to avoid NaN
      if (sumSq < 1e-12) {
          output[flatIdx] = 0.0;
      } else {
          output[flatIdx] = (norm * norm) / sumSq;
      }
    }
  `
};