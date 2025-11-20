/**
 * GPU kernel for softmax normalization along any dimension
 *   softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))
 */

import { KernelSpec } from '../kernel';

/**
 * Fused softmax kernel for any dimension - Level 4 (Subgroup Operations)
 * Uses WebGPU subgroup operations for warp-level reductions
 */
export const softmaxKernel: KernelSpec = {
  name: "softmax_dim",
  config: [
    { name: "numSlices" }
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
    "numSlices",
    1,
    1
  ],
  // Minimal shared memory - only need storage for cross-subgroup reductions
  workgroupVariables: [
    { name: "sharedMax", shaderType: ["array<f32>", 8] },  // One per subgroup (256/32 = 8)
    { name: "sharedSum", shaderType: ["array<f32>", 8] }
  ],
  shader: `
// Level 4: Subgroup operations (warp-level primitives)
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

// Each workgroup processes one slice (row)
let sliceIdx = workgroup_id.x;  // Workgroup ID
let tid = local_id.x;  // Thread ID within workgroup (0-255)
let sg_id = subgroup_invocation_id;  // Thread ID within subgroup
let sg_size = subgroup_size;  // Subgroup size (typically 32)
let subgroup_idx = tid / sg_size;  // Which subgroup this thread belongs to

// Decode slice index to coordinates
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

// ===== STEP 1: Find maximum value =====
// Each thread computes local max for its chunk
var localMax = -3.402823e+38;  // -FLT_MAX
for (var i = tid; i < reduceSize; i = i + 256u) {
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
  if (val > localMax) {
    localMax = val;
  }
}

// Warp-level reduction using subgroup operations
// subgroupMax reduces within the subgroup in O(log subgroup_size) steps
let subgroupMaxVal = subgroupMax(localMax);

// Store subgroup result to shared memory (only first thread in each subgroup writes)
if (sg_id == 0u) {
  sharedMax[subgroup_idx] = subgroupMaxVal;
}
workgroupBarrier();

// Final reduction across subgroups (typically 8 subgroups in a 256-thread workgroup)
var maxVal = sharedMax[0];
if (tid < 8u) {
  for (var j = 1u; j < 8u; j = j + 1u) {
    if (sharedMax[j] > maxVal) {
      maxVal = sharedMax[j];
    }
  }
}
// Broadcast global max to all threads
maxVal = subgroupBroadcast(maxVal, 0u);
workgroupBarrier();

// ===== STEP 2: Compute sum of exp(x - max) =====
var localSum = 0.0;
for (var i = tid; i < reduceSize; i = i + 256u) {
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
  localSum = localSum + exp(input[flatIdx] - maxVal);
}

// Warp-level reduction using subgroup operations
let subgroupSumVal = subgroupAdd(localSum);

// Store subgroup result to shared memory
if (sg_id == 0u) {
  sharedSum[subgroup_idx] = subgroupSumVal;
}
workgroupBarrier();

// Final reduction across subgroups
var sumExp = 0.0;
if (tid < 8u) {
  for (var j = 0u; j < 8u; j = j + 1u) {
    sumExp = sumExp + sharedSum[j];
  }
}
// Broadcast global sum to all threads
sumExp = subgroupBroadcast(sumExp, 0u);
workgroupBarrier();

// ===== STEP 3: Write normalized output =====
for (var i = tid; i < reduceSize; i = i + 256u) {
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
