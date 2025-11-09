/**
 * GPU-based TopK implementation for WebGPU
 * reference: https://medium.com/@avacado-cheese/how-pytorch-topk-operation-works-df061ed4cae0
 * TODO(PAPR): Further optimizations possible by adapting more advanced selection algorithms
 * 
 * Algorithm: Parallel Radix Select (simplified)
 * - Each workgroup handles one slice [M] and finds its top-k
 * - Uses local sorting with shared memory for efficiency
 * - For our case: input [N, H, W, M] -> output [N, H, W, k]
*/

import type { KernelSpec } from '../kernel';

/**
 * TopK kernel for 4D tensors along last dimension
 */
export const topk4DKernel: KernelSpec = {
  name: "topk4d",
  config: [],
  parameters: [
    {
      name: "N",
      shaderType: "u32",
    },
    {
      name: "H",
      shaderType: "u32",
    },
    {
      name: "W",
      shaderType: "u32",
    },
    {
      name: "M",  // Size of last dimension
      shaderType: "u32",
    },
    {
      name: "k",  // Number of elements to select
      shaderType: "u32",
    },
    {
      name: "largest",  // 1 for largest, 0 for smallest
      shaderType: "u32",
    },
  ],
  inputs: [
    {
      name: "input",
      shaderType: "array<f32>",
    },
  ],
  outputs: [
    {
      name: "values",
      shaderType: "array<f32>",
      size: "N * H * W * k",
    },
    {
      name: "indices",
      shaderType: "array<f32>",
      size: "N * H * W * k",
    },
  ],
  workgroupSize: [256, 1, 1], 
  workgroupCount: ["65535", "((N * H * W + 65534) / 65535)", 1],  // 2D dispatch
  workgroupVariables: [
    {
      name: "shared_values",
      shaderType: ["array<f32>", 257],  // 256 for results + 1 for kth_value
    },
    {
      name: "shared_indices",
      shaderType: ["array<u32>", 256],  // 256 for results (indices)
    },
    {
      name: "shared_temp",
      shaderType: ["array<u32>", 258],  // 256 for prefix sum + iter_total + global_output_count
    },
  ],
  shader: `
    let workgroup_1d = (global_id.x / 256u) + global_id.y * 65535u;
    let slice_idx = workgroup_1d;
    let total_slices = parameters.N * parameters.H * parameters.W;
    let local_idx = local_id.x;
    let M = parameters.M;
    let k = parameters.k;

    let is_valid_slice = slice_idx < total_slices;
    let input_offset = slice_idx * M;
    let actual_k = min(k, M);

    // Step 1: Find k-th value threshold (single-threaded for simplicity)
    // In PyTorch, this is done by radixFindKthValues + computeBlockwiseWithinKCounts
    if (local_idx == 0u) {
        if (parameters.largest == 1u) {
            shared_values[256] = -3.402823e+38;  // -FLT_MAX for largest
        } else {
            shared_values[256] = 3.402823e+38;   // FLT_MAX for smallest
        }
    }
    workgroupBarrier();

    if (local_idx == 0u && is_valid_slice) {
        // Simple selection to find k-th value
        for (var i = 0u; i < actual_k; i++) {
            shared_values[i] = input[input_offset + i];
        }

        for (var i = actual_k; i < M; i++) {
            let val = input[input_offset + i];
            var worst_val = shared_values[0];
            var worst_pos = 0u;

            for (var j = 1u; j < actual_k; j++) {
                if (parameters.largest == 1u) {
                    if (shared_values[j] < worst_val) {
                        worst_val = shared_values[j];
                        worst_pos = j;
                    }
                } else {
                    if (shared_values[j] > worst_val) {
                        worst_val = shared_values[j];
                        worst_pos = j;
                    }
                }
            }

            var should_replace = false;
            if (parameters.largest == 1u) {
                should_replace = val > worst_val;
            } else {
                should_replace = val < worst_val;
            }

            if (should_replace) {
                shared_values[worst_pos] = val;
            }
        }

        // Find k-th value (threshold)
        var kth_value = shared_values[0];
        for (var j = 1u; j < actual_k; j++) {
            if (parameters.largest == 1u) {
                if (shared_values[j] < kth_value) {
                    kth_value = shared_values[j];
                }
            } else {
                if (shared_values[j] > kth_value) {
                    kth_value = shared_values[j];
                }
            }
        }

        // Store k-th value in shared memory for all threads
        shared_values[256] = kth_value;
    }

    workgroupBarrier();
    let kth_value = shared_values[256];

    // Step 2: ALL threads scan input IN ORDER using ITERATIVE approach (matching PyTorch gatherTopK)
    // PyTorch behavior: Collect values in TWO passes
    // Pass 1: val < kth_value (strictly less than)
    // Pass 2: val == kth_value (exactly equal) - fills remaining slots

    let num_iterations = (M + 255u) / 256u;
    if (local_idx == 0u) {
        shared_temp[257] = 0u;  // global_output_count at index 257
    }
    workgroupBarrier();

    // PASS 1: Collect values strictly less than kth_value
    for (var iter = 0u; iter < num_iterations; iter++) {
        let global_output_count = shared_temp[257];
        let should_continue = global_output_count < actual_k;

        let base_idx = iter * 256u;
        let idx = base_idx + local_idx;

        var my_count = 0u;
        var my_val: f32 = 0.0;
        var my_idx: u32 = 0u;

        if (should_continue && is_valid_slice && idx < M) {
            let val = input[input_offset + idx];
            var include = false;

            if (parameters.largest == 1u) {
                include = val > kth_value;
            } else {
                include = val < kth_value;
            }

            if (include) {
                my_count = 1u;
                my_val = val;
                my_idx = idx;
            }
        }

        // Step 3: Compute prefix sum for THIS iteration (BlockScan ExclusiveSum)
        shared_temp[local_idx] = my_count;
        workgroupBarrier();

        var write_offset = 0u;
        for (var t = 0u; t < local_idx; t++) {
            write_offset += shared_temp[t];
        }

        var iter_total = 0u;
        if (local_idx == 0u) {
            for (var t = 0u; t < 256u; t++) {
                iter_total += shared_temp[t];
            }
            shared_temp[256] = iter_total;  // Store total at index 256
        }
        workgroupBarrier();
        iter_total = shared_temp[256];

        // Step 4: Write results to shared memory at correct positions
        if (should_continue && is_valid_slice && my_count > 0u) {
            let output_pos = global_output_count + write_offset;
            if (output_pos < actual_k) {
                shared_values[output_pos] = my_val;
                shared_indices[output_pos] = my_idx;
            }
        }

        workgroupBarrier();

        if (local_idx == 0u) {
            shared_temp[257] = global_output_count + iter_total;
        }
        workgroupBarrier();
    }

    workgroupBarrier();

    for (var iter = 0u; iter < num_iterations; iter++) {
        let global_output_count = shared_temp[257];
        let should_continue = global_output_count < actual_k;

        let base_idx = iter * 256u;
        let idx = base_idx + local_idx;

        var my_count = 0u;
        var my_val: f32 = 0.0;
        var my_idx: u32 = 0u;

        if (should_continue && is_valid_slice && idx < M) {
            let val = input[input_offset + idx];
            var include = false;

            include = abs(val - kth_value) < 1e-7;

            if (include) {
                my_count = 1u;
                my_val = val;
                my_idx = idx;
            }
        }

        // Step 3: Compute prefix sum for THIS iteration (BlockScan ExclusiveSum)
        shared_temp[local_idx] = my_count;
        workgroupBarrier();

        var write_offset = 0u;
        for (var t = 0u; t < local_idx; t++) {
            write_offset += shared_temp[t];
        }

        var iter_total = 0u;
        if (local_idx == 0u) {
            for (var t = 0u; t < 256u; t++) {
                iter_total += shared_temp[t];
            }
            shared_temp[256] = iter_total;  // Store total at index 256
        }
        workgroupBarrier();
        iter_total = shared_temp[256];

        // Step 4: Write results to shared memory at correct positions
        if (should_continue && is_valid_slice && my_count > 0u) {
            let output_pos = global_output_count + write_offset;
            if (output_pos < actual_k) {
                shared_values[output_pos] = my_val;
                shared_indices[output_pos] = my_idx;
            }
        }

        workgroupBarrier();

        if (local_idx == 0u) {
            shared_temp[257] = global_output_count + iter_total;
        }
        workgroupBarrier();
    }

    workgroupBarrier();

    // Step 5: Thread 0 writes results to global memory
    if (local_idx == 0u && is_valid_slice) {
        let output_offset = slice_idx * k;

        for (var i = 0u; i < actual_k; i++) {
            values[output_offset + i] = shared_values[i];
            indices[output_offset + i] = f32(shared_indices[i]);  // Cast u32 to f32
        }

        for (var i = actual_k; i < k; i++) {
            values[output_offset + i] = 0.0;
            indices[output_offset + i] = 0.0;  // f32 zero
        }
    }
  `
};

/**
 * Optimized TopK kernel using parallel 1-value selection
 */
export const topk4DKernelOptimized: KernelSpec = {
  name: "topk4d_opt",
  config: [],
  parameters: [
    {
      name: "N",
      shaderType: "u32",
    },
    {
      name: "H",
      shaderType: "u32",
    },
    {
      name: "W",
      shaderType: "u32",
    },
    {
      name: "M",
      shaderType: "u32",
    },
    {
      name: "k",
      shaderType: "u32",
    },
    {
      name: "largest",
      shaderType: "u32",
    },
  ],
  inputs: [
    {
      name: "input",
      shaderType: "array<f32>",
    },
  ],
  outputs: [
    {
      name: "values",
      shaderType: "array<f32>",
      size: "N * H * W * k",
    },
    {
      name: "indices",
      shaderType: "array<f32>",
      size: "N * H * W * k",
    },
  ],
  workgroupSize: [128, 1, 1],
  workgroupCount: ["65535", "(N * H * W + 65534) / 65535", 1],
  workgroupVariables: [
    {
      name: "shared_values",
      shaderType: ["array<f32>", 128],  
    },
    {
      name: "shared_indices",
      shaderType: ["array<u32>", 128],  
    },
    {
      name: "shared_temp",
      shaderType: ["array<u32>", 130], 
    },
  ],
  shader: `
    let workgroup_1d = (global_id.x / 128u) + global_id.y * 65535u;
    let slice_idx = workgroup_1d;
    let total_slices = parameters.N * parameters.H * parameters.W;
    let local_idx = local_id.x;
    let M = parameters.M;
    let k = parameters.k;
    let is_valid_slice = slice_idx < total_slices;
    let input_offset = slice_idx * M;
    let actual_k = min(k, M);

    let my_offset = local_idx;

    if (parameters.largest == 1u) {
        shared_values[my_offset] = -3.402823e+38;      // -FLT_MAX
    } else {
        shared_values[my_offset] = 3.402823e+38;       // FLT_MAX
    }
    shared_indices[my_offset] = 0u;
    workgroupBarrier();

    // Phase 1: Each thread finds its 1 best value from assigned elements
    let items_per_thread = (M + 127u) / 128u;
    for (var i = 0u; i < items_per_thread; i++) {
        let idx = local_idx + i * 128u;
        if (is_valid_slice && idx < M) {
            let val = input[input_offset + idx];

            // Compare with our stored value
            var val0 = shared_values[my_offset];

            // Determine if val is better than our current best
            var better = false;
            if (parameters.largest == 1u) {
                better = val > val0;
            } else {
                better = val < val0;
            }

            if (better) {
                shared_values[my_offset] = val;
                shared_indices[my_offset] = idx;
            }
        }
    }

    workgroupBarrier();

    // Phase 2: Thread 0 performs selection sort on 128 candidates to find top-k in order
    if (local_idx == 0u && is_valid_slice) {
        let output_offset = slice_idx * k;

        for (var result_idx = 0u; result_idx < actual_k; result_idx++) {
            var best_val: f32;
            var best_idx: u32 = 0u;
            var best_pos: u32 = 0u;

            if (parameters.largest == 1u) {
                best_val = -3.402823e+38;
            } else {
                best_val = 3.402823e+38;
            }

            for (var cand = 0u; cand < 128u; cand++) {
                let cand_val = shared_values[cand];

                if (parameters.largest == 1u) {
                    if (cand_val > best_val) {
                        best_val = cand_val;
                        best_idx = shared_indices[cand];
                        best_pos = cand;
                    }
                } else {
                    if (cand_val < best_val) {
                        best_val = cand_val;
                        best_idx = shared_indices[cand];
                        best_pos = cand;
                    }
                }
            }

            values[output_offset + result_idx] = best_val;
            indices[output_offset + result_idx] = f32(best_idx);

            if (parameters.largest == 1u) {
                shared_values[best_pos] = -3.402823e+38;
            } else {
                shared_values[best_pos] = 3.402823e+38;
            }
        }

        for (var i = actual_k; i < k; i++) {
            values[output_offset + i] = 0.0;
            indices[output_offset + i] = 0.0;
        }
    }
  `
};
