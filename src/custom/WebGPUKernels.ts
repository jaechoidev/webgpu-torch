/**
 * Custom WebGPU Kernels
 * These are kernels to add features that are not available in webgpu-torch v0.4.0.
 * It is recommended to check if any of these can be implemented natively in future versions.
 */

import type { Tensor } from '../tensor';
import * as torch from '../index';
import { registerKernel } from '../kernels';
import { topk4DKernel, topk4DKernelOptimized } from './TopKKernel';
import { sliceKernel } from './SliceKernel';
import { transposeKernel } from './TransposeKernel';
import { softmaxKernel, squaremaxKernel } from './SoftmaxKernel';
import { permuteKernel } from './PermuteKernel';
import { sliceAssignKernel } from './SliceAssignKernel';
import { maxpool2dKernel } from './MaxPool2dKernel';
import { clampKernel } from './ClampKernel';
class TensorCache {
  private cache = new Map<string, Tensor[]>();
  private maxSize = 10;
  private getKey(shape: number[]): string {
    return shape.join(',');
  }

  getTensor(shape: number[]): Tensor {
    const key = this.getKey(shape);
    const tensors = this.cache.get(key) || [];
    if (tensors.length > 0) {
      return tensors.pop()!;
    }
    return torch.zeros(shape);
  }

  returnTensor(tensor: Tensor): void {
    const key = this.getKey(tensor.shape);
    const tensors = this.cache.get(key) || [];

    if (tensors.length < this.maxSize) {
      tensors.push(torch.zeros(tensor.shape));
      this.cache.set(key, tensors);
    }
  }

  clear(): void {
    this.cache.clear();
  }
}
const tensorCache = new TensorCache();

class MemoryManager {
  private tensorsToClean: Tensor[] = [];
  private cleanupThreshold = 200;

  markForCleanup(tensor: Tensor): void {
    this.tensorsToClean.push(tensor);
    if (this.tensorsToClean.length >= this.cleanupThreshold) {
      this.forceCleanup();
    }
  }

  forceCleanup(): void {
    console.log(`[MEMORY_MANAGER] Cleaning up ${this.tensorsToClean.length} tensor references`);
    this.tensorsToClean.length = 0;
  }

  getStats(): { pendingCleanup: number; cacheSize: number } {
    return {
      pendingCleanup: this.tensorsToClean.length,
      cacheSize: tensorCache['cache'].size
    };
  }
}

const memoryManager = new MemoryManager();

export { memoryManager };

let kernelsRegistered = false;
export function ensureKernelsRegistered() {
  if (!kernelsRegistered) {
    registerKernel('topk4d', topk4DKernel);
    registerKernel('topk4d_opt', topk4DKernelOptimized);
    registerKernel('slice', sliceKernel);
    registerKernel('transpose_last2', transposeKernel);
    registerKernel('softmax_dim', softmaxKernel);
    registerKernel('permute', permuteKernel);
    registerKernel('slice_assign', sliceAssignKernel);
    registerKernel('maxpool2d', maxpool2dKernel);
    registerKernel('clamp', clampKernel);
    registerKernel('squaremax', squaremaxKernel)
    kernelsRegistered = true;
  }
}


/**
 * Create a 1D tensor with evenly spaced values
 * Equivalent to torch.arange(start, end, step)
 *
 * @param start - Start value (inclusive)
 * @param end - End value (exclusive)
 * @param step - Step size (default: 1)
 * @returns 1D tensor with values [start, start+step, start+2*step, ..., end-step]
 */
export function arange(start: number, end: number, step: number = 1): Tensor {
  const values: number[] = [];
  for (let i = start; i < end; i += step) {
    values.push(i);
  }
  return torch.tensor(values);
}

/**
 * Clamp tensor values to a range [min, max]
 * Equivalent to torch.clamp(input, min, max)
 *
 * @param input - Input tensor
 * @param min - Minimum value
 * @param max - Maximum value
 * @returns Clamped tensor
 */
export function clamp(input: Tensor, minVal: number, maxVal: number): Tensor {
  ensureKernelsRegistered();

  const size = input.shape.reduce((a, b) => a * b, 1);
  const params = {
    size,
    min_val: minVal,
    max_val: maxVal,
  };
  const result = input.runKernel('clamp', {}, params, [input.shape]);
  return result[0];
}

/**
 * Slice range specification: [start, end] or null for full dimension
 */
export type SliceRange = [number | null, number | null] | null;

/**
 * Slice tensor along multiple dimensions (zero-copy operation)
 * Matches PyTorch slicing: tensor[start:end, start:end, ...]
 *
 * @param input - Input tensor
 * @param ranges - Array of slice ranges, one per dimension
 * @returns Sliced tensor view (zero-copy)
 */
export function slice(input: Tensor, ranges: SliceRange[]): Tensor {
  const shape = input.shape;
  const ndim = shape.length;

  if (ranges.length !== ndim) {
    throw new Error(
      `slice: number of ranges (${ranges.length}) doesn't match tensor dimensions (${ndim})`
    );
  }

  if (ndim < 1 || ndim > 5) {
    throw new Error(`slice: GPU kernel supports 1D-5D tensors, got ${ndim}D`);
  }

  const newShape: number[] = [];
  const offsets: number[] = [];

  for (let i = 0; i < ndim; i++) {
    const dimSize = shape[i];
    const range = ranges[i];

    if (range === null || range === undefined) {
      newShape.push(dimSize);
      offsets.push(0);
    } else {
      let start = range[0];
      let end = range[1];

      if (start === null || start === undefined) {
        start = 0;
      }

      if (end === null || end === undefined) {
        end = dimSize;
      }

      if (start < 0) start = dimSize + start;
      if (end < 0) end = dimSize + end;

      start = Math.max(0, Math.min(start, dimSize));
      end = Math.max(0, Math.min(end, dimSize));

      if (start > end) {
        throw new Error(
          `slice: invalid range [${start}, ${end}] for dimension ${i} of size ${dimSize}`
        );
      }

      const newSize = end - start;
      newShape.push(newSize);
      offsets.push(start);
    }
  }

  ensureKernelsRegistered();

  const inputShape5D = [...shape, ...Array(5 - ndim).fill(1)];
  const outputShape5D = [...newShape, ...Array(5 - ndim).fill(1)];
  const offsets5D = [...offsets, ...Array(5 - ndim).fill(0)];

  // Pre-compute strides for faster GPU indexing
  const inputStride0 = inputShape5D[1] * inputShape5D[2] * inputShape5D[3] * inputShape5D[4];
  const inputStride1 = inputShape5D[2] * inputShape5D[3] * inputShape5D[4];
  const inputStride2 = inputShape5D[3] * inputShape5D[4];
  const inputStride3 = inputShape5D[4];

  const outputStride0 = outputShape5D[1] * outputShape5D[2] * outputShape5D[3] * outputShape5D[4];
  const outputStride1 = outputShape5D[2] * outputShape5D[3] * outputShape5D[4];
  const outputStride2 = outputShape5D[3] * outputShape5D[4];
  const outputStride3 = outputShape5D[4];

  const params = {
    ndim: ndim,
    inputD0: inputShape5D[0],
    inputD1: inputShape5D[1],
    inputD2: inputShape5D[2],
    inputD3: inputShape5D[3],
    inputD4: inputShape5D[4],
    outputD0: outputShape5D[0],
    outputD1: outputShape5D[1],
    outputD2: outputShape5D[2],
    outputD3: outputShape5D[3],
    outputD4: outputShape5D[4],
    offset0: offsets5D[0],
    offset1: offsets5D[1],
    offset2: offsets5D[2],
    offset3: offsets5D[3],
    offset4: offsets5D[4],
    inputStride0,
    inputStride1,
    inputStride2,
    inputStride3,
    outputStride0,
    outputStride1,
    outputStride2,
    outputStride3,
  };

  const result = input.runKernel(
    'slice',
    {},
    params,
    [newShape]
  );

  return result[0];
}

/**
 * Copy data from source to a slice of destination tensor
 * Equivalent to: dest[ranges] = src in Python
 *
 * @param dest - Destination tensor to modify
 * @param ranges - Slice ranges for assignment
 * @param src - Source tensor to copy from
 * @returns Modified destination tensor
 */
export function sliceCopy(
  dest: Tensor,
  ranges: SliceRange[],
  src: Tensor
): Tensor {
  ensureKernelsRegistered();

  if (!dest.shape || !src.shape) {
    throw new Error('sliceCopy: Input tensors must have valid shape property');
  }

  if (dest.shape.length === 0 || src.shape.length === 0) {
    throw new Error('sliceCopy: Input tensors must have at least 1 dimension');
  }

  const destShape = dest.shape;
  const srcShape = src.shape;
  const ndim = destShape.length;

  if (ranges.length !== ndim) {
    throw new Error(
      `sliceCopy: number of ranges (${ranges.length}) doesn't match tensor dimensions (${ndim})`
    );
  }

  const starts: number[] = [];
  let isFullCopy = true;
  for (let i = 0; i < ndim; i++) {
    const dimSize = destShape[i];
    const range = ranges[i];

    let start: number, end: number;
    if (range === null || range === undefined) {
      start = 0;
      end = dimSize;
    } else {
      start = range[0] ?? 0;
      end = range[1] ?? dimSize;

      if (start < 0) start = dimSize + start;
      if (end < 0) end = dimSize + end;

      start = Math.max(0, Math.min(start, dimSize));
      end = Math.max(0, Math.min(end, dimSize));
    }

    starts.push(start);

    const sliceSize = end - start;
    if (sliceSize !== srcShape[i]) {
      throw new Error(
        `sliceCopy: source shape [${srcShape}] doesn't match slice shape at dim ${i} (expected ${sliceSize}, got ${srcShape[i]})`
      );
    }

    if (start !== 0 || end !== dimSize) {
      isFullCopy = false;
    }
  }

  if (isFullCopy && JSON.stringify(destShape) === JSON.stringify(srcShape)) {
    console.log('sliceCopy: Full copy detected, returning source tensor directly');
    return src;
  }

  const destShape5D = [...destShape, ...Array(5 - ndim).fill(1)];
  const srcShape5D = [...srcShape, ...Array(5 - ndim).fill(1)];
  const starts5D = [...starts, ...Array(5 - ndim).fill(0)];

  // Pre-compute strides for faster GPU indexing
  const destStride0 = destShape5D[1] * destShape5D[2] * destShape5D[3] * destShape5D[4];
  const destStride1 = destShape5D[2] * destShape5D[3] * destShape5D[4];
  const destStride2 = destShape5D[3] * destShape5D[4];
  const destStride3 = destShape5D[4];

  const srcStride0 = srcShape5D[1] * srcShape5D[2] * srcShape5D[3] * srcShape5D[4];
  const srcStride1 = srcShape5D[2] * srcShape5D[3] * srcShape5D[4];
  const srcStride2 = srcShape5D[3] * srcShape5D[4];
  const srcStride3 = srcShape5D[4];

  const params = {
    ndim,
    destD0: destShape5D[0],
    destD1: destShape5D[1],
    destD2: destShape5D[2],
    destD3: destShape5D[3],
    destD4: destShape5D[4],
    srcD0: srcShape5D[0],
    srcD1: srcShape5D[1],
    srcD2: srcShape5D[2],
    srcD3: srcShape5D[3],
    srcD4: srcShape5D[4],
    start0: starts5D[0],
    start1: starts5D[1],
    start2: starts5D[2],
    start3: starts5D[3],
    start4: starts5D[4],
    destStride0,
    destStride1,
    destStride2,
    destStride3,
    srcStride0,
    srcStride1,
    srcStride2,
    srcStride3,
  };

  // TODO(PAPR): Use runKernelInplace to update destination buffer directly
  // need to check if performance and vram usage are improved compared to creating a new tensor
  dest.runKernelInplace('slice_assign', {}, params, src);
  return dest;
}

/**
 * Validate tensor data integrity for toArrayAsync() compatibility
 * This helper function can be used to detect common issues with tensor downloads
 */
export async function validateTensorData(tensor: Tensor, expectedElements?: number): Promise<{
  isValid: boolean;
  actualElements: number;
  sampleData?: number[];
  hasZeros?: boolean;
  hasValidRange?: boolean;
}> {
  try {
    const data = await tensor.toArrayAsync();
    const flatData: number[] = [];
    function flatten(arr: any): void {
      if (Array.isArray(arr)) {
        for (const item of arr) {
          flatten(item);
        }
      } else if (typeof arr === 'number') {
        flatData.push(arr);
      }
    }
    flatten(data);

    const actualElements = flatData.length;
    const hasZeros = flatData.some(val => Math.abs(val) < 1e-6);
    const hasValidRange = flatData.some(val => !isNaN(val) && isFinite(val));

    const sampleData = flatData.slice(0, 10);

    const isValid = expectedElements ? actualElements === expectedElements : true;

    return {
      isValid,
      actualElements,
      sampleData,
      hasZeros,
      hasValidRange
    };
  } catch (error) {
    return {
      isValid: false,
      actualElements: 0,
      sampleData: undefined,
      hasZeros: undefined,
      hasValidRange: undefined
    };
  }
}

/**
 * Find K smallest elements along a dimension (for K-NN)
 */

const USE_GPU_TOPK = true; 

export async function topk(
  input: Tensor,
  k: number,
  dim: number = -1,
  largest: boolean = true,
  sorted: boolean = true,
  useOptimized: boolean = false
): Promise<[Tensor, Tensor]> {
  const shape = input.shape;

  const ndim = shape.length;
  if (dim < 0) dim = ndim + dim;

  if (ndim === 4 && dim === 3) {
    if (USE_GPU_TOPK) {
      return topk4DGPU(input, k, largest, useOptimized);
    } else {
      return topk4DBatched(input, shape, k, largest, sorted);
    }
  }
  const data = await input.toArrayAsync();
  if (ndim === 2 && dim === 1) {
    return topk2D(data as number[][], shape, k, largest, sorted);
  }

  throw new Error(`topk: ${ndim}D tensors with dim=${dim} not yet implemented`);
}

/**
 * Specialized topk for 2D tensors along dim=1
 * Input shape: [N, M], returns [N, k]
 */
async function topk2D(
  data: number[][],
  shape: number[],
  k: number,
  largest: boolean,
  sorted: boolean
): Promise<[Tensor, Tensor]> {
  const [N, M] = shape;
  const values: number[][] = [];
  const indices: number[][] = [];

  for (let i = 0; i < N; i++) {
    const row = data[i];

    const pairs: [number, number][] = row.map((val, idx) => [val, idx]);

    pairs.sort((a, b) => {
      if (largest) {
        return b[0] - a[0]; 
      } else {
        return a[0] - b[0]; 
      }
    });

    const topKPairs = pairs.slice(0, k);

    if (!sorted) {
      topKPairs.sort((a, b) => a[1] - b[1]);
    }

    values.push(topKPairs.map(p => p[0]));
    indices.push(topKPairs.map(p => p[1]));
  }

  return [
    torch.tensor(values),
    torch.tensor(indices)
  ];
}

/**
 * GPU-based topk for 4D tensors using WebGPU compute shader
 * Input shape: [N, H, W, M], returns [N, H, W, k]
 */
async function topk4DBatched(
  input: Tensor,
  shape: number[],
  k: number,
  largest: boolean,
  sorted: boolean
): Promise<[Tensor, Tensor]> {
  const [N, H, W, M] = shape;

  ensureKernelsRegistered();

  const params = {
    N,
    H,
    W,
    M,
    k,
    largest: largest ? 1 : 0,
  };

  try {
    const result = input.runKernel(
      'topk4d',
      {},  
      params,
      [[N, H, W, k], [N, H, W, k]] 
    );

    const valuesOutput = result[0];
    const indicesOutput = result[1];

    return [valuesOutput, indicesOutput];
  } catch (error) {
    throw new Error(`GPU topk failed and CPU fallback not viable: ${error}`);
  }
}

/**
 * Specialized topk for 4D tensors along dim=3 (last dimension)
 * Input shape: [N, H, W, M], returns [N, H, W, k]
 */
async function topk4D(
  data: any,
  shape: number[],
  k: number,
  largest: boolean,
  sorted: boolean
): Promise<[Tensor, Tensor]> {
  const [N, H, W, M] = shape;

  const values_flat = new Float32Array(N * H * W * k);
  const indices_flat = new Float32Array(N * H * W * k);

  let flatIdx = 0;
  for (let n = 0; n < N; n++) {
    for (let h = 0; h < H; h++) {
      for (let w = 0; w < W; w++) {
        let row: number[];
        if (Array.isArray(data[n]?.[h]?.[w])) {
          row = data[n][h][w];
        } else if (Array.isArray(data[n]?.[h]) && !Array.isArray(data[n][h][0])) {
          row = data[n][h];
        } else if (Array.isArray(data[n]) && !Array.isArray(data[n][0])) {
          row = data[n];
        } else {
          throw new Error(`topk4D: Cannot access data at [${n}][${h}][${w}]`);
        }

        if (!row || row.length !== M) {
          throw new Error(`topk4D: Expected row length ${M}, got ${row?.length} at [${n}][${h}][${w}]`);
        }
        const pairs: [number, number][] = row.map((val, idx) => [val, idx]);

        pairs.sort((a, b) => {
          if (largest) {
            return b[0] - a[0]; 
          } else {
            return a[0] - b[0]; 
          }
        });
        const topKPairs = pairs.slice(0, k);

        if (!sorted) {
          topKPairs.sort((a, b) => a[1] - b[1]);
        }

        for (let i = 0; i < k; i++) {
          values_flat[flatIdx] = topKPairs[i][0];
          indices_flat[flatIdx] = topKPairs[i][1];
          flatIdx++;
        }
      }
    }
  }

  const values_1d = torch.tensor(Array.from(values_flat));
  const indices_1d = torch.tensor(Array.from(indices_flat));

  const values_result = torch.reshape(values_1d, [N, H, W, k]);
  const indices_result = torch.reshape(indices_1d, [N, H, W, k]);

  return [values_result, indices_result];
}

/**
 * GPU-based topk for 4D tensors using custom WebGPU kernel
 * Input shape: [N, H, W, M], returns [N, H, W, k]
 */
async function topk4DGPU(
  input: Tensor,
  k: number,
  largest: boolean = true,
  useOptimized: boolean = false
): Promise<[Tensor, Tensor]> {
  ensureKernelsRegistered();

  const shape = input.shape;
  if (shape.length !== 4) {
    throw new Error(`topk4DGPU: Expected 4D tensor, got ${shape.length}D`);
  }

  const [N, H, W, M] = shape;
  const outputShape = [N, H, W, k];
  const kernelName = useOptimized ? 'topk4d_opt' : 'topk4d';

  const outputs = input.runKernel(
    kernelName,
    {}, // config (empty for this kernel)
    {
      N,
      H,
      W,
      M,
      k,
      largest: largest ? 1 : 0
    },
    [outputShape, outputShape] 
  );

  return [outputs[0], outputs[1]];
}

/**
 * Repeat/tile a tensor along specified dimensions
 * Similar to PyTorch's repeat() or NumPy's tile()
 *
 * @param input - Input tensor
 * @param repeats - Number of repetitions for each dimension
 * @returns Tiled tensor
 */
export async function repeat(input: Tensor, repeats: number[]): Promise<Tensor> {
  if (input.shape.length !== repeats.length) {
    throw new Error(`repeat: number of dimensions in repeats (${repeats.length}) must match input dimensions (${input.shape.length})`);
  }

  const outputShape = input.shape.map((size, i) => size * repeats[i]);
  const outputSize = outputShape.reduce((a, b) => a * b, 1);

  const inputData = await input.toArrayAsync();

  function flattenArray(arr: any): number[] {
    const result: number[] = [];
    function flatten(a: any) {
      if (Array.isArray(a)) {
        for (const item of a) {
          flatten(item);
        }
      } else {
        result.push(a);
      }
    }
    flatten(arr);
    return result;
  }

  const inputFlat = flattenArray(inputData);
  const outputFlat = new Float32Array(outputSize);

  const inputStrides = [];
  const outputStrides = [];
  let stride = 1;
  for (let i = input.shape.length - 1; i >= 0; i--) {
    inputStrides[i] = stride;
    stride *= input.shape[i];
  }
  stride = 1;
  for (let i = outputShape.length - 1; i >= 0; i--) {
    outputStrides[i] = stride;
    stride *= outputShape[i];
  }

  for (let outIdx = 0; outIdx < outputSize; outIdx++) {
    let remaining = outIdx;
    const outIndices = [];
    for (let d = outputShape.length - 1; d >= 0; d--) {
      outIndices[d] = remaining % outputShape[d];
      remaining = Math.floor(remaining / outputShape[d]);
    }

    const inIndices = outIndices.map((idx, d) => idx % input.shape[d]);

    let inIdx = 0;
    for (let d = 0; d < input.shape.length; d++) {
      inIdx += inIndices[d] * inputStrides[d];
    }

    outputFlat[outIdx] = inputFlat[inIdx];
  }

  const flatTensor = torch.tensor(Array.from(outputFlat));
  return torch.reshape(flatTensor, outputShape);
}

/**
 * Softmax normalization along a dimension (GPU-based fused kernel)
 * softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * @param input - Input tensor
 * @param dim - Dimension to apply softmax (default: -1, last dimension only)
 * @param keepdim - Whether to keep the dimension (default: false)
 * @returns Softmax normalized tensor
 */
export function softmax(input: Tensor, dim: number = -1, keepdim: boolean = false): Tensor {
  ensureKernelsRegistered();

  const shape = input.shape;
  const ndim = shape.length;

  if (dim < 0) dim = ndim + dim;

  if (dim < 0 || dim >= ndim) {
    throw new Error(`softmax: invalid dimension ${dim} for ${ndim}D tensor`);
  }

  if (ndim < 1 || ndim > 5) {
    throw new Error(`softmax: GPU kernel supports 1D-5D tensors, got ${ndim}D`);
  }

  const shape5D = [...shape, ...Array(5 - ndim).fill(1)];

  const reduceDim5D = dim;

  const totalSize = shape5D.reduce((a, b) => a * b, 1);
  const reduceSize = shape5D[reduceDim5D];
  const numSlices = totalSize / reduceSize;

  const config = {
    numSlices: numSlices
  };

  const params = {
    ndim: ndim,
    reduceDim: reduceDim5D,
    D0: shape5D[0],
    D1: shape5D[1],
    D2: shape5D[2],
    D3: shape5D[3],
    D4: shape5D[4],
  };

  const result = input.runKernel('softmax_dim', config, params, [shape]);
//   const result = input.runKernel('softmax_fast', config, params, [shape]);

  if (!keepdim) {
    return torch.squeeze(result[0], dim);
  }

  return result[0];
}
export function squaremax(input: Tensor, dim: number = -1, keepdim: boolean = false): Tensor {
  ensureKernelsRegistered();

  const shape = input.shape;
  const ndim = shape.length;

  if (dim < 0) dim = ndim + dim;

  if (dim < 0 || dim >= ndim) {
    throw new Error(`softmax: invalid dimension ${dim} for ${ndim}D tensor`);
  }

  if (ndim < 1 || ndim > 5) {
    throw new Error(`softmax: GPU kernel supports 1D-5D tensors, got ${ndim}D`);
  }

  const shape5D = [...shape, ...Array(5 - ndim).fill(1)];

  const reduceDim5D = dim;

  const totalSize = shape5D.reduce((a, b) => a * b, 1);
  const reduceSize = shape5D[reduceDim5D];
  const numThreads = totalSize / reduceSize;

  const config = {
    numThreads: numThreads
  };

  const params = {
    ndim: ndim,
    reduceDim: reduceDim5D,
    D0: shape5D[0],
    D1: shape5D[1],
    D2: shape5D[2],
    D3: shape5D[3],
    D4: shape5D[4],
  };

  const result = input.runKernel('squaremax', config, params, [shape]);
//   const result = input.runKernel('softmax_fast', config, params, [shape]);

  if (!keepdim) {
    return torch.squeeze(result[0], dim);
  }

  return result[0];
}
/**
 * Extract scalar value from 0-d or 1-element tensor
 * Replacement for .item()
 */
export async function item(tensor: Tensor): Promise<number> {
  const array = await tensor.toArrayAsync();
  if (Array.isArray(array)) {
    return array[0] as number;
  }
  return array as number;
}

/**
 * Permute tensor dimensions (arbitrary dimension reordering)
 * Equivalent to PyTorch's tensor.permute(*dims)
 *
 * @param input - Input tensor (1D-5D)
 * @param dims - Permutation pattern (e.g., [1,0,2,3] swaps first two dimensions)
 * @returns Permuted tensor
 */
export function permute(input: Tensor, dims: number[]): Tensor {
  ensureKernelsRegistered();

  const shape = input.shape;
  const ndim = shape.length;

  if (dims.length !== ndim) {
    throw new Error(`permute: permutation must have same length as number of dimensions. Got permutation length ${dims.length} for ${ndim}D tensor`);
  }

  const sortedDims = [...dims].sort((a, b) => a - b);
  for (let i = 0; i < ndim; i++) {
    if (sortedDims[i] !== i) {
      throw new Error(`permute: invalid permutation ${dims}. Must contain each dimension index 0..${ndim - 1} exactly once`);
    }
  }

  let isIdentity = true;
  for (let i = 0; i < ndim; i++) {
    if (dims[i] !== i) {
      isIdentity = false;
      break;
    }
  }
  if (isIdentity) {
    return input;  
  }

  const newShape = dims.map(d => shape[d]);

  const inputShape5D = [...shape, ...Array(5 - ndim).fill(1)];
  const outputShape5D = [...newShape, ...Array(5 - ndim).fill(1)];
  const perm5D = [...dims, ...Array(5 - ndim).fill(0).map((_, i) => ndim + i)];

  const params = {
    ndim: ndim,
    perm0: perm5D[0],
    perm1: perm5D[1],
    perm2: perm5D[2],
    perm3: perm5D[3],
    perm4: perm5D[4],
    inputD0: inputShape5D[0],
    inputD1: inputShape5D[1],
    inputD2: inputShape5D[2],
    inputD3: inputShape5D[3],
    inputD4: inputShape5D[4],
    outputD0: outputShape5D[0],
    outputD1: outputShape5D[1],
    outputD2: outputShape5D[2],
    outputD3: outputShape5D[3],
    outputD4: outputShape5D[4],
  };

  const result = input.runKernel('permute', {}, params, [newShape]);
  return result[0];
}

/**
 * Transpose last two dimensions of a tensor (CPU fallback)
 *
 * @param input - Input tensor of any shape
 * @returns Tensor with last two dimensions transposed
 *
 */
export function transpose(input: Tensor): Tensor {
  ensureKernelsRegistered();

  if (!(window as any)._transpose_v13_logged) {
    (window as any)._transpose_v13_logged = true;
  }

  const shape = input.shape;
  const ndim = shape.length;

  if (ndim < 2) {
    throw new Error(`transpose: tensor must have at least 2 dimensions, got ${ndim}D`);
  }

  // For 2D tensors, use built-in .t()
  if (ndim === 2) {
    return input.t();
  }

  // For 3D-5D tensors, use GPU kernel
  if (ndim >= 3 && ndim <= 5) {
    const newShape = [...shape];
    [newShape[ndim - 2], newShape[ndim - 1]] = [newShape[ndim - 1], newShape[ndim - 2]];

    const inputShape5D = [...shape, ...Array(5 - ndim).fill(1)];
    const outputShape5D = [...newShape, ...Array(5 - ndim).fill(1)];

    const params = {
      ndim: ndim,
      inputD0: inputShape5D[0],
      inputD1: inputShape5D[1],
      inputD2: inputShape5D[2],
      inputD3: inputShape5D[3],
      inputD4: inputShape5D[4],
      outputD0: outputShape5D[0],
      outputD1: outputShape5D[1],
      outputD2: outputShape5D[2],
      outputD3: outputShape5D[3],
      outputD4: outputShape5D[4],
    };

    const result = input.runKernel('transpose_last2', {}, params, [newShape]);
    return result[0];
  }

  throw new Error(`transpose: GPU kernel supports 2D-5D tensors, got ${ndim}D`);
}

/**
 * MaxPool2d - 2D max pooling with kernel=2, stride=2
 *
 * @param input - Input tensor [N, C, H, W]
 * @returns Pooled tensor [N, C, H/2, W/2]
 */
export function maxpool2d(input: Tensor): Tensor {
  ensureKernelsRegistered();

  const [N, C, H, W] = input.shape;

  if (input.shape.length !== 4) {
    throw new Error(`maxpool2d expects 4D input [N,C,H,W], got ${input.shape.length}D`);
  }

  const H_out = Math.floor(H / 2);
  const W_out = Math.floor(W / 2);

  const params = {
    N,
    C,
    H_in: H,
    W_in: W,
    H_out,
    W_out,
  };

  const outputShape = [N, C, H_out, W_out];
  const result = input.runKernel('maxpool2d', {}, params, [outputShape]);

  return result[0];
}