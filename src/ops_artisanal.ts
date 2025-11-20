import { shouldCreateGradient } from "./autograd";
import { Tensor } from "./tensor";
import type { Deviceish } from "./device";
import type { Dtype } from "./dtype";
import { slice, permute } from "./custom/WebGPUKernels";
import {
    broadcastBatchedMatmul,
    contiguousStridedShape,
    defaultStrides,
    reshapeBatchedMatmul,
    shapesAreEqual,
    type Shape,
    type StridedShape,
    type Strides,
    shapeSize,
    check,
    validateIdx,
    validateDimLength,
    canonicalizeDim,
} from "./shape";
import type { TensorData, TensorSpec, MemoryFormat } from "./tensor";
import { KernelParamsInput } from "./kernel";
import { GatherFunction, LinearFunction } from "./functions_artisanal";

/**
 * WEBGPU 64-CHANNEL BUG WORKAROUND
 *
 * WebGPU has a fundamental limit where compute operations only execute for the first 64 channels.
 * This helper splits channel operations into batches of 64 to work around the limitation.
 *
 * @param totalChannels - Total number of channels to process
 * @param channelDim - Which dimension contains channels (e.g., 0 for weight dim0, 1 for NCHW dim1)
 * @param operation - Function that processes a channel range and returns a tensor
 * @returns Concatenated result tensor with all channels
 */
function splitChannelOperation(
    totalChannels: number,
    channelDim: number,
    operation: (startChannel: number, endChannel: number) => Tensor
): Tensor {
    const CHANNEL_BATCH_SIZE = 64;

    if (totalChannels <= CHANNEL_BATCH_SIZE) {
        // No split needed
        return operation(0, totalChannels);
    }

    // console.error(`[SPLIT_CHANNEL_OP] Splitting ${totalChannels} channels into batches of ${CHANNEL_BATCH_SIZE}`);

    const results: Tensor[] = [];
    for (let start = 0; start < totalChannels; start += CHANNEL_BATCH_SIZE) {
        const end = Math.min(start + CHANNEL_BATCH_SIZE, totalChannels);
        // console.error(`[SPLIT_CHANNEL_OP] Processing channels ${start}-${end-1}`);
        results.push(operation(start, end));
    }

    // Concatenate all results along the channel dimension
    const finalResult = cat(results, channelDim);
    // console.error(`[SPLIT_CHANNEL_OP] Concatenated ${results.length} batches, final shape:`, finalResult.shape);

    return finalResult;
}

export function cat(inputs: Tensor[], dim: number = 0): Tensor {
    if (inputs.length === 0) {
        throw new Error("cat requires at least one tensor");
    }
    if (inputs.length === 1) {
        return inputs[0];
    }

    // Normalize negative dimension
    const ndim = inputs[0].shape.length;
    if (dim < 0) {
        dim = ndim + dim;
    }
    if (dim < 0 || dim >= ndim) {
        throw new Error(`Invalid dimension ${dim} for tensor with ${ndim} dimensions`);
    }

    // Validate all tensors have same rank and compatible shapes
    for (let i = 1; i < inputs.length; i++) {
        if (inputs[i].shape.length !== ndim) {
            throw new Error(`All tensors must have same number of dimensions`);
        }
        for (let d = 0; d < ndim; d++) {
            if (d !== dim && inputs[i].shape[d] !== inputs[0].shape[d]) {
                throw new Error(`All tensors must have same shape except in concat dimension`);
            }
        }
    }

    // Calculate output shape
    const outputShape = inputs[0].shape.slice();
    outputShape[dim] = inputs.reduce((sum, t) => sum + t.shape[dim], 0);

    const dtype = inputs[0].dtype;
    const device = inputs[0].device;

    // Log large concatenations for debugging
    // const outputSize = outputShape.reduce((a, b) => a * b, 1) * 4; // assuming float32
    // if (outputSize > 100 * 1024 * 1024) { // > 100MB
    //     console.warn(`[CAT] Large concatenation: ${inputs.length} inputs, output shape [${outputShape}], ` +
    //                  `output size: ${(outputSize / 1024 / 1024).toFixed(2)} MB, dim: ${dim}`);
    //     inputs.forEach((t, i) => {
    //         const inputSize = t.shape.reduce((a, b) => a * b, 1) * 4;
    //         console.warn(`  Input ${i}: shape [${t.shape}], size: ${(inputSize / 1024 / 1024).toFixed(2)} MB`);
    //     });
    // }

    // Debug: Check device type

    // WebGPU default limit is 8 storage buffers per shader stage
    // cat kernel uses: N inputs + 1 output + 1 parameters = N+2 storage buffers
    // Therefore, max inputs = 8 - 2 = 6 to stay within default limit
    // HOWEVER, we also need to respect maxStorageBufferBindingSize (128MB by default)
    // So we use a smaller batch size to avoid creating outputs > 128MB
    const MAX_CAT_INPUTS = 3; // Reduced to avoid exceeding maxStorageBufferBindingSize

    // If more than MAX_CAT_INPUTS inputs, use hierarchical batching (like PyTorch)
    // Works for both WebGPU and CPU devices
    if (inputs.length > MAX_CAT_INPUTS && ndim <= 6) {
        // Concatenate in batches of MAX_CAT_INPUTS
        const batches: Tensor[] = [];
        for (let i = 0; i < inputs.length; i += MAX_CAT_INPUTS) {
            const batch = inputs.slice(i, Math.min(i + MAX_CAT_INPUTS, inputs.length));
            batches.push(cat(batch, dim)); // Recursive call
        }
        return cat(batches, dim); // Final concatenation
    }

    // Check if we can use GPU kernel (up to MAX_CAT_INPUTS inputs on WebGPU with default limits)
    if (device.type === "webgpu" && inputs.length <= MAX_CAT_INPUTS && ndim <= 6) {
        const totalSize = outputShape.reduce((a, b) => a * b, 1);

        // Build parameters
        const params: any = {
            rank: ndim,
            dim: dim,
            outputSize: totalSize,
            outputShape0: ndim > 0 ? outputShape[0] : 1,
            outputShape1: ndim > 1 ? outputShape[1] : 1,
            outputShape2: ndim > 2 ? outputShape[2] : 1,
            outputShape3: ndim > 3 ? outputShape[3] : 1,
            outputShape4: ndim > 4 ? outputShape[4] : 1,
            outputShape5: ndim > 5 ? outputShape[5] : 1,
            numInputs: inputs.length,
        };

        // Add parameters for each input (up to MAX_CAT_INPUTS = 6)
        for (let i = 0; i < MAX_CAT_INPUTS; i++) {
            if (i < inputs.length) {
                const input = inputs[i];
                // DimSize is the size along the concatenation dimension
                params[`input${i}DimSize`] = input.shape[dim];
                params[`input${i}Stride0`] = ndim > 0 ? input.strides[0] : 1;
                params[`input${i}Stride1`] = ndim > 1 ? input.strides[1] : 1;
                params[`input${i}Stride2`] = ndim > 2 ? input.strides[2] : 1;
                params[`input${i}Stride3`] = ndim > 3 ? input.strides[3] : 1;
                params[`input${i}Stride4`] = ndim > 4 ? input.strides[4] : 1;
                params[`input${i}Stride5`] = ndim > 5 ? input.strides[5] : 1;
            } else {
                // Dummy values for unused inputs
                params[`input${i}DimSize`] = 1;
                params[`input${i}Stride0`] = 1;
                params[`input${i}Stride1`] = 1;
                params[`input${i}Stride2`] = 1;
                params[`input${i}Stride3`] = 1;
                params[`input${i}Stride4`] = 1;
                params[`input${i}Stride5`] = 1;
            }
        }

        // Build additional inputs array for inputs 1-5 (MAX_CAT_INPUTS-1)
        const additionalInputs: Tensor[] = [];
        for (let i = 1; i < MAX_CAT_INPUTS; i++) {
            additionalInputs.push(i < inputs.length ? inputs[i] : inputs[0]); // Dummy for unused
        }

        // Use runKernel on first input, passing additional inputs
        return inputs[0].runKernel("cat", {dtype}, params, [outputShape], ...additionalInputs)[0];
    }

    // No WebGPU kernel available - throw descriptive error
    throw new Error(`cat() on device "${device.type}" with ${inputs.length} inputs and ${ndim}D tensors is not supported. WebGPU kernel only supports ≤${MAX_CAT_INPUTS} inputs and ≤6D tensors on WebGPU device. Check that tensors are created on WebGPU device, not CPU.`);
}

/**
 * Concatenates a sequence of tensors along a new dimension.
 * All tensors must have the same shape.
 *
 * @param inputs - Sequence of tensors to stack
 * @param dim - Dimension along which to stack (default: 0)
 * @returns Stacked tensor with one additional dimension
 *
 * Example:
 *   a = torch.tensor([1, 2, 3])  // shape: [3]
 *   b = torch.tensor([4, 5, 6])  // shape: [3]
 *   torch.stack([a, b], dim=0)   // shape: [2, 3]
 *   torch.stack([a, b], dim=1)   // shape: [3, 2]
 *
 * Implementation: Uses batched unsqueeze + cat to avoid exceeding 5D tensor limit
 */
export function stack(inputs: Tensor[], dim: number = 0): Tensor {
    if (inputs.length === 0) {
        throw new Error("stack requires at least one tensor");
    }
    if (inputs.length === 1) {
        return inputs[0].unsqueeze(dim);
    }

    // Validate all tensors have same shape
    const shape = inputs[0].shape;
    for (let i = 1; i < inputs.length; i++) {
        if (inputs[i].shape.length !== shape.length) {
            throw new Error(`All tensors must have same number of dimensions for stack`);
        }
        for (let d = 0; d < shape.length; d++) {
            if (inputs[i].shape[d] !== shape[d]) {
                throw new Error(`All tensors must have same shape for stack. Got ${inputs[i].shape} vs ${shape}`);
            }
        }
    }

    // Normalize dimension
    const ndim = shape.length;
    if (dim < 0) {
        dim = ndim + 1 + dim;  // +1 because we're adding a dimension
    }
    if (dim < 0 || dim > ndim) {
        throw new Error(`Invalid dimension ${dim} for stack (valid range: [-${ndim+1}, ${ndim}])`);
    }

    // WORKAROUND for webgpu-torch limitation: cat() only supports ≤6D tensors AND ≤8 inputs
    if (ndim >= 6) {
        throw new Error(`stack() would create ${ndim + 1}D tensor, but only ≤6D is supported`);
    }

    if (inputs.length > 8) {
        // For >8 inputs, use recursive batching
        const batchSize = 7;
        const batches: Tensor[] = [];

        for (let i = 0; i < inputs.length; i += batchSize) {
            const batchEnd = Math.min(i + batchSize, inputs.length);
            const batch = inputs.slice(i, batchEnd);
            // Recursively stack batches
            batches.push(stack(batch, dim));
        }

        // Cat the batched stacks along the stack dimension
        return cat(batches, dim);
    } else {
        // Normal case: unsqueeze each tensor at dim, then cat at dim
        const unsqueezed = inputs.map(t => t.unsqueeze(dim));
        return cat(unsqueezed, dim);
    }
}

export function clone(
    input: Tensor,
    memoryFormat: MemoryFormat = "preserveFormat"
): Tensor {
    if (shouldCreateGradient(input)) {
        throw new Error("clone gradient not supported yet");
        // return CloneFunction.apply(input);
    } else {
        const newStorage = input.storage.clone();
        return new Tensor({
            data: newStorage,
            shape: input.shape,
            dtype: input.dtype,
            requiresGrad: input.requiresGrad,
        });
    }
}

/**
 * Applies a 2D convolution over an input image composed of several input planes.
 *
 * #### Forward
 * ```
 * output[y, x] = sum(Ky, sum(Kx, input[y + ky, x + kx] * weight[ky, kx])) + bias
 * ```
 *
 * @param input input tensor of shape [B, inChannels, iH, iW]
 * @param weight filters of shape [outChannels, inChannels, kH, kW]
 * @param bias optional bias tensor of shape [outChannels]
 * @param stride the stride of the convolving kernel. Can be a single number or a tuple (sH, sW). Default: 1
 * @param padding implicit padding on both sides of the kernel. Can be a single number or a tuple (padH, padW). Default: 0
 *     `padding="valid"` is the same as no padding. `padding="same"` pads the input so the output has the shape as the input.
 *     However, this mode can only be used when `stride` is 1.
 * @returns
 */
export function conv2d(
    input: Tensor,
    weight: Tensor,
    bias?: Tensor,
    stride?: number | [number, number],
    padding?: number | [number, number] | "valid" | "same"
): Tensor {
    if (shouldCreateGradient(input, weight)) {
        throw new Error("conv2d gradient not supported yet");
    } else {
        if (input.shape.length !== 4 || weight.shape.length !== 4) {
            throw new Error(
                `Expected image tensor, got ${input.shape} and kernel ${weight.shape}`
            );
        }
        if (input.shape[1] !== weight.shape[1]) {
            throw new Error(
                `Expected number of chennels in input image to match number of channels in kernel, got ${input.shape} and ${weight.shape}`
            );
        }

        // Parse stride parameter
        const strideH = typeof stride === 'number' ? stride : (stride?.[0] ?? 1);
        const strideW = typeof stride === 'number' ? stride : (stride?.[1] ?? 1);

        // Parse padding parameter
        let padH = 0;
        let padW = 0;
        if (padding === 'valid') {
            padH = 0;
            padW = 0;
        } else if (padding === 'same') {
            // For 'same' padding with stride=1, calculate padding to maintain size
            if (strideH !== 1 || strideW !== 1) {
                throw new Error("'same' padding only supported with stride=1");
            }
            padH = Math.floor((weight.shape[2] - 1) / 2);
            padW = Math.floor((weight.shape[3] - 1) / 2);
        } else if (typeof padding === 'number') {
            padH = padding;
            padW = padding;
        } else if (Array.isArray(padding)) {
            padH = padding[0];
            padW = padding[1];
        }

        // Calculate output dimensions using proper conv2d formula
        // output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
        const outputHeight = Math.floor((input.shape[2] + 2 * padH - weight.shape[2]) / strideH) + 1;
        const outputWidth = Math.floor((input.shape[3] + 2 * padW - weight.shape[3]) / strideW) + 1;

        const batchSize = input.shape[0];
        const inputChannels = input.shape[1];
        const outputChannels = weight.shape[0];
        const kernelHeight = weight.shape[2];
        const kernelWidth = weight.shape[3];

        // ========== Im2col + GEMM Approach with Input Channel Splitting ==========
        // Split input channels to keep im2col buffer under 200MB (reduced for memory optimization)
        // im2col output: [B * H_out * W_out, C_in_split * K_H * K_W]
        const maxInputChannels = Math.max(1, Math.floor(200000000 / (batchSize * outputHeight * outputWidth * kernelHeight * kernelWidth * 4)));

        const result = splitChannelOperation(
            outputChannels,
            1, // Channel dimension in NCHW format
            (startOutChannel: number, endOutChannel: number) => {
                const outChannelCount = endOutChannel - startOutChannel;

                // Accumulate results across input channel batches
                let accumulated: Tensor | null = null;

                for (let ic = 0; ic < inputChannels; ic += maxInputChannels) {
                    const currentInputChannels = Math.min(maxInputChannels, inputChannels - ic);

                    // Slice input channels
                    const inputSlice = slice(input, [null, [ic, ic + currentInputChannels], null, null]);

                    // Step 1: Apply im2col transformation
                    const im2colParams = {
                        batchSize: batchSize,
                        inputChannels: currentInputChannels,
                        inputHeight: input.shape[2],
                        inputWidth: input.shape[3],
                        kernelHeight: kernelHeight,
                        kernelWidth: kernelWidth,
                        outputHeight: outputHeight,
                        outputWidth: outputWidth,
                        padH: padH,
                        padW: padW,
                        strideH: strideH,
                        strideW: strideW,
                        dilationH: 1,
                        dilationW: 1,
                    };

                    // Use im2col_transposed to avoid expensive transpose operation
                    const im2colOutput = inputSlice.runKernel(
                        "im2col_transposed",
                        { dtype: inputSlice.dtype },
                        im2colParams,
                        [[currentInputChannels * kernelHeight * kernelWidth, batchSize * outputHeight * outputWidth]],
                    )[0];

                    // Step 2: Slice and reshape weight
                    const weightSlice = slice(weight, [[startOutChannel, endOutChannel], [ic, ic + currentInputChannels], null, null]);
                    const weightReshaped = weightSlice.reshape([outChannelCount, currentInputChannels * kernelHeight * kernelWidth]);

                    // Step 3: GEMM - weight @ im2col^T (already transposed by kernel)
                    // weight: [C_out, C_in*K*K]
                    // im2col_transposed: [C_in*K*K, B*H*W]
                    // result: [C_out, B*H*W]
                    const partialResult = fastmm(weightReshaped, im2colOutput);

                    // Step 4: Reshape and permute
                    // [C_out, B*H*W] -> [C_out, B, H, W] -> [B, C_out, H, W]
                    const resultReshaped = partialResult.reshape([outChannelCount, batchSize, outputHeight, outputWidth]);
                    const result = permute(resultReshaped, [1, 0, 2, 3]);

                    // Accumulate
                    accumulated = accumulated === null ? result : accumulated.add(result);
                }

                return accumulated!;
            }
        );

        // Step 5: Add bias if provided
        const finalResult = bias
            ? splitChannelOperation(
                bias.shape[0],
                1, // Channel dimension in NCHW format
                (startChannel: number, endChannel: number) => {
                    const channelCount = endChannel - startChannel;

                    // Slice bias: [out_channels]
                    const biasSlice = slice(bias, [[startChannel, endChannel]]);

                    // Slice result along channel dimension: [batch, channels, H, W]
                    const resultSlice = slice(result, [null, [startChannel, endChannel], null, null]);

                    // Reshape bias for broadcasting: [1, channels, 1, 1]
                    const biasReshaped = biasSlice.reshape([1, channelCount, 1, 1]);

                    // Add bias to this channel batch
                    return resultSlice.add(biasReshaped);
        }
            )
            : result;

        return finalResult;
    }
}

export function conv_transpose2d(
    input: Tensor,
    weight: Tensor,
    bias?: Tensor,
    stride?: number | [number, number],
    padding?: number | [number, number]
): Tensor {
    if (shouldCreateGradient(input, weight)) {
        throw new Error("conv_transpose2d gradient not supported yet");
    } else {
        if (input.shape.length !== 4 || weight.shape.length !== 4) {
            throw new Error(
                `Expected 4D tensors, got input ${input.shape} and weight ${weight.shape}`
            );
        }
        // ConvTranspose2d weight shape: [in_channels, out_channels, kH, kW]
        // Check that input channels match
        if (input.shape[1] !== weight.shape[0]) {
            throw new Error(
                `Input channels (${input.shape[1]}) must match weight input channels (${weight.shape[0]})`
            );
        }

        // Parse stride parameter (default: 1)
        const strideH = typeof stride === 'number' ? stride : (stride?.[0] ?? 1);
        const strideW = typeof stride === 'number' ? stride : (stride?.[1] ?? 1);

        // Parse padding parameter (default: 0)
        let padH = 0;
        let padW = 0;
        if (typeof padding === 'number') {
            padH = padding;
            padW = padding;
        } else if (Array.isArray(padding)) {
            padH = padding[0];
            padW = padding[1];
        }

        // Calculate output dimensions using ConvTranspose2d formula
        // output_size = (input_size - 1) * stride + kernel_size - 2 * padding
        const outputHeight = (input.shape[2] - 1) * strideH + weight.shape[2] - 2 * padH;
        const outputWidth = (input.shape[3] - 1) * strideW + weight.shape[3] - 2 * padW;

        const params = {
            batchSize: input.shape[0],
            inputChannels: weight.shape[0],      // weight[0] = in_channels
            outputChannels: weight.shape[1],     // weight[1] = out_channels
            inputHeight: input.shape[2],
            inputWidth: input.shape[3],
            kernelHeight: weight.shape[2],
            kernelWidth: weight.shape[3],
            outputHeight: outputHeight,
            outputWidth: outputWidth,
            padH: padH,
            padW: padW,
            strideH: strideH,
            strideW: strideW,
        };

        // WORKAROUND: WebGPU 64-channel execution limit
        // Use splitChannelOperation for kernel and bias
        const result = splitChannelOperation(
            params.outputChannels,
            1, // Channel dimension in NCHW format
            (startChannel: number, endChannel: number) => {
                const channelCount = endChannel - startChannel;

                // Slice weight tensor: [in_channels, out_channels, kH, kW]
                const weightSlice = slice(weight, [null, [startChannel, endChannel], null, null]);

                // Create params for this channel batch
                const batchParams = { ...params, outputChannels: channelCount };

                // Run kernel for this channel batch
                return input.runKernel(
            "conv_transpose2d",
            { dtype: input.dtype },
                    batchParams,
                    [[batchParams.batchSize, channelCount, batchParams.outputHeight, batchParams.outputWidth]],
                    weightSlice
        )[0];
            }
        );

        // Add bias if provided
        const finalResult = bias
            ? splitChannelOperation(
                bias.shape[0],
                1, // Channel dimension in NCHW format
                (startChannel: number, endChannel: number) => {
                    const channelCount = endChannel - startChannel;

                    // Slice bias: [out_channels]
                    const biasSlice = slice(bias, [[startChannel, endChannel]]);

                    // Slice result along channel dimension
                    const resultSlice = slice(result, [null, [startChannel, endChannel], null, null]);

                    // Reshape bias for broadcasting: [1, channels, 1, 1]
                    const biasReshaped = biasSlice.reshape([1, channelCount, 1, 1]);

                    // Add bias to this channel batch
                    return resultSlice.add(biasReshaped);
        }
            )
            : result;

        return finalResult;
    }
}

function collapseView(a: Tensor, start: number, end: number): Tensor {
    const newShape = collapseViewHelper(a, start, end);
    if (newShape === null) {
        throw new Error("Attempting to view a collapsed tensor, but no such view exists!");
    }
    return a.withShape(newShape.shape, newShape.strides);
}

function collapse(a: Tensor, start: number, end: number): Tensor {
    // const newShape = collapsedShape(a, start, end);
    throw new Error("collapse not implemented yet");
}

/** Flattens a contiguous range of dims into a 1D tensor.
 *  
 * `flatten`, unlike other shape operators, returns the input tensor on a no-op
 * (unless a 0D tensor is flattened, in which case it's returned in 1D).
 * */
export function flatten(
    a: Tensor,
    startDim: number = 0,
    endDim: number = -1
): Tensor {
    startDim = canonicalizeDim(a.ndim, startDim);
    endDim = canonicalizeDim(a.ndim, endDim);
    // Short-circuits on no-op
    if (startDim == endDim && a.ndim != 0) {
        return a;
    }
    // Tries to take a view
    const newShape = collapseViewHelper(a, startDim, endDim);
    if (newShape !== null) {
        return collapseView(a, startDim, endDim);
    }
    return collapse(a, startDim, endDim);
}

/** Gathers values along an axis specified by dim. */
export function gather(input: Tensor, dim: number, index: Tensor): Tensor {
    if (shouldCreateGradient(input, index)) {
        return GatherFunction.apply(input, dim, index);
    }
    return GatherFunction.forward([input, dim, index]);
}

/**
 * Applies a linear transformation to the incoming data: `y = xA^T + b`.
 * @param input `(*, inFeatures)` where `*` means any number of additional dimensions, including none
 * @param weight `(outFeatures, inFeatures)` or `(inFeatures)`
 * @param bias `(outFeatures)` or `()`
 * @returns `(*, outFeatures)` or `(*)`, based on the shape of the weight
 */
export function linear(input: Tensor, weight: Tensor, bias?: Tensor): Tensor {
    if (shouldCreateGradient(input, weight, bias)) {
        return LinearFunction.apply(input, weight, bias);
    }
    return LinearFunction.forward([input, weight, bias]);
}

export function matmul(input: Tensor, other: Tensor): Tensor {
    const a: StridedShape = { shape: input.shape, strides: input.strides };
    const b: StridedShape = { shape: other.shape, strides: other.strides };
    const adims = a.shape.length;
    const bdims = b.shape.length;
    if (adims === 0 || bdims === 0) {
        throw new Error("matmul requires at least 1D tensors");
    }
    let atensor = input;
    let btensor = other;
    let op: string;
    let aop: StridedShape;
    let bop: StridedShape;
    let outputShape: Shape;
    // If both tensors are 1-dimensional, the dot product (scalar) is returned
    if (adims === 1 && bdims === 1) {
        if (a.shape[0] !== b.shape[0]) {
            throw new Error(
                `inconsistent tensor size, expected tensor [${a.shape}] and src [${b.shape}] to have the same number of elements, but got ${a.shape[0]} and ${b.shape[0]} elements respectively`
            );
        }
        return fastmm(input, other);

        op = "dot";
        aop = a;
        bop = b;
        outputShape = [];
    }
    // If both arguments are 2-dimensional, the matrix-matrix product is returned
    else if (adims === 2 && bdims === 2) {
        return fastmm(input, other);
        op = "fastmm";
        aop = a;
        bop = b;
        outputShape = [a.shape[0], b.shape[1]];
        if (aop.shape[1] !== b.shape[0]) {
            throw new Error(
                `mat1 and mat2 shapes cannot be multiplied (${a.shape[0]}x${a.shape[1]} and ${b.shape[0]}x${b.shape[1]})`
            );
        }
    }
    // If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension
    else if (adims === 1 && bdims === 2) {
        return fastmm(input, other);

        const aopshape = b.shape.slice();
        const aopstrides = b.strides.slice();
        aopshape[bdims - 1] = b.shape[bdims - 2];
        aopshape[bdims - 2] = b.shape[bdims - 1];
        aopstrides[bdims - 1] = b.strides[bdims - 2];
        aopstrides[bdims - 2] = b.strides[bdims - 1];
        op = "mv";
        aop = { shape: aopshape, strides: aopstrides };
        bop = a;
        outputShape = [b.shape[bdims - 1]];
        atensor = other;
        btensor = input;
        if (aop.shape[1] !== bop.shape[0]) {
            throw new Error(
                `mat1 and mat2 shapes cannot be multiplied (1x${bop.shape[0]} and ${aop.shape[1]}x${aop.shape[0]})`
            );
        }
    }
    // If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned
    else if (adims === 2 && bdims == 1) {
        return fastmm(input, other);


        op = "mv";
        aop = a;
        bop = b;
        outputShape = [a.shape[0]];
        if (aop.shape[1] !== bop.shape[0]) {
            throw new Error(
                `size mismatch, got ${aop.shape[0]}, ${aop.shape[0]}x${aop.shape[1]},${b.shape[0]}`
            );
        }
    } else if (adims >= 1 && bdims >= 1 && (adims > 2 || bdims > 2)) {
        op = "bmm";
        const broadcast = broadcastBatchedMatmul(input, other);
        aop = reshapeBatchedMatmul(broadcast.a);
        bop = reshapeBatchedMatmul(broadcast.b);
        outputShape = broadcast.output.shape;
        if (aop.shape[2] !== bop.shape[1]) {
            throw new Error(
                `mat1 and mat2 shapes cannot be multiplied (${aop.shape[1]}x${aop.shape[2]} and ${bop.shape[1]}x${bop.shape[2]})`
            );
        }
    } else {
        throw new Error(
            `matmul not supported for ${adims}D and ${bdims}D tensors`
        );
    }
    let params: KernelParamsInput = {};
    if (op === "bmm") {
        // console.log('BATCH MULT\n')
        params = {
            batchSize: Math.max(aop.shape[0], bop.shape[0]),
            aRows: aop.shape[1],
            aCols: aop.shape[2],
            bCols: bop.shape[2],
            aBatchStride: aop.strides[0],
            aRowStride: aop.strides[1],
            aColStride: aop.strides[2],
            bBatchStride: bop.strides[0],
            bRowStride: bop.strides[1],
            bColStride: bop.strides[2],
            alpha: 1.0,
        };
    } else if (op === "mm") {
        params = {
            aRows: aop.shape[0],
            aCols: aop.shape[1],
            bCols: bop.shape[1],
            aRowStride: aop.strides[0],
            aColStride: aop.strides[1],
            bRowStride: bop.strides[0],
            bColStride: bop.strides[1],
            alpha: 1.0,
        };
    } else if (op === "mv") {
        params = {
            aRows: aop.shape[0],
            aCols: aop.shape[1],
            aRowStride: aop.strides[0],
            aColStride: aop.strides[1],
            bRowStride: bop.strides[0],
        };
    } else if (op === "dot") {
        params = {
            aRows: aop.shape[0],
            aRowStride: aop.strides[0],
            bRowStride: bop.strides[0],
        };
    }
    return atensor.runKernel(
        op,
        { resultDtype: input.dtype },
        params,
        [outputShape],
        btensor
    )[0];
}

export function mm(input: Tensor, other: Tensor): Tensor {
    if (shouldCreateGradient(input, other)) {
        throw new Error("mm gradient not supported yet");
    } else {
    if (input.shape.length !== 2 || other.shape.length !== 2) {
        throw new Error(
            `Expected 2D tensors, got ${input.shape} and ${other.shape}`
        );
    }
    if (input.shape[1] !== other.shape[0]) {
        throw new Error(
            `Expected tensors inner dimensions to be compatible, got ${input.shape} and ${other.shape}`
        );
    }
    return fastmm(input, other);
    const params = {
        aRows: input.shape[0],
        aCols: input.shape[1],
        bCols: other.shape[1],
        aRowStride: input.strides[0],
        aColStride: input.strides[1],
        bRowStride: other.strides[0],
        bColStride: other.strides[1],
        alpha: 1.0,
    };
    // console.log('mm parameters:',params)
    return input.runKernel(
        "mm",
        { resultDtype: input.dtype },
        params,
        [[params.aRows, params.bCols]],
        other
    )[0];


    }
}
export function fastmm(input: Tensor, other: Tensor): Tensor {
    // console.log('fastmm func\n')
    if (shouldCreateGradient(input, other)) {
        throw new Error("mm gradient not supported yet");
    } else {
    if (input.shape.length !== 2 || other.shape.length !== 2) {
        throw new Error(
            `Expected 2D tensors, got ${input.shape} and ${other.shape}`
        );
    }
    if (input.shape[1] !== other.shape[0]) {
        throw new Error(
            `Expected tensors inner dimensions to be compatible, got ${input.shape} and ${other.shape}`
        );
    }
    const params = {
        aRows: input.shape[0],
        aCols: input.shape[1],
        bCols: other.shape[1],
        aRowStride: input.strides[0],
        aColStride: input.strides[1],
        bRowStride: other.strides[0],
        bColStride: other.strides[1],       
    };
    // console.log(input.strides)
    // console.log(`CYP: ${input.shape}, ${other.shape}, ${input.strides}, ${other.strides}`);
    return input.runKernel(
        "fastmm2",
        { resultDtype: input.dtype },
        params,
        [[params.aRows, params.bCols]],
        other
    )[0];
    }
}

export function numel(input: Tensor): number {
    return shapeSize(input.shape);
}

function inferSize(shape: Shape, numel: number): Shape {
    let dim = null;
    let newsize = 1;
    for (let i = 0; i < shape.length; i++) {
        let d = shape[i];
        if (d == -1) {
            check(dim === null, () => "only one dimension can be inferred");
            dim = i;
        } else if (d >= 0) {
            newsize *= d;
        } else {
            check(false, () => `invalid shape dimension ${d}`);
        }
    }
    check(
        numel == newsize ||
            (dim !== null && newsize > 0 && numel % newsize == 0),
        () => `shape '[${shape}]' is invalid for input of size ${numel}`
    );
    if (dim !== null) {
        check(
            newsize != 0,
            () => `cannot reshape tensor of 0 elements into shape ${shape} because the unspecified dimension size -1 can be any 
value and is ambiguous`
        );
        shape[dim] = Math.floor(numel / newsize);
    }
    return shape;
}

function splitDim(a: Tensor, dim: number, outerLength: number): Tensor {
    validateIdx(a.ndim, dim);
    validateDimLength(outerLength);
    const innerLength = Math.floor(a.shape[dim] / outerLength);
    if (a.shape[dim] % outerLength !== 0) {
        throw new Error(
            `Attempting to split dimension of length ${a.shape[dim]}, but out length of ${outerLength} divides it with a remainder!`
        );
    }
    const newShape: Shape = [];
    const newStrides: Strides = [];
    for (let idx = 0; idx < a.ndim; idx++) {
        if (idx === dim) {
            newShape.push(outerLength);
            newShape.push(innerLength);
            newStrides.push(a.strides[idx] * innerLength);
            newStrides.push(a.strides[idx]);
        } else {
            newShape.push(a.shape[idx]);
            newStrides.push(a.strides[idx]);
        }
    }
    return a.withShape(newShape, newStrides);
}
function validateCollapseArgs(a: Tensor, start: number, end: number): void {
    const ndim = Math.max(1, a.ndim);
    validateIdx(ndim, start);
    validateIdx(ndim, end);
    check(
        end >= start,
        () =>
            `Attempting to collapse but end, ${end}, is less than start, ${start}`
    );
}
function collapseViewHelper(
    a: Tensor,
    start: number,
    end: number
): StridedShape | null {
    validateCollapseArgs(a, start, end);
    let shape: Shape;
    let strides: Strides;
    if (a.ndim === 0) {
        shape = [1];
        strides = [1];
    } else {
        shape = a.shape;
        strides = a.strides;
    }

    if (a.ndim === 0 || start === end) {
        return { shape, strides };
    }
    let length = shape[end];
    let stride = strides[end];
    for (let idx = end - 1; idx >= start; idx--) {
        if (shape[idx] === 0 || shape[idx + 1] === 0) {
            length = 0;
            stride = 0;
            break;
        }
        if (shape[idx] === 1) {
            continue;
        }
        length *= shape[idx];
        stride = Math.min(stride, strides[idx]);
        if (
            a.numel() > 0 &&
            shape[idx + 1] != 1 &&
            !(strides[idx] === strides[idx + 1] * shape[idx + 1])
        ) {
            return null;
        }
    }
    const newShape = shape
        .slice(0, start)
        .concat([length])
        .concat(shape.slice(end + 1));
    let newStrides = strides
        .slice(0, start)
        .concat([stride])
        .concat(strides.slice(end + 1));
    if (a.numel() === 0) {
        newStrides = defaultStrides(newShape);
    }
    return { shape: newShape, strides: newStrides };
}

function primitiveReshape(a: Tensor, shape: Shape): Tensor {
    throw new Error("Copying reshape not implemented");
}

function reshapeViewHelper(
    a: Tensor,
    shapeInput: Shape,
    allowCopy: boolean = false
): Tensor {
    const shape = inferSize(shapeInput, a.numel());

    // Short-circuits if shape is the same
    if (shapesAreEqual(a.shape, shape)) {
        return a.withShape(a.shape, a.strides);
    }

    // Special-cases tensors with no elements
    if (a.numel() === 0) {
        return a.withShape(shape, defaultStrides(shape));
    }

    // Special-cases reshaping zero dim tensors
    if (a.ndim === 0) {
        let _a = a;
        for (let length of shape) {
            if (length !== 1) {
                throw new Error("Expected length to be 1.");
            }
            _a = squeeze(_a, -1);
        }
        return _a;
    }

    // Special-cases reshaping to zero dim tensors
    if (shape.length === 0) {
        let _a = a;
        for (let length of a.shape) {
            if (length !== 1) {
                throw new Error("Expected length to be 1.");
            }
            _a = squeeze(_a, -1);
        }
        return _a;
    }

    // Handles general case: a 1+D tensor reshaped into a distinct 1+D shape
    let idx = 0;
    let a_ = a;
    for (let length of shape) {
        // Handles tail unsqueezes
        if (idx >= a_.ndim) {
            if (length !== 1) {
                throw new Error("Expected length to be 1.");
            }
            let lastDim = a_.ndim - 1;
            a_ = splitDim(a_, lastDim, a_.shape[lastDim]);
            idx++;
            continue;
        }

        // Skips dimensions that are already the correct length
        if (length === a_.shape[idx]) {
            idx++;
            continue;
        }
        // Gathers enough original dimensions such that this new dimension can be created
        // Note that this accumulation will terminate because we've verified a and the shape
        // specify the same number of elements above
        let accum = a_.shape[idx];
        let end = idx;
        while (accum % length !== 0) {
            end++;
            accum *= a_.shape[end];
        }
        if (end !== idx) {
            let newShapeStrides = collapseViewHelper(a_, idx, end);
            if (newShapeStrides === null) {
                if (allowCopy) {
                    return primitiveReshape(a, shape);
                }
                throw new Error(
                    `Cannot view a tensor with shape ${a.shape} and strides ${a.strides} as a tensor with shape ${shape}!`
                );
            }
            a_ = flatten(a_, idx, end);
        }
        if (accum !== length) {
            a_ = splitDim(a_, idx, length);
        }
        idx++;
    }
    while (idx < a_.ndim) {
        if (a_.shape[idx] !== 1) {
            throw new Error("Expected shape at index " + idx + " to be 1.");
        }
        a_ = squeeze(a_, idx);
    }
    return a_;
}

export function reshape(input: Tensor, shape: number[]): Tensor {
    return reshapeViewHelper(input, shape, true);
}

export function squeeze(input: Tensor, dim?: number | number[]): Tensor {
    let dims: number[];
    if (dim === undefined) {
        dims = [];
        for (let i = 0; i < input.shape.length; i++) {
            if (input.shape[i] === 1) {
                dims.push(i);
            }
        }
    } else if (typeof dim === "number") {
        dims = [dim];
    } else {
        dims = dim;
    }
    const inputRank = input.shape.length;
    const minDim = inputRank > 0 ? -inputRank : -1;
    const maxDim = inputRank > 0 ? inputRank - 1 : 0;
    for (let i in dims) {
        let d = dims[i];
        if (d < minDim || d > maxDim) {
            throw new Error(
                `Dimension out of range (expected to be in range of [${minDim}, ${maxDim}], but got ${d})`
            );
        }
        if (d < 0) {
            dims[i] = input.shape.length + d;
        }
    }
    dims.sort();
    const outputShape: Shape = [];
    const outputStrides: number[] = [];
    let j = 0;
    for (let i = 0; i < inputRank; i++) {
        if (j < dims.length && i === dims[j]) {
            if (input.shape[i] !== 1) {
                outputShape.push(input.shape[i]);
                outputStrides.push(input.strides[i]);
            }
            j++;
        } else {
            outputShape.push(input.shape[i]);
            outputStrides.push(input.strides[i]);
        }
    }
    return input.withShape(outputShape, outputStrides);
}

export function t(input: Tensor): Tensor {
    if (input.shape.length !== 2) {
        throw new Error(`Expected 2D tensor, got ${input.shape}`);
    }
    if (shouldCreateGradient(input)) {
        throw new Error("t gradient not supported yet");
        // return TransposeFunction.apply(input, 0, 1);
    } else {
    let newShape = input.shape.slice();
    newShape.reverse();
    let newStrides = input.strides.slice();
    newStrides.reverse();
    return input.withShape(newShape, newStrides);
    }
}

export function tensor(spec: TensorSpec): Tensor;
export function tensor(
    array: TensorData,
    dtype?: Dtype,
    device?: Deviceish,
    requiresGrad?: boolean
): Tensor;
export function tensor(
    arrayOrSpec: TensorData | TensorSpec,
    dtype?: Dtype,
    device?: Deviceish,
    requiresGrad?: boolean
): Tensor {
    if (arrayOrSpec.hasOwnProperty("data")) {
        return new Tensor(arrayOrSpec as TensorSpec);
    }
    return new Tensor(arrayOrSpec as TensorData, dtype, device, requiresGrad);
}

export function unsqueeze(input: Tensor, dim?: number): Tensor {
    const inputRank = input.shape.length;
    const minDim = inputRank > 0 ? -inputRank - 1 : -2;
    const maxDim = inputRank > 0 ? inputRank + 1 : 2;
    let unsqueezeOutputDim: number;
    if (dim === undefined) {
        unsqueezeOutputDim = 0;
    } else if (dim < minDim || dim >= maxDim) {
        throw new Error(
            `Dimension out of range (expected to be in range of [${minDim}, ${
                maxDim - 1
            }], but got ${dim})`
        );
    } else if (dim < 0) {
        unsqueezeOutputDim = dim + inputRank + 1;
    } else {
        unsqueezeOutputDim = dim;
    }
    const outputShape: Shape = [];
    const outputStrides: number[] = [];
    let inputDim = 0;
    for (let outputDim = 0; outputDim < inputRank + 1; outputDim++) {
        if (outputDim === unsqueezeOutputDim) {
            outputShape.push(1);
            if (outputDim === 0) {
                outputStrides.push(input.strides[0] * input.shape[0]);
            } else {
                outputStrides.push(outputStrides[outputDim - 1]);
            }
        } else {
            outputShape.push(input.shape[inputDim]);
            outputStrides.push(input.strides[inputDim]);
            inputDim++;
        }
    }
    return input.withShape(outputShape, outputStrides);
}

export function view(input: Tensor, shape: number[]): Tensor {
    return reshapeViewHelper(input, shape, false);
}
