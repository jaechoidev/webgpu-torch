import { KernelSpec } from "./kernel";

export const kernels: { [name: string]: KernelSpec } = {
    conv2d: {
        name: "conv2d",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: [
            {
                name: "batchSize",
                shaderType: "u32",
            },
            {
                name: "inputChannels",
                shaderType: "u32",
            },
            {
                name: "outputChannels",
                shaderType: "u32",
            },
            {
                name: "inputHeight",
                shaderType: "u32",
            },
            {
                name: "inputWidth",
                shaderType: "u32",
            },
            {
                name: "kernelHeight",
                shaderType: "u32",
            },
            {
                name: "kernelWidth",
                shaderType: "u32",
            },
            {
                name: "outputHeight",
                shaderType: "u32",
            },
            {
                name: "outputWidth",
                shaderType: "u32",
            },
            {
                name: "padH",
                shaderType: "u32",
            },
            {
                name: "padW",
                shaderType: "u32",
            },
            {
                name: "strideH",
                shaderType: "u32",
            },
            {
                name: "strideW",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>",
            },
            {
                name: "weight",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "batchSize * outputChannels * outputHeight * outputWidth",
            },
        ],
        workgroupSize: [4, 4, 1],
        workgroupCount: ["(outputWidth + 3) / 4", "(outputHeight + 3) / 4", 1],
        shader: `
    // Standard conv2d: output[b,c_out,y,x] = sum over c_in,ky,kx of input[b,c_in,y*stride+ky-pad,x*stride+kx-pad] * weight[c_out,c_in,ky,kx]
    //
    // NOTE: This kernel processes ≤64 channels at a time.
    // The JavaScript layer (ops_artisanal.ts::splitChannelOperation) splits larger channel counts into 64-channel batches.
    // This avoids the WebGPU/driver bug where loops >64 iterations may not execute correctly.

    if (global_id.x >= parameters.outputWidth || global_id.y >= parameters.outputHeight) {
        return;
    }

    for (var batch = 0u; batch < parameters.batchSize; batch++) {
        for (var outputChannel = 0u; outputChannel < parameters.outputChannels; outputChannel++) {
            var result = 0.0;

            // Perform convolution
            for (var inputChannel = 0u; inputChannel < parameters.inputChannels; inputChannel++) {
                for (var kernelY = 0u; kernelY < parameters.kernelHeight; kernelY++) {
                    for (var kernelX = 0u; kernelX < parameters.kernelWidth; kernelX++) {
                        // Calculate input coordinates with stride and padding
                        let inputYWithPad = i32(global_id.y * parameters.strideH + kernelY);
                        let inputXWithPad = i32(global_id.x * parameters.strideW + kernelX);

                        let inputY = inputYWithPad - i32(parameters.padH);
                        let inputX = inputXWithPad - i32(parameters.padW);

                        // Check bounds (padding is zero)
                        if (inputY < 0 || inputY >= i32(parameters.inputHeight) ||
                            inputX < 0 || inputX >= i32(parameters.inputWidth)) {
                            continue;
                        }

                        let inputYUnsigned = u32(inputY);
                        let inputXUnsigned = u32(inputX);

                        // Calculate indices
                        var inputIndex =
                            batch * parameters.inputChannels * parameters.inputHeight * parameters.inputWidth +
                            inputChannel * parameters.inputHeight * parameters.inputWidth +
                            inputYUnsigned * parameters.inputWidth +
                            inputXUnsigned;
                        var kernelIndex =
                            outputChannel * parameters.inputChannels * parameters.kernelHeight * parameters.kernelWidth +
                            inputChannel * parameters.kernelHeight * parameters.kernelWidth +
                            kernelY * parameters.kernelWidth +
                            kernelX;

                        result = result + input[inputIndex] * weight[kernelIndex];
                    }
                }
            }

            // Write output
            let outputIndex =
                batch * parameters.outputChannels * parameters.outputHeight * parameters.outputWidth +
                outputChannel * parameters.outputHeight * parameters.outputWidth +
                global_id.y * parameters.outputWidth +
                global_id.x;
            output[outputIndex] = result;
        }
    }
`
    },
    conv_transpose2d: {
        name: "conv_transpose2d",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: [
            {
                name: "batchSize",
                shaderType: "u32",
            },
            {
                name: "inputChannels",
                shaderType: "u32",
            },
            {
                name: "outputChannels",
                shaderType: "u32",
            },
            {
                name: "inputHeight",
                shaderType: "u32",
            },
            {
                name: "inputWidth",
                shaderType: "u32",
            },
            {
                name: "kernelHeight",
                shaderType: "u32",
            },
            {
                name: "kernelWidth",
                shaderType: "u32",
            },
            {
                name: "outputHeight",
                shaderType: "u32",
            },
            {
                name: "outputWidth",
                shaderType: "u32",
            },
            {
                name: "padH",
                shaderType: "u32",
            },
            {
                name: "padW",
                shaderType: "u32",
            },
            {
                name: "strideH",
                shaderType: "u32",
            },
            {
                name: "strideW",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>",
            },
            {
                name: "weight",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "batchSize * outputChannels * outputHeight * outputWidth",
            },
        ],
        workgroupSize: [16, 16, 1],
        workgroupCount: [
            "outputWidth/16",
            "outputHeight/16",
            "batchSize * outputChannels",
        ],
        shader: `
    if (global_id.x >= parameters.outputWidth || global_id.y >= parameters.outputHeight) {
        return;
    }

    // global_id.z encodes both batch and output channel
    let batch = global_id.z / parameters.outputChannels;
    let outputChannel = global_id.z % parameters.outputChannels;

    var result = 0.0;

    // ConvTranspose2d: For each output position (out_h, out_w),
    // accumulate contributions from all input positions that can reach it.
    //
    // Formula: output[out_h, out_w] = sum over all (in_h, in_w, k_h, k_w) where:
    //   out_h = in_h * stride_h + k_h - pad_h
    //   out_w = in_w * stride_w + k_w - pad_w
    //
    // Rearranged: For a given (out_h, out_w), find all valid (in_h, in_w, k_h, k_w)
    //   in_h = (out_h + pad_h - k_h) / stride_h  (must be integer and in range)
    //   in_w = (out_w + pad_w - k_w) / stride_w  (must be integer and in range)

    let out_h = global_id.y;
    let out_w = global_id.x;

    // Iterate over all kernel positions
    for (var k_h = 0u; k_h < parameters.kernelHeight; k_h++) {
        for (var k_w = 0u; k_w < parameters.kernelWidth; k_w++) {
            // Calculate which input position contributes through this kernel position
            // out_h = in_h * stride_h + k_h - pad_h
            // => in_h = (out_h + pad_h - k_h) / stride_h

            // Check if subtraction would underflow (k_h > out_h + padH)
            // WGSL u32 subtraction wraps on underflow, so we must check first
            if (k_h > out_h + parameters.padH || k_w > out_w + parameters.padW) {
                continue;
            }

            let numerator_h = out_h + parameters.padH - k_h;
            let numerator_w = out_w + parameters.padW - k_w;

            // Check if division is exact (integer) and result is in valid range
            if (numerator_h % parameters.strideH == 0u && numerator_w % parameters.strideW == 0u) {
                let in_h = numerator_h / parameters.strideH;
                let in_w = numerator_w / parameters.strideW;

                // Check if input position is valid
                if (in_h < parameters.inputHeight && in_w < parameters.inputWidth) {
                    // Accumulate contribution from all input channels
                    for (var inputChannel = 0u; inputChannel < parameters.inputChannels; inputChannel++) {
                        // Input index: [batch, inputChannel, in_h, in_w]
                        let inputIndex =
                            batch * parameters.inputChannels * parameters.inputHeight * parameters.inputWidth +
                            inputChannel * parameters.inputHeight * parameters.inputWidth +
                            in_h * parameters.inputWidth +
                            in_w;

                        // Weight shape for ConvTranspose2d: [inputChannels, outputChannels, kernelHeight, kernelWidth]
                        // This is DIFFERENT from Conv2d which has [outputChannels, inputChannels, ...]
                        let weightIndex =
                            inputChannel * parameters.outputChannels * parameters.kernelHeight * parameters.kernelWidth +
                            outputChannel * parameters.kernelHeight * parameters.kernelWidth +
                            k_h * parameters.kernelWidth +
                            k_w;

                        result = result + input[inputIndex] * weight[weightIndex];
                    }
                }
            }
        }
    }

    // Output index: [batch, outputChannel, out_h, out_w]
    let outputIndex =
        batch * parameters.outputChannels * parameters.outputHeight * parameters.outputWidth +
        outputChannel * parameters.outputHeight * parameters.outputWidth +
        out_h * parameters.outputWidth +
        out_w;

    output[outputIndex] = result;
`
    },
    dot: {
        name: "dot",
        config: [
            {
                name: "resultDtype",
            },
        ],
        parameters: [
            {
                name: "aRows",
                shaderType: "u32",
            },
            {
                name: "aRowStride",
                shaderType: "u32",
            },
            {
                name: "bRowStride",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>",
            },
            {
                name: "b",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "1",
            },
        ],
        workgroupSize: [1, 1, 1],
        workgroupCount: [1, 1, 1],
        shader: `
    var result = 0.0;
    var aIndex = 0u;
    var bIndex = 0u;
    for (var aRow = 0u; aRow < parameters.aRows; aRow = aRow + 1u) {
        result = result + a[aIndex] * b[bIndex];
        aIndex = aIndex + parameters.aRowStride;
        bIndex = bIndex + parameters.bRowStride;
    }
    output[0] = result;
`
    },
    mv: {
        name: "mv",
        config: [
            {
                name: "resultDtype",
            },
        ],
        parameters: [
            {
                name: "aRows",
                shaderType: "u32",
            },
            {
                name: "aCols",
                shaderType: "u32",
            },
            {
                name: "aRowStride",
                shaderType: "u32",
            },
            {
                name: "aColStride",
                shaderType: "u32",
            },
            {
                name: "bRowStride",
                shaderType: "u32",
            },
       ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>",
            },
            {
                name: "b",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "aRows",
            },
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["aRows/256", 1, 1],
        shader: `
    let outputRow = global_id.x;
    if (outputRow >= parameters.aRows) {
        return;
    }
    var result = 0.0;
    var aIndex = outputRow * parameters.aRowStride;
    var bIndex = 0u;
    for (var aCol = 0u; aCol < parameters.aCols; aCol = aCol + 1u) {
        result = result + a[aIndex] * b[bIndex];
        aIndex = aIndex + parameters.aColStride;
        bIndex = bIndex + parameters.bRowStride;
    }
    output[outputRow] = result;
`
    },
    mm: {
        name: "mm",
        config: [
            {
                name: "resultDtype",
            },
        ],
        parameters: [
            {
                name: "aRows",
                shaderType: "u32",
            },
            {
                name: "aCols",
                shaderType: "u32",
            },
            {
                name: "bCols",
                shaderType: "u32",
            },
            {
                name: "aRowStride",
                shaderType: "u32",
            },
            {
                name: "aColStride",
                shaderType: "u32",
            },
            {
                name: "bRowStride",
                shaderType: "u32",
            },
            {
                name: "bColStride",
                shaderType: "u32",
            },
            {
                name: "alpha",
                shaderType: "f32",
            },
        ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>",
            },
            {
                name: "b",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "aRows * bCols",
            },
        ],
        workgroupSize: [16, 16, 1],
        workgroupCount: ["aRows/16", "bCols/16", 1],
        shader: `
    let outputRow = global_id.x;
    let outputCol = global_id.y;
    if (outputRow >= parameters.aRows || outputCol >= parameters.bCols) {
        return;
    }
    var result = 0.0;
    var aIndex = outputRow * parameters.aRowStride;
    var bIndex = outputCol * parameters.bColStride;
    for (var aCol = 0u; aCol < parameters.aCols; aCol = aCol + 1u) {
        result = result + a[aIndex] * b[bIndex];
        aIndex = aIndex + parameters.aColStride;
        bIndex = bIndex + parameters.bRowStride;
    }
    let outputIndex = outputCol + outputRow * parameters.bCols;
    output[outputIndex] = result;
`
    },
    bmm: {
        name: "bmm",
        config: [
            {
                name: "resultDtype",
            },
        ],
        parameters: [
            {
                name: "batchSize",
                shaderType: "u32",
            },
            {
                name: "aRows",
                shaderType: "u32",
            },
            {
                name: "aCols",
                shaderType: "u32",
            },
            {
                name: "bCols",
                shaderType: "u32",
            },
            {
                name: "aBatchStride",
                shaderType: "u32",
            },
            {
                name: "aRowStride",
                shaderType: "u32",
            },
            {
                name: "aColStride",
                shaderType: "u32",
            },
            {
                name: "bBatchStride",
                shaderType: "u32",
            },
            {
                name: "bRowStride",
                shaderType: "u32",
            },
            {
                name: "bColStride",
                shaderType: "u32",
            },
            {
                name: "alpha",
                shaderType: "f32",
            },
        ],
        inputs: [
            {
                name: "a",
                shaderType: "array<f32>",
            },
            {
                name: "b",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "batchSize * aRows * bCols",
            },
        ],
        workgroupSize: [8, 8, 4],
        workgroupCount: ["aRows/8", "bCols/8", "batchSize/4"],
        shader: `
    let outputRow = global_id.x;
    let outputCol = global_id.y;
    let outputBatch = global_id.z;
    if (outputRow >= parameters.aRows || outputCol >= parameters.bCols || outputBatch >= parameters.batchSize) {
        return;
    }
    var result = 0.0;
    var aIndex = outputBatch * parameters.aBatchStride + outputRow * parameters.aRowStride;
    var bIndex = outputBatch * parameters.bBatchStride + outputCol * parameters.bColStride;
    for (var aCol = 0u; aCol < parameters.aCols; aCol = aCol + 1u) {
        result = result + a[aIndex] * b[bIndex];
        aIndex = aIndex + parameters.aColStride;
        bIndex = bIndex + parameters.bRowStride;
    }
    let outputRowStride = parameters.bCols;
    let outputBatchStride = parameters.aRows * outputRowStride;
    let outputIndex = outputBatch * outputBatchStride + outputRow * outputRowStride + outputCol;
    output[outputIndex] = result;
`
    },
    sumDim: {
        name: "sumDim",
        config: [
            {
                name: "dtype",
            },
            {
                name: "workgroupSize",
            },
        ],
        parameters: [
            {
                name: "size",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "dimToSum",
                shaderType: "u32",
            },
            {
                name: "inputShape",
                shaderType: "vec3<u32>",
            },
            {
                name: "inputStrides",
                shaderType: "vec3<u32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: ["workgroupSize", "workgroupSize", "workgroupSize"],
        workgroupCount: [1, 1, 1],
        shader: `
        // Global index flattening for the reformatted 3D tensor
        var flatGlobalId: u32 = global_id.x * parameters.inputStrides.x + global_id.y * parameters.inputStrides.y + global_id.z * parameters.inputStrides.z;
    
        // Initialize sum
        var sum: f32 = 0.0;
    
        let numReductions: u32 = parameters.inputShape.y;
    
        // Sum reduction
        for (var i: u32 = 0; i < numReductions; i = i + 1) {
            // Compute the input index by adding the reduction offset to the current flat global index
            var dataIndex: u32 = flatGlobalId + i * parameters.inputStrides.y;
    
            if (dataIndex < input.length()) {
                // Accumulate the input value into sum
                sum = sum + input[dataIndex];
            }
        }
    
        // Write the reduced sum value to output tensor
        if (flatGlobalId < output.length()) {
            output[flatGlobalId] = sum;
        }
    `
    },
    uniform_: {
        name: "uniform_",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: [
            {
                name: "size",
                shaderType: "u32",
            },
            {
                name: "seed",
                shaderType: "u32",
            },
            {
                name: "lowerBound",
                shaderType: "f32",
            },
            {
                name: "upperBound",
                shaderType: "f32",
            },
        ],
        inputs: [],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "size",
            }
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["size/256", 1, 1],
        shader: `
        let outputIndex = global_id.x;
        if (outputIndex >= parameters.size) {
            return;
        }

        var b = 0u;
        let seed = u32(u32(parameters.seed + outputIndex) * 1099087573u);
        b = ((seed << 13) ^ seed) >> 19;
        let z1 = ((seed & 429496729u) << 12) ^ b;
        b = ((seed << 2) ^ seed) >> 25;
        let z2 = ((seed & 4294967288u) << 4) ^ b;
        b = ((seed << 3) ^ seed) >> 11;
        let z3 = ((seed & 429496280u) << 17) ^ b;
        let z4 = u32(u32(1664525u * seed) + 1013904223u);
        let r = z1 ^ z2 ^ z3 ^ z4;
        let u = f32(u32(r)) * 2.3283064365387e-10;

        output[outputIndex] = parameters.lowerBound + u * (parameters.upperBound - parameters.lowerBound);
    `
    },
    gather: {
        name: "gather",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: [
            {
                name: "dim",
                shaderType: "u32",
            },
            {
                name: "outputSize",
                shaderType: "u32",
            },
            {
                name: "rank",
                shaderType: "u32",
            },
            {
                name: "inputShape0",
                shaderType: "u32",
            },
            {
                name: "inputShape1",
                shaderType: "u32",
            },
            {
                name: "inputShape2",
                shaderType: "u32",
            },
            {
                name: "inputShape3",
                shaderType: "u32",
            },
            {
                name: "inputShape4",
                shaderType: "u32",
            },
            {
                name: "inputStride0",
                shaderType: "u32",
            },
            {
                name: "inputStride1",
                shaderType: "u32",
            },
            {
                name: "inputStride2",
                shaderType: "u32",
            },
            {
                name: "inputStride3",
                shaderType: "u32",
            },
            {
                name: "inputStride4",
                shaderType: "u32",
            },
            {
                name: "indexStride0",
                shaderType: "u32",
            },
            {
                name: "indexStride1",
                shaderType: "u32",
            },
            {
                name: "indexStride2",
                shaderType: "u32",
            },
            {
                name: "indexStride3",
                shaderType: "u32",
            },
            {
                name: "indexStride4",
                shaderType: "u32",
            },
            {
                name: "outputShape0",
                shaderType: "u32",
            },
            {
                name: "outputShape1",
                shaderType: "u32",
            },
            {
                name: "outputShape2",
                shaderType: "u32",
            },
            {
                name: "outputShape3",
                shaderType: "u32",
            },
            {
                name: "outputShape4",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>",
            },
            {
                name: "index",
                shaderType: "array<f32>",  // FIXED: was i32, but torch.tensor() creates f32 by default
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize",
            },
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["65535", "((outputSize + 255) / 256 + 65534) / 65535", 1],
        shader: `
    let flat_out = global_id.x + global_id.y * 65535u * 256u;
    if (flat_out >= parameters.outputSize) {
        return;
    }

    let rank = parameters.rank;
    let dim = parameters.dim;

    // Unroll the algorithm for up to 5D
    // We can't use dynamic array indexing in WGSL, so we unroll manually

    // Step 1: Convert flat output index to coordinates
    var coord0 = 0u;
    var coord1 = 0u;
    var coord2 = 0u;
    var coord3 = 0u;
    var coord4 = 0u;

    if (rank == 1u) {
        coord0 = flat_out;
    } else if (rank == 2u) {
        let s1 = parameters.outputShape1;
        coord0 = flat_out / s1;
        coord1 = flat_out % s1;
    } else if (rank == 3u) {
        let s1 = parameters.outputShape1;
        let s2 = parameters.outputShape2;
        let s12 = s1 * s2;
        coord0 = flat_out / s12;
        let rem = flat_out % s12;
        coord1 = rem / s2;
        coord2 = rem % s2;
    } else if (rank == 4u) {
        let s1 = parameters.outputShape1;
        let s2 = parameters.outputShape2;
        let s3 = parameters.outputShape3;
        let s123 = s1 * s2 * s3;
        let s23 = s2 * s3;
        coord0 = flat_out / s123;
        var rem = flat_out % s123;
        coord1 = rem / s23;
        rem = rem % s23;
        coord2 = rem / s3;
        coord3 = rem % s3;
    } else if (rank == 5u) {
        let s1 = parameters.outputShape1;
        let s2 = parameters.outputShape2;
        let s3 = parameters.outputShape3;
        let s4 = parameters.outputShape4;
        let s1234 = s1 * s2 * s3 * s4;
        let s234 = s2 * s3 * s4;
        let s34 = s3 * s4;
        coord0 = flat_out / s1234;
        var rem = flat_out % s1234;
        coord1 = rem / s234;
        rem = rem % s234;
        coord2 = rem / s34;
        rem = rem % s34;
        coord3 = rem / s4;
        coord4 = rem % s4;
    }

    // Step 2: Get index value from index tensor
    var flat_index = 0u;
    if (rank >= 1u) { flat_index = flat_index + coord0 * parameters.indexStride0; }
    if (rank >= 2u) { flat_index = flat_index + coord1 * parameters.indexStride1; }
    if (rank >= 3u) { flat_index = flat_index + coord2 * parameters.indexStride2; }
    if (rank >= 4u) { flat_index = flat_index + coord3 * parameters.indexStride3; }
    if (rank >= 5u) { flat_index = flat_index + coord4 * parameters.indexStride4; }

    let idx_value = u32(index[flat_index]);

    // Step 3: Build input coordinates (replace dim coordinate with idx_value)
    var in0 = coord0;
    var in1 = coord1;
    var in2 = coord2;
    var in3 = coord3;
    var in4 = coord4;

    if (dim == 0u) { in0 = idx_value; }
    else if (dim == 1u) { in1 = idx_value; }
    else if (dim == 2u) { in2 = idx_value; }
    else if (dim == 3u) { in3 = idx_value; }
    else if (dim == 4u) { in4 = idx_value; }

    // Step 4: Convert input coordinates to flat index
    var flat_input = 0u;
    if (rank >= 1u) { flat_input = flat_input + in0 * parameters.inputStride0; }
    if (rank >= 2u) { flat_input = flat_input + in1 * parameters.inputStride1; }
    if (rank >= 3u) { flat_input = flat_input + in2 * parameters.inputStride2; }
    if (rank >= 4u) { flat_input = flat_input + in3 * parameters.inputStride3; }
    if (rank >= 5u) { flat_input = flat_input + in4 * parameters.inputStride4; }

    // Step 5: Perform gather
    output[flat_out] = input[flat_input];
`
    },
    cat: {
        name: "cat",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: [
            {
                name: "rank",
                shaderType: "u32",
            },
            {
                name: "dim",
                shaderType: "u32",
            },
            {
                name: "outputSize",
                shaderType: "u32",
            },
            {
                name: "outputShape0",
                shaderType: "u32",
            },
            {
                name: "outputShape1",
                shaderType: "u32",
            },
            {
                name: "outputShape2",
                shaderType: "u32",
            },
            {
                name: "outputShape3",
                shaderType: "u32",
            },
            {
                name: "outputShape4",
                shaderType: "u32",
            },
            {
                name: "outputShape5",
                shaderType: "u32",
            },
            {
                name: "numInputs",
                shaderType: "u32",
            },
            // For each input: shape[dim] and strides[0..5]
            // We'll support up to 8 inputs, each with 7 parameters (dimSize + 6 strides)
            {
                name: "input0DimSize",
                shaderType: "u32",
            },
            {
                name: "input0Stride0",
                shaderType: "u32",
            },
            {
                name: "input0Stride1",
                shaderType: "u32",
            },
            {
                name: "input0Stride2",
                shaderType: "u32",
            },
            {
                name: "input0Stride3",
                shaderType: "u32",
            },
            {
                name: "input0Stride4",
                shaderType: "u32",
            },
            {
                name: "input0Stride5",
                shaderType: "u32",
            },
            {
                name: "input1DimSize",
                shaderType: "u32",
            },
            {
                name: "input1Stride0",
                shaderType: "u32",
            },
            {
                name: "input1Stride1",
                shaderType: "u32",
            },
            {
                name: "input1Stride2",
                shaderType: "u32",
            },
            {
                name: "input1Stride3",
                shaderType: "u32",
            },
            {
                name: "input1Stride4",
                shaderType: "u32",
            },
            {
                name: "input1Stride5",
                shaderType: "u32",
            },
            {
                name: "input2DimSize",
                shaderType: "u32",
            },
            {
                name: "input2Stride0",
                shaderType: "u32",
            },
            {
                name: "input2Stride1",
                shaderType: "u32",
            },
            {
                name: "input2Stride2",
                shaderType: "u32",
            },
            {
                name: "input2Stride3",
                shaderType: "u32",
            },
            {
                name: "input2Stride4",
                shaderType: "u32",
            },
            {
                name: "input2Stride5",
                shaderType: "u32",
            },
        ],
        inputs: [
            {
                name: "input0",
                shaderType: "array<f32>",
            },
            {
                name: "input1",
                shaderType: "array<f32>",
            },
            {
                name: "input2",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "outputSize",
            },
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["65535", "((outputSize + 255) / 256 + 65534) / 65535", 1],
        shader: `
    let flat_out = global_id.x + global_id.y * 65535u * 256u;
    if (flat_out >= parameters.outputSize) {
        return;
    }

    let rank = parameters.rank;
    let concat_dim = parameters.dim;

    // Step 1: Convert flat output index to multi-dimensional coordinates
    var coord0 = 0u;
    var coord1 = 0u;
    var coord2 = 0u;
    var coord3 = 0u;
    var coord4 = 0u;
    var coord5 = 0u;

    if (rank == 1u) {
        coord0 = flat_out;
    } else if (rank == 2u) {
        let s1 = parameters.outputShape1;
        coord0 = flat_out / s1;
        coord1 = flat_out % s1;
    } else if (rank == 3u) {
        let s1 = parameters.outputShape1;
        let s2 = parameters.outputShape2;
        let s12 = s1 * s2;
        coord0 = flat_out / s12;
        let rem = flat_out % s12;
        coord1 = rem / s2;
        coord2 = rem % s2;
    } else if (rank == 4u) {
        let s1 = parameters.outputShape1;
        let s2 = parameters.outputShape2;
        let s3 = parameters.outputShape3;
        let s123 = s1 * s2 * s3;
        let s23 = s2 * s3;
        coord0 = flat_out / s123;
        var rem = flat_out % s123;
        coord1 = rem / s23;
        rem = rem % s23;
        coord2 = rem / s3;
        coord3 = rem % s3;
    } else if (rank == 5u) {
        let s1 = parameters.outputShape1;
        let s2 = parameters.outputShape2;
        let s3 = parameters.outputShape3;
        let s4 = parameters.outputShape4;
        let s1234 = s1 * s2 * s3 * s4;
        let s234 = s2 * s3 * s4;
        let s34 = s3 * s4;
        coord0 = flat_out / s1234;
        var rem = flat_out % s1234;
        coord1 = rem / s234;
        rem = rem % s234;
        coord2 = rem / s34;
        rem = rem % s34;
        coord3 = rem / s4;
        coord4 = rem % s4;
    } else if (rank == 6u) {
        let s1 = parameters.outputShape1;
        let s2 = parameters.outputShape2;
        let s3 = parameters.outputShape3;
        let s4 = parameters.outputShape4;
        let s5 = parameters.outputShape5;
        let s12345 = s1 * s2 * s3 * s4 * s5;
        let s2345 = s2 * s3 * s4 * s5;
        let s345 = s3 * s4 * s5;
        let s45 = s4 * s5;
        coord0 = flat_out / s12345;
        var rem = flat_out % s12345;
        coord1 = rem / s2345;
        rem = rem % s2345;
        coord2 = rem / s345;
        rem = rem % s345;
        coord3 = rem / s45;
        rem = rem % s45;
        coord4 = rem / s5;
        coord5 = rem % s5;
    }

    // Step 2: Determine which input tensor this element comes from
    // and adjust the concat dimension coordinate
    var concat_coord = coord0;
    if (concat_dim == 1u) { concat_coord = coord1; }
    else if (concat_dim == 2u) { concat_coord = coord2; }
    else if (concat_dim == 3u) { concat_coord = coord3; }
    else if (concat_dim == 4u) { concat_coord = coord4; }
    else if (concat_dim == 5u) { concat_coord = coord5; }

    var input_idx = 0u;
    var offset = 0u;
    var value = f32(0);
    var found = false;

    // Check each input sequentially
    if (parameters.numInputs > 0u && !found) {
        if (concat_coord < offset + parameters.input0DimSize) {
            let local_coord = concat_coord - offset;
            var idx = 0u;
            if (concat_dim == 0u) {
                if (rank >= 1u) { idx += local_coord * parameters.input0Stride0; }
                if (rank >= 2u) { idx += coord1 * parameters.input0Stride1; }
                if (rank >= 3u) { idx += coord2 * parameters.input0Stride2; }
                if (rank >= 4u) { idx += coord3 * parameters.input0Stride3; }
                if (rank >= 5u) { idx += coord4 * parameters.input0Stride4; }
                if (rank >= 6u) { idx += coord5 * parameters.input0Stride5; }
            } else if (concat_dim == 1u) {
                if (rank >= 1u) { idx += coord0 * parameters.input0Stride0; }
                if (rank >= 2u) { idx += local_coord * parameters.input0Stride1; }
                if (rank >= 3u) { idx += coord2 * parameters.input0Stride2; }
                if (rank >= 4u) { idx += coord3 * parameters.input0Stride3; }
                if (rank >= 5u) { idx += coord4 * parameters.input0Stride4; }
                if (rank >= 6u) { idx += coord5 * parameters.input0Stride5; }
            } else if (concat_dim == 2u) {
                if (rank >= 1u) { idx += coord0 * parameters.input0Stride0; }
                if (rank >= 2u) { idx += coord1 * parameters.input0Stride1; }
                if (rank >= 3u) { idx += local_coord * parameters.input0Stride2; }
                if (rank >= 4u) { idx += coord3 * parameters.input0Stride3; }
                if (rank >= 5u) { idx += coord4 * parameters.input0Stride4; }
                if (rank >= 6u) { idx += coord5 * parameters.input0Stride5; }
            } else if (concat_dim == 3u) {
                if (rank >= 1u) { idx += coord0 * parameters.input0Stride0; }
                if (rank >= 2u) { idx += coord1 * parameters.input0Stride1; }
                if (rank >= 3u) { idx += coord2 * parameters.input0Stride2; }
                if (rank >= 4u) { idx += local_coord * parameters.input0Stride3; }
                if (rank >= 5u) { idx += coord4 * parameters.input0Stride4; }
                if (rank >= 6u) { idx += coord5 * parameters.input0Stride5; }
            } else if (concat_dim == 4u) {
                if (rank >= 1u) { idx += coord0 * parameters.input0Stride0; }
                if (rank >= 2u) { idx += coord1 * parameters.input0Stride1; }
                if (rank >= 3u) { idx += coord2 * parameters.input0Stride2; }
                if (rank >= 4u) { idx += coord3 * parameters.input0Stride3; }
                if (rank >= 5u) { idx += local_coord * parameters.input0Stride4; }
                if (rank >= 6u) { idx += coord5 * parameters.input0Stride5; }
            } else if (concat_dim == 5u) {
                if (rank >= 1u) { idx += coord0 * parameters.input0Stride0; }
                if (rank >= 2u) { idx += coord1 * parameters.input0Stride1; }
                if (rank >= 3u) { idx += coord2 * parameters.input0Stride2; }
                if (rank >= 4u) { idx += coord3 * parameters.input0Stride3; }
                if (rank >= 5u) { idx += coord4 * parameters.input0Stride4; }
                if (rank >= 6u) { idx += local_coord * parameters.input0Stride5; }
            }
            value = input0[idx];
            found = true;
        }
        offset += parameters.input0DimSize;
    }

    if (parameters.numInputs > 1u && !found) {
        if (concat_coord < offset + parameters.input1DimSize) {
            let local_coord = concat_coord - offset;
            var idx = 0u;
            if (concat_dim == 0u) {
                if (rank >= 1u) { idx += local_coord * parameters.input1Stride0; }
                if (rank >= 2u) { idx += coord1 * parameters.input1Stride1; }
                if (rank >= 3u) { idx += coord2 * parameters.input1Stride2; }
                if (rank >= 4u) { idx += coord3 * parameters.input1Stride3; }
                if (rank >= 5u) { idx += coord4 * parameters.input1Stride4; }
                if (rank >= 6u) { idx += coord5 * parameters.input1Stride5; }
            } else if (concat_dim == 1u) {
                if (rank >= 1u) { idx += coord0 * parameters.input1Stride0; }
                if (rank >= 2u) { idx += local_coord * parameters.input1Stride1; }
                if (rank >= 3u) { idx += coord2 * parameters.input1Stride2; }
                if (rank >= 4u) { idx += coord3 * parameters.input1Stride3; }
                if (rank >= 5u) { idx += coord4 * parameters.input1Stride4; }
                if (rank >= 6u) { idx += coord5 * parameters.input1Stride5; }
            } else if (concat_dim == 2u) {
                if (rank >= 1u) { idx += coord0 * parameters.input1Stride0; }
                if (rank >= 2u) { idx += coord1 * parameters.input1Stride1; }
                if (rank >= 3u) { idx += local_coord * parameters.input1Stride2; }
                if (rank >= 4u) { idx += coord3 * parameters.input1Stride3; }
                if (rank >= 5u) { idx += coord4 * parameters.input1Stride4; }
                if (rank >= 6u) { idx += coord5 * parameters.input1Stride5; }
            } else if (concat_dim == 3u) {
                if (rank >= 1u) { idx += coord0 * parameters.input1Stride0; }
                if (rank >= 2u) { idx += coord1 * parameters.input1Stride1; }
                if (rank >= 3u) { idx += coord2 * parameters.input1Stride2; }
                if (rank >= 4u) { idx += local_coord * parameters.input1Stride3; }
                if (rank >= 5u) { idx += coord4 * parameters.input1Stride4; }
                if (rank >= 6u) { idx += coord5 * parameters.input1Stride5; }
            } else if (concat_dim == 4u) {
                if (rank >= 1u) { idx += coord0 * parameters.input1Stride0; }
                if (rank >= 2u) { idx += coord1 * parameters.input1Stride1; }
                if (rank >= 3u) { idx += coord2 * parameters.input1Stride2; }
                if (rank >= 4u) { idx += coord3 * parameters.input1Stride3; }
                if (rank >= 5u) { idx += local_coord * parameters.input1Stride4; }
                if (rank >= 6u) { idx += coord5 * parameters.input1Stride5; }
            } else if (concat_dim == 5u) {
                if (rank >= 1u) { idx += coord0 * parameters.input1Stride0; }
                if (rank >= 2u) { idx += coord1 * parameters.input1Stride1; }
                if (rank >= 3u) { idx += coord2 * parameters.input1Stride2; }
                if (rank >= 4u) { idx += coord3 * parameters.input1Stride3; }
                if (rank >= 5u) { idx += coord4 * parameters.input1Stride4; }
                if (rank >= 6u) { idx += local_coord * parameters.input1Stride5; }
            }
            value = input1[idx];
            found = true;
        }
        offset += parameters.input1DimSize;
    }

    // Handle inputs 2-7 using a loop-like pattern
    for (var input_i = 2u; input_i < parameters.numInputs; input_i++) {
        if (!found) {
            var dimSize = 0u;
            var stride0 = 0u;
            var stride1 = 0u;
            var stride2 = 0u;
            var stride3 = 0u;
            var stride4 = 0u;
            var stride5 = 0u;

            // Get parameters for current input
            if (input_i == 2u) {
                dimSize = parameters.input2DimSize;
                stride0 = parameters.input2Stride0;
                stride1 = parameters.input2Stride1;
                stride2 = parameters.input2Stride2;
                stride3 = parameters.input2Stride3;
                stride4 = parameters.input2Stride4;
                stride5 = parameters.input2Stride5;
            }

            if (concat_coord < offset + dimSize) {
                let local_coord = concat_coord - offset;
                var idx = 0u;
                if (concat_dim == 0u) {
                    if (rank >= 1u) { idx += local_coord * stride0; }
                    if (rank >= 2u) { idx += coord1 * stride1; }
                    if (rank >= 3u) { idx += coord2 * stride2; }
                    if (rank >= 4u) { idx += coord3 * stride3; }
                    if (rank >= 5u) { idx += coord4 * stride4; }
                    if (rank >= 6u) { idx += coord5 * stride5; }
                } else if (concat_dim == 1u) {
                    if (rank >= 1u) { idx += coord0 * stride0; }
                    if (rank >= 2u) { idx += local_coord * stride1; }
                    if (rank >= 3u) { idx += coord2 * stride2; }
                    if (rank >= 4u) { idx += coord3 * stride3; }
                    if (rank >= 5u) { idx += coord4 * stride4; }
                    if (rank >= 6u) { idx += coord5 * stride5; }
                } else if (concat_dim == 2u) {
                    if (rank >= 1u) { idx += coord0 * stride0; }
                    if (rank >= 2u) { idx += coord1 * stride1; }
                    if (rank >= 3u) { idx += local_coord * stride2; }
                    if (rank >= 4u) { idx += coord3 * stride3; }
                    if (rank >= 5u) { idx += coord4 * stride4; }
                    if (rank >= 6u) { idx += coord5 * stride5; }
                } else if (concat_dim == 3u) {
                    if (rank >= 1u) { idx += coord0 * stride0; }
                    if (rank >= 2u) { idx += coord1 * stride1; }
                    if (rank >= 3u) { idx += coord2 * stride2; }
                    if (rank >= 4u) { idx += local_coord * stride3; }
                    if (rank >= 5u) { idx += coord4 * stride4; }
                    if (rank >= 6u) { idx += coord5 * stride5; }
                } else if (concat_dim == 4u) {
                    if (rank >= 1u) { idx += coord0 * stride0; }
                    if (rank >= 2u) { idx += coord1 * stride1; }
                    if (rank >= 3u) { idx += coord2 * stride2; }
                    if (rank >= 4u) { idx += coord3 * stride3; }
                    if (rank >= 5u) { idx += local_coord * stride4; }
                    if (rank >= 6u) { idx += coord5 * stride5; }
                } else if (concat_dim == 5u) {
                    if (rank >= 1u) { idx += coord0 * stride0; }
                    if (rank >= 2u) { idx += coord1 * stride1; }
                    if (rank >= 3u) { idx += coord2 * stride2; }
                    if (rank >= 4u) { idx += coord3 * stride3; }
                    if (rank >= 5u) { idx += coord4 * stride4; }
                    if (rank >= 6u) { idx += local_coord * stride5; }
                }

                // Read from the appropriate input based on input_i
                if (input_i == 2u) { value = input2[idx]; }

                found = true;
            }
            offset += dimSize;
        }
    }

    output[flat_out] = value;
`
    }
};
