import {
    AutoFunction,
    FunctionInput,
    GradientContext,
    GradientFunctionOutput,
} from "./autograd";
import { matmul } from "./ops_artisanal";
import { sum } from "./ops_opgen";
import { Shape } from "./shape";
import type { Tensor } from "./tensor";

export class GatherFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        let [input, dim, index] = inputs as [Tensor, number, Tensor];
        if (dim < 0) {
            dim += input.shape.length;
        }
        const rank = input.shape.length;
        if (index.shape.length !== rank) {
            throw new Error(`Index shape ${index.shape} does not match input shape ${input.shape}`);
        }
        const outputShape = index.shape.slice();

        // Compute total output size
        let outputSize = 1;
        for (let i = 0; i < outputShape.length; i++) {
            outputSize *= outputShape[i];
        }

        // Pass shape and stride information to kernel
        const params: any = {
            dim,
            outputSize,
            rank
        };

        // Pass input shape and strides (up to 5D, pad with 1s if less)
        for (let i = 0; i < 5; i++) {
            params[`inputShape${i}`] = i < rank ? input.shape[i] : 1;
            params[`inputStride${i}`] = i < rank ? input.strides[i] : 0;
            params[`indexStride${i}`] = i < rank ? index.strides[i] : 0;
            params[`outputShape${i}`] = i < rank ? outputShape[i] : 1;
        }

        return input.runKernel("gather", {dtype: input.dtype}, params, [outputShape], index)[0];
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ) {
        const [input, dim, index] = inputs as [Tensor, number, Tensor];
        ctx.saveForBackward(input, index);
        ctx.dim = dim;
    }
    static backward(
        ctx: GradientContext,
        gradOutput: Tensor
    ): GradientFunctionOutput[] {
        const [input, index] = ctx.savedTensors;
        const dim: number = ctx.dim;
        throw new Error("Gather backward not implemented");
    }
}

function sumTo(tensor: Tensor, shape: Shape): Tensor {
    const sizes = tensor.shape;
    const reduceDims: number[] = [];
    const leadingDims = sizes.length - shape.length;
    for (let i = 0; i < leadingDims; i++) {
        reduceDims.push(i);
    }
    for (let i = leadingDims; i < sizes.length; i++) {
        if (shape[i - leadingDims] === 1 && sizes[i] !== 1) {
            reduceDims.push(i);
        }
    }
    if (reduceDims.length > 0) {
        tensor = tensor.sum(reduceDims, true);
    }
    return leadingDims > 0 ? tensor.view(shape) : tensor;
}

export class LinearFunction extends AutoFunction {
    static forward(inputs: FunctionInput[]): Tensor {
        const [input, weight, bias] = inputs as [Tensor, Tensor, Tensor?];
        const output = matmul(input, weight.t());
        if (bias) {
            output.add_(bias);
        }
        return output;
    }
    static setupContext(
        ctx: GradientContext,
        inputs: FunctionInput[],
        output: Tensor
    ) {
        const [input, weight, bias] = inputs as [Tensor, Tensor, Tensor];
        ctx.saveForBackward(input, weight, bias);
    }
    static backward(
        ctx: GradientContext,
        gradOutput: Tensor
    ): GradientFunctionOutput[] {
        const [input, weight, bias] = ctx.savedTensors;
        let inputGrad: Tensor | null = null;
        let weightGrad: Tensor | null = null;
        let biasGrad: Tensor | null = null;
        if (ctx.needsInputGradient[0]) {
            inputGrad = matmul(gradOutput, weight);
        }
        if (ctx.needsInputGradient[1]) {
            weightGrad = matmul(gradOutput.t(), input);
        }
        if (ctx.needsInputGradient[2]) {
            biasGrad = sumTo(gradOutput, bias.shape);
        }
        return [inputGrad, weightGrad, biasGrad];
    }
}
