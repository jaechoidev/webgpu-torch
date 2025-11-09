// Generate code from op_spec.ts and op_table.ts
import { KernelInputSpec, KernelOutputSpec, KernelParamSpec, KernelSpec } from "./kernel";
import {
    ExprCode,
    exprCodeToWebGLShader,
    exprNodeToWebGLShader,
    parseCode,
    substituteIdentifiers,
} from "./expr";
import { BinaryOpSpec, OpSpec, ReductionOpSpec, UnaryOpSpec } from "./op_spec";

export class CodeWriter {
    private indentLevel = 0;
    private lines: string[] = [];
    indent() {
        this.indentLevel++;
    }
    dedent() {
        this.indentLevel--;
    }
    writeLine(line: string) {
        this.lines.push("    ".repeat(this.indentLevel) + line);
    }
    toString() {
        return this.lines.join("\n");
    }
}

export function opSpecToKernelSpecs(op: OpSpec): KernelSpec[] {
    if (op.type == "reduction") {
        return getReductionKernelSpecs(op as ReductionOpSpec);
    } else if (op.type == "binary") {
        return getBinaryKernelSpecs(op as BinaryOpSpec);
    } else {
        return getUnaryKernelSpecs(op as UnaryOpSpec);
    }
}

function getReductionKernelSpecs(op: ReductionOpSpec): KernelSpec[] {
    const specs = [getReductionKernelSpec(op), getReductionDimKernelSpec(op)];
    if (op.backward) {
        specs.push(getReductionGradKernelSpec(op, op.backward));
    }
    return specs;
}

function getBinaryKernelSpecs(op: BinaryOpSpec): KernelSpec[] {
    const specs = [
        getBinaryKernelSpec(op, false, false, false),
        getBinaryKernelSpec(op, true, false, false),
        getBinaryKernelSpec(op, false, true, false),
        getBinaryKernelSpec(op, true, true, false),
        getBinaryKernelSpec(op, false, false, true),
        getBinaryKernelSpec(op, true, false, true),
    ];
    if (op.backward) {
        specs.push(getBinaryGradKernelSpec(op, op.backward, false));
        specs.push(getBinaryGradKernelSpec(op, op.backward, true));
    }
    return specs;
}

function getUnaryKernelSpecs(op: UnaryOpSpec): KernelSpec[] {
    const specs = [getUnaryKernelSpec(op), getUnaryInplaceKernelSpec(op)];
    if (op.backward) {
        specs.push(getUnaryGradKernelSpec(op, op.backward));
    }
    return specs;
}

function getReductionKernelSpec(op: ReductionOpSpec): KernelSpec {
    const initCode = exprCodeToWebGLShader(op.init, {
        input: "input[local_id.x]",
        output: "accumulator",
    });
    const forwardCode = exprCodeToWebGLShader(op.forward, {
        input: "input[i]",
        output: "accumulator",
    });
    const reduceCode =
        op.reduce === undefined
            ? ""
            : exprCodeToWebGLShader(op.reduce, {
                  input: "input[i]",
                  output: "accumulator",
                  inputSize: "parameters.size",
              });
    const shader = `
    var ${initCode};
    // Load inputData into local memory
    for (var i = local_id.x; i < parameters.size; i += $$workgroupSize$$) {
        ${forwardCode};
    }
    // Write partial group sum to outputData
    output[local_id.x] = accumulator;

    workgroupBarrier(); // Make sure all threads have completed reduction

    // First thread sums up results from all other threads
    if (local_id.x == 0u) {
        var numToSum = min(parameters.size, $$workgroupSize$$u);
        for (var i = 1u; i < numToSum; i++) {
            accumulator ${op.combineOp}= output[i];
        }
        // Store final reduction in the first element of result array
        ${reduceCode};
        output[0] = accumulator;
    }
`;
    return {
        name: op.name,
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
                name: "input",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "output",
                shaderType: "array<f32>",
                size: "workgroupSize",
            },
        ],
        workgroupSize: ["workgroupSize", 1, 1],
        workgroupCount: [1, 1, 1],
        shader: shader,
    };
}

function getReductionGradKernelSpec(
    op: ReductionOpSpec,
    backwardExprCode: ExprCode
): KernelSpec {
    const backwardShaderCode = exprCodeToWebGLShader(backwardExprCode, {
        input: "input[index]",
        output: "output[0]",
        outputGrad: "outputGrad[0]",
        inputGrad: "inputGrad[index]",
        inputSize: "parameters.size",
    });
    const shader = `
    let index = global_id.x;
    if (index >= parameters.size) {
        return;
    }
    ${backwardShaderCode};
`;
    return {
        name: op.name + "_grad",
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
                name: "input",
                shaderType: "array<f32>",
            },
            {
                name: "output",
                shaderType: "array<f32>",
            },
            {
                name: "outputGrad",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "inputGrad",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: ["workgroupSize", 1, 1],
        workgroupCount: ["size/workgroupSize", 1, 1],
        shader: shader,
    };
}

function getReductionDimKernelSpec(op: ReductionOpSpec): KernelSpec {
    const initCode = exprCodeToWebGLShader(op.init, {
        input: "input[inputIndex]",
        output: "accumulator",
    });
    const forwardCode = exprCodeToWebGLShader(op.forward, {
        input: "input[inputIndex]",
        output: "accumulator",
    });
    const reduceCode =
        op.reduce === undefined
            ? ""
            : exprCodeToWebGLShader(op.reduce, {
                  input: "input[inputIndex]",
                  output: "accumulator",
                  inputSize: "dimN",
              });
    let shader = `
    // Fix for 2D dispatch: properly map 2D workgroup coordinates to 1D index
    let workgroup_id_x = global_id.x / 256u;
    let workgroup_id_y = global_id.y;
    let workgroup_linear = workgroup_id_x + workgroup_id_y * 65535u;
    let outputIndex = workgroup_linear * 256u + local_id.x;
    if (outputIndex >= parameters.size) {
        return;
    }
    var i = outputIndex;
    var outputIndex0 = u32(i / parameters.outputStride0);
    i = i % parameters.outputStride0;
    var outputIndex1 = u32(i / parameters.outputStride1);
    i = i % parameters.outputStride1;
    var outputIndex2 = u32(i / parameters.outputStride2);
    i = i % parameters.outputStride2;
    var outputIndex3 = u32(i / parameters.outputStride3);
    i = i % parameters.outputStride3;
    var outputIndex4 = i;
    var outputIndices = array<u32, 5>(outputIndex0, outputIndex1, outputIndex2, outputIndex3, outputIndex4);
    var inputShapes = array<u32, 5>(parameters.inputShape0, parameters.inputShape1, parameters.inputShape2, parameters.inputShape3, parameters.inputShape4);
    var inputStrides = array<u32, 5>(parameters.inputStride0, parameters.inputStride1, parameters.inputStride2, parameters.inputStride3, parameters.inputStride4);

    var dimIdx = i32($$dim$$);
    if (dimIdx < 0) {
        dimIdx = dimIdx + i32($$maxdim$$);
    }
    let dimN = inputShapes[u32(dimIdx)];
    var ${initCode};
    for (var dimI = 0u; dimI < dimN; dimI++) {
        outputIndices[u32(dimIdx)] = dimI;
        let inputIndex =
            outputIndices[0] * inputStrides[0] +
            outputIndices[1] * inputStrides[1] +
            outputIndices[2] * inputStrides[2] +
            outputIndices[3] * inputStrides[3] +
            outputIndices[4] * inputStrides[4];
        ${forwardCode};
    }
    ${reduceCode};
    output[outputIndex] = accumulator;
`;
    return {
        name: op.name + "_dim",
        config: [
            {
                name: "dtype",
            },
            {
                name: "dim",
            },
            {
                name: "maxdim",
            },
        ],
        parameters: [
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
                name: "outputStride0",
                shaderType: "u32",
            },
            {
                name: "outputStride1",
                shaderType: "u32",
            },
            {
                name: "outputStride2",
                shaderType: "u32",
            },
            {
                name: "outputStride3",
                shaderType: "u32",
            },
            {
                name: "outputStride4",
                shaderType: "u32",
            },
            {
                name: "size",
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
                name: "output",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["65535", "((size + 255) / 256 + 65534) / 65535", 1],
        shader: shader,
    };
}

function getBinaryKernelSpec(
    op: BinaryOpSpec,
    inplace: boolean,
    isOtherScalar: boolean,
    strided: boolean
): KernelSpec {
    const maxdim = 5;
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
    ];
    if (strided) {
        // Add output shapes (needed for index decomposition)
        for (let dim = 0; dim < maxdim; dim++) {
            parameters.push({
                name: `outputShape${dim}`,
                shaderType: "u32",
            });
        }
        // Add strides for input, other, and output
        for (let dim = 0; dim < maxdim; dim++) {
            parameters.push({
                name: `inputStrides${dim}`,
                shaderType: "u32",
            });
            parameters.push({
                name: `otherStrides${dim}`,
                shaderType: "u32",
            });
            parameters.push({
                name: `outputStrides${dim}`,
                shaderType: "u32",
            });
        }
    }
    const subs: any = {
        input: "input[inputIndex]",
        other: "other[otherIndex]",
        output: "output[outputIndex]",
    };
    if (inplace) {
        subs.output = "input[outputIndex]";
    }
    if (isOtherScalar) {
        subs.other = "parameters.other";
        parameters.push({
            name: "other",
            shaderType: "f32",
        });
    }
    if (op.alpha !== undefined && op.alpha) {
        parameters.push({
            name: "alpha",
            shaderType: "f32",
        });
        subs["alpha"] = "parameters.alpha";
    }
    const shaderSnippet = exprCodeToWebGLShader(op.forward, subs);
    let shader: string;
    if (strided) {
        shader = `
        // Support 2D dispatch for large tensors in strided operations
        let workgroup_x = global_id.x / 256u;
        let workgroup_y = global_id.y;
        let workgroup_1d = workgroup_y * 65535u + workgroup_x;
        let outputIndex = workgroup_1d * 256u + (global_id.x % 256u);

        if (outputIndex >= parameters.size) {
            return;
        }

        let dim_1234 = parameters.outputShape1 * parameters.outputShape2 * parameters.outputShape3 * parameters.outputShape4;
        let dim_234 = parameters.outputShape2 * parameters.outputShape3 * parameters.outputShape4;
        let dim_34 = parameters.outputShape3 * parameters.outputShape4;
        let dim_4 = parameters.outputShape4;

        var i = outputIndex;
        let outputIndex0 = i / dim_1234;
        i = i % dim_1234;
        let outputIndex1 = i / dim_234;
        i = i % dim_234;
        let outputIndex2 = i / dim_34;
        i = i % dim_34;
        let outputIndex3 = i / dim_4;
        let outputIndex4 = i % dim_4;

        let inputIndex =
            outputIndex0 * parameters.inputStrides0 +
            outputIndex1 * parameters.inputStrides1 +
            outputIndex2 * parameters.inputStrides2 +
            outputIndex3 * parameters.inputStrides3 +
            outputIndex4 * parameters.inputStrides4;
        let otherIndex =
            outputIndex0 * parameters.otherStrides0 +
            outputIndex1 * parameters.otherStrides1 +
            outputIndex2 * parameters.otherStrides2 +
            outputIndex3 * parameters.otherStrides3 +
            outputIndex4 * parameters.otherStrides4;
        ${shaderSnippet};`;
    }
    else {
        shader = `
        // Use 2D dispatch to handle large tensors (>65535*256 elements)
        let workgroup_x = global_id.x / 256u;
        let workgroup_y = global_id.y;
        let workgroup_1d = workgroup_y * 65535u + workgroup_x;
        let outputIndex = workgroup_1d * 256u + (global_id.x % 256u);
        if (outputIndex >= parameters.size) {
            return;
        }
        let inputIndex = outputIndex;
        let otherIndex = outputIndex;
        ${shaderSnippet};`;
    }
    const inputs: KernelInputSpec[] = [];
    if (!isOtherScalar) {
        inputs.push({
            name: "other",
            shaderType: "array<f32>",
        });
    }
    let outputName = "input";
    if (!inplace) {
        inputs.splice(0, 0, {
            name: "input",
            shaderType: "array<f32>",
        });
        outputName = "output";
    }
    let name = op.name;
    if (isOtherScalar) {
        name += "_scalar";
    }
    if (strided) {
        name += "_strided";
    }
    if (inplace) {
        name += "_";
    }
    return {
        name: name,
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: parameters,
        inputs: inputs,
        outputs: [
            {
                name: outputName,
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["65535", "((size + 255) / 256 + 65534) / 65535", 1],
        shader: shader,
    };
}

function getBinaryGradKernelSpec(
    op: BinaryOpSpec,
    backward: ExprCode,
    isOtherScalar: boolean
): KernelSpec {
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
    ];
    const subs: any = {
        input: "input[idx]",
        inputGrad: "inputGrad[idx]",
        output: "output[idx]",
        outputGrad: "outputGrad[idx]",
        other: "other[idx]",
        otherGrad: "otherGrad[idx]",
    };
    if (isOtherScalar) {
        subs.other = "parameters.other";
        subs.otherGrad = "otherGrad";
        parameters.push({
            name: "other",
            shaderType: "f32",
        });
    }
    if (op.alpha !== undefined && op.alpha) {
        parameters.push({
            name: "alpha",
            shaderType: "f32",
        });
        subs["alpha"] = "parameters.alpha";
    }
    const ast = parseCode(backward);
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToWebGLShader(shaderAst);
    const shader = `
        // Use 2D dispatch to handle large tensors (>65535*256 elements)
        let workgroup_1d = (global_id.x / 256u) + global_id.y * 65535u;
        let idx = workgroup_1d * 256u + (global_id.x % 256u);
        if (idx >= parameters.size) {
            return;
        }
        ${isOtherScalar? "var otherGrad = 0.0;" : ""}
        ${shaderSnippet};`;
    const inputs: KernelInputSpec[] = [
        {
            name: "input",
            shaderType: "array<f32>",
        }];
    if (!isOtherScalar) {
        inputs.push({
            name: "other",
            shaderType: "array<f32>",
        });
    }
    inputs.push({
            name: "outputGrad",
            shaderType: "array<f32>",
        });
    const outputs: KernelOutputSpec[] = [
        {
            name: "inputGrad",
            shaderType: "array<f32>",
            size: "size",
        }];
    if (!isOtherScalar) {
        outputs.push({
            name: "otherGrad",
            shaderType: "array<f32>",
            size: "size",
        });
    }
    return {
        name: op.name + (isOtherScalar ? "_scalar" : "") + "_grad",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters,
        inputs,
        outputs,
        workgroupSize: [256, 1, 1],
        workgroupCount: ["65535", "((size + 255) / 256 + 65534) / 65535", 1],
        shader,
    };
}

function getUnaryKernelSpec(op: UnaryOpSpec): KernelSpec {
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
        // {
        //     name: "strideX",
        //     shaderType: "u32",
        // },
    ];
    const subs: any = {
        input: "input[index]",
        output: "output[index]",
        };
        if (op.alpha !== undefined && op.alpha) {
            parameters.push({
                name: "alpha",
                shaderType: "f32",
            });
            subs["alpha"] = "parameters.alpha";
        }
        const ast = parseCode(op.forward);
        const shaderAst = substituteIdentifiers(ast, subs);
        const shaderSnippet = exprNodeToWebGLShader(shaderAst);
        const shader = `
        // Use 2D dispatch to handle large tensors (>65535*256 elements)
        let workgroup_1d = (global_id.x / 256u) + global_id.y * 65535u;
        let idx = workgroup_1d * 256u + (global_id.x % 256u);
        if (idx >= parameters.size) {
            return;
        }
        ${shaderSnippet};`;
    return {
        name: op.name,
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: parameters,
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
        workgroupCount: ["65535", "((size + 255) / 256 + 65534) / 65535", 1],
        shader: shader,
    };
}

function getUnaryInplaceKernelSpec(op: UnaryOpSpec): KernelSpec {
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
    ];
    const ast = parseCode(op.forward);
    const subs: any = {
        input: "input[idx]",
        output: "input[idx]",
    };
    if (op.alpha !== undefined && op.alpha) {
        parameters.push({
            name: "alpha",
            shaderType: "f32",
        });
        subs["alpha"] = "parameters.alpha";
    }
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToWebGLShader(shaderAst);
    const shader = `
        // Use 2D dispatch to handle large tensors (>65535*256 elements)
        // Map 2D workgroup dispatch back to 1D element index
        let workgroup_1d = (global_id.x / 256u) + global_id.y * 65535u;
        let idx = workgroup_1d * 256u + (global_id.x % 256u);
        if (idx >= parameters.size) {
            return;
        }
        ${shaderSnippet};`;
    return {
        name: op.name + "_",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: parameters,
        inputs: [],
        outputs: [
            {
                name: "input",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["65535", "((size + 255) / 256 + 65534) / 65535", 1],
        shader: shader,
    };
}

function getUnaryGradKernelSpec(
    op: UnaryOpSpec,
    backward: ExprCode
): KernelSpec {
    const parameters: KernelParamSpec[] = [
        {
            name: "size",
            shaderType: "u32",
        },
    ];
    const subs: any = {
        input: "input[idx]",
        inputGrad: "inputGrad[idx]",
        output: "output[idx]",
        outputGrad: "outputGrad[idx]",
    };
    if (op.alpha !== undefined && op.alpha) {
        parameters.push({
            name: "alpha",
            shaderType: "f32",
        });
        subs["alpha"] = "parameters.alpha";
    }
    const ast = parseCode(backward);
    const shaderAst = substituteIdentifiers(ast, subs);
    const shaderSnippet = exprNodeToWebGLShader(shaderAst);
    const shader = `
        // Use 2D dispatch to handle large tensors (>65535*256 elements)
        let workgroup_1d = (global_id.x / 256u) + global_id.y * 65535u;
        let idx = workgroup_1d * 256u + (global_id.x % 256u);
        if (idx >= parameters.size) {
            return;
        }
        ${shaderSnippet};`;
    return {
        name: op.name + "_grad",
        config: [
            {
                name: "dtype",
            },
        ],
        parameters: parameters,
        inputs: [
            {
                name: "input",
                shaderType: "array<f32>",
            },
            {
                name: "outputGrad",
                shaderType: "array<f32>",
            },
        ],
        outputs: [
            {
                name: "inputGrad",
                shaderType: "array<f32>",
                size: "size",
            },
        ],
        workgroupSize: [256, 1, 1],
        workgroupCount: ["65535", "((size + 255) / 256 + 65534) / 65535", 1],
        shader: shader,
    };
}
