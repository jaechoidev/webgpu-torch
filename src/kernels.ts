import { KernelSpec } from "./kernel";

export const registry: { [name: string]: KernelSpec } = {};

import { kernels as kernels_opgen } from "./kernels_opgen";
for (const name in kernels_opgen) {
    registry[name] = kernels_opgen[name];
}

import { kernels as kernels_artisanal } from "./kernels_artisanal";
for (const name in kernels_artisanal) {
    registry[name] = kernels_artisanal[name];
}

export function registerKernel(name: string, spec: KernelSpec, overwrite: boolean = false): void {
    if (registry[name] !== undefined && !overwrite) {
        throw new Error(`Kernel "${name}" is already registered. Use overwrite=true to replace it.`);
    }
    registry[name] = spec;
}

export function registerKernels(kernels: { [name: string]: KernelSpec }, overwrite: boolean = false): void {
    for (const name in kernels) {
        registerKernel(name, kernels[name], overwrite);
    }
}
