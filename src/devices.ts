import type { Device, DeviceType, Deviceish } from "./device";
import { DeviceCPU } from "./device_cpu";
import { DeviceWebGPU } from "./device_webgpu";

export const cpuDevice = new DeviceCPU();
let webgpuDevice: DeviceWebGPU | null = null;

const devices: { [id: string]: Device } = {
    cpu: cpuDevice,
};

export async function discoverWebGPUDevicesAsync(): Promise<boolean> {
    if (!(navigator as any).gpu) {
        // console.warn("No WebGPU devices found");
        return false;
    }
    const id = "webgpu";
    if (id in devices) {
        return true;
    }
    const adapter = await (navigator as any).gpu.requestAdapter();

    // Check if timestamp-query feature is supported
    const canTimestamp = adapter.features.has('timestamp-query');
    console.log(`[WebGPU] timestamp-query support: ${canTimestamp}`);

    const device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageBuffersPerShaderStage: 10,
            maxBufferSize: 2147483648,  // 2GB
            maxStorageBufferBindingSize: 2147483644  // 2GB - 4 bytes
        },
        requiredFeatures: [
            ...(canTimestamp ? ['timestamp-query' as GPUFeatureName] : [])
        ]
    });
    const dev = new DeviceWebGPU(id, adapter, device);
    devices[id] = dev;
    webgpuDevice = dev;
    console.log("Found WebGPU device", device);
    return true;
}

export function getDevice(device?: Deviceish | null): Device {
    if (device === null || device === undefined) {
        return webgpuDevice || cpuDevice;
    } else if (typeof device === "string") {
        if (device in devices) {
            return devices[device];
        } else {
            const found = findDeviceWithType(device as DeviceType);
            if (found) {
                return found;
            }
            throw new Error(`Device ${device} not found`);
        }
    } else {
        return device;
    }
}

function findDeviceWithType(type: DeviceType): Device | null {
    for (const id in devices) {
        if (devices[id].type === type) {
            return devices[id];
        }
    }
    return null;
}
