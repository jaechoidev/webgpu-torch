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

    console.log(`[WebGPU] Adapter limits:`, {
        maxBufferSize: `${(adapter.limits.maxBufferSize / 1024 / 1024).toFixed(2)} MB`,
        maxStorageBufferBindingSize: `${(adapter.limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(2)} MB`,
    });

    // Request maximum limits supported by the adapter for large tensor operations
    const device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
        }
    });

    // Log the actual limits we got
    console.log(`[WebGPU] Device limits (requested):`, {
        maxBufferSize: `${(device.limits.maxBufferSize / 1024 / 1024).toFixed(2)} MB`,
        maxStorageBufferBindingSize: `${(device.limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(2)} MB`,
        maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
    });

    // Calculate maximum renderable resolution based on buffer limits
    // The largest intermediate tensor in UNet is typically [1, 256, H, W] (256MB at 512x512)
    // Formula: maxStorageBufferBindingSize = 1 * channels * H * W * 4 bytes
    const maxBindingSizeBytes = device.limits.maxStorageBufferBindingSize;
    const maxChannels = 256; 
    const bytesPerElement = 4; 
    const maxPixels = maxBindingSizeBytes / (maxChannels * bytesPerElement);
    const maxResolution = Math.floor(Math.sqrt(maxPixels));

    console.log(`[WebGPU] Maximum renderable resolution estimate: ${maxResolution}x${maxResolution}`);
    console.log(`  (Based on largest UNet intermediate tensor: [1, ${maxChannels}, H, W] = ${(maxChannels * maxResolution * maxResolution * bytesPerElement / 1024 / 1024).toFixed(2)} MB)`);
    console.log(`  Supported resolutions: 256x256, 512x512, ${maxResolution >= 1024 ? '1024x1024' : ''} ${maxResolution >= 2048 ? ', 2048x2048' : ''}`);


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
