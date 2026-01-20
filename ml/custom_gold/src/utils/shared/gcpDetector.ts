import { readFileSync, statSync } from "fs";
import { platform, networkInterfaces } from "os";

export const GCE_LINUX_BIOS_PATHS = {
    BIOS_DATE: "/sys/class/dmi/id/bios_date",
    BIOS_VENDOR: "/sys/class/dmi/id/bios_vendor"
};

const GCE_MAC_PREFIX = /^42:01/;

/**
 * Detects if the current environment is a Google Cloud Serverless environment.
 */
export function isGoogleCloudServerless(): boolean {
    return !!(
        process.env.CLOUD_RUN_JOB ||
        process.env.FUNCTION_NAME ||
        process.env.K_SERVICE
    );
}

/**
 * Detects if the current environment is a Google Compute Engine Linux instance
 * by checking BIOS vendor information.
 */
export function isGoogleComputeEngineLinux(): boolean {
    if (platform() !== "linux") return false;
    try {
        statSync(GCE_LINUX_BIOS_PATHS.BIOS_DATE);
        const vendor = readFileSync(GCE_LINUX_BIOS_PATHS.BIOS_VENDOR, "utf8");
        return /Google/.test(vendor);
    } catch {
        return false;
    }
}

/**
 * Detects if the current environment is a Google Compute Engine instance
 * by checking the MAC address of network interfaces.
 */
export function isGoogleComputeEngineMACAddress(): boolean {
    const interfaces = networkInterfaces();
    for (const interfaceList of Object.values(interfaces)) {
        if (!interfaceList) continue;
        for (const { mac } of interfaceList) {
            if (GCE_MAC_PREFIX.test(mac)) return true;
        }
    }
    return false;
}

/**
 * Detects if the current environment is any Google Compute Engine instance.
 */
export function isGoogleComputeEngine(): boolean {
    return isGoogleComputeEngineLinux() || isGoogleComputeEngineMACAddress();
}

/**
 * Main entry point for detecting GCP residency.
 */
export function detectGCPResidency(): boolean {
    return isGoogleCloudServerless() || isGoogleComputeEngine();
}
