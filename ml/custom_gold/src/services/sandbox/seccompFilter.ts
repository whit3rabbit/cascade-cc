import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

/**
 * Seccomp filter management.
 * Deobfuscated from Qn1, kA1, GEB in chunk_220.ts.
 */

function getArchName(): string | null {
    const arch = process.arch as string;
    switch (arch) {
        case "x64":
            return "x64";
        case "arm64":
            return "arm64";
        default:
            return null;
    }
}

function findAsset(assetPath: string): string | null {
    // @ts-ignore
    const currentFile = fileURLToPath(import.meta.url);
    const currentDir = path.dirname(currentFile);

    const searchPaths = [
        path.join(currentDir, assetPath),
        path.join(currentDir, "..", assetPath),
        path.join(currentDir, "..", "..", assetPath)
    ];

    for (const p of searchPaths) {
        if (fs.existsSync(p)) return p;
    }
    return null;
}

/**
 * Locates the pre-generated Seccomp BPF filter for the current architecture.
 */
export function getSeccompFilterPath(): string | null {
    const arch = getArchName();
    if (!arch) return null;
    return findAsset(`vendor/seccomp/${arch}/unix-block.bpf`);
}

/**
 * Locates the apply-seccomp helper binary for the current architecture.
 */
export function getApplySeccompBinaryPath(): string | null {
    const arch = getArchName();
    if (!arch) return null;
    return findAsset(`vendor/seccomp/${arch}/apply-seccomp`);
}
