/**
 * File: src/services/installer/ChromeExtensionInstaller.ts
 * Role: Manages installation of the Chrome Native Host manifest for the "Claude in Chrome" extension.
 */

import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { homedir } from "node:os";

/**
 * Returns platform-specific paths for Chrome Native Host manifests.
 */
function getNativeHostManifestPaths(): string[] {
    const platform = process.platform;
    if (platform === "win32") {
        const appData = process.env.LOCALAPPDATA || join(homedir(), "AppData", "Local");
        return [join(appData, "Claude Code", "NativeMessagingHosts")];
    }
    if (platform === "darwin") {
        return [join(homedir(), "Library", "Application Support", "Google", "Chrome", "NativeMessagingHosts")];
    }
    return [join(homedir(), ".config", "google-chrome", "NativeMessagingHosts")];
}

interface NativeHostManifest {
    name: string;
    description: string;
    path: string;
    type: string;
    allowed_origins: string[];
}

/**
 * Installs the JSON manifest required for Chrome to talk to the local Claude process.
 */
export async function installNativeHostManifest(executablePath: string): Promise<void> {
    const manifestPaths = getNativeHostManifestPaths();
    const manifest: NativeHostManifest = {
        name: "com.anthropic.claudecode",
        description: "Claude Code Browser Extension Native Host",
        path: executablePath,
        type: "stdio",
        allowed_origins: [
            "chrome-extension://fcoeoabgfenejglbffodgkkbkcdhcgfn/"
        ]
    };

    const content = JSON.stringify(manifest, null, 2);

    for (const dir of manifestPaths) {
        try {
            await mkdir(dir, { recursive: true });
            const filePath = join(dir, "com.anthropic.claudecode.json");
            await writeFile(filePath, content, "utf8");
            console.log(`[ChromeInstaller] Installed manifest at ${filePath}`);
        } catch (err: unknown) {
            const message = err instanceof Error ? err.message : String(err);
            console.error(`[ChromeInstaller] Failed to install manifest in ${dir}: ${message}`);
        }
    }
}
