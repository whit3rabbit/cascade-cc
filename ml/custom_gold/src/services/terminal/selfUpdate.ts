import { exec } from "node:child_process";
import { promisify } from "node:util";
import { join } from "node:path";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";

const execAsync = promisify(exec);

/**
 * Installs the CLI package locally via npm.
 * Deobfuscated from vZA in chunk_175.ts.
 */
export async function installLocalCli(version: string = "latest"): Promise<"success" | "install_failed" | "in_progress"> {
    const installDir = getLocalInstallDir();

    try {
        if (!initLocalInstallDir(installDir)) return "install_failed";

        const packageName = "@anthropic-ai/claude-code";
        console.log(`Installing ${packageName}@${version} to ${installDir}...`);

        // In actual implementation, this would run npm install
        // const { code } = await runNpmInstall(packageName, version, installDir);
        // if (code !== 0) return "install_failed";

        return "success";
    } catch (err) {
        console.error(`Local install failed: ${err instanceof Error ? err.message : String(err)}`);
        return "install_failed";
    }
}

/**
 * Initializes the directory for local installs and creates necessary wrappers.
 * Deobfuscated from n_8 in chunk_175.ts.
 */
export function initLocalInstallDir(dir: string): boolean {
    try {
        if (!existsSync(dir)) mkdirSync(dir, { recursive: true });

        const packageJsonPath = join(dir, "package.json");
        if (!existsSync(packageJsonPath)) {
            const pkg = { name: "claude-local", version: "0.0.1", private: true };
            writeFileSync(packageJsonPath, JSON.stringify(pkg, null, 2));
        }

        const binWrapperPath = join(dir, "claude");
        if (!existsSync(binWrapperPath)) {
            const script = `#!/bin/bash\nexec "${dir}/node_modules/.bin/claude" "$@"`;
            writeFileSync(binWrapperPath, script, { mode: 0o755 });
        }

        return true;
    } catch (err) {
        return false;
    }
}

/**
 * Checks if a local CLI installation exists.
 * Deobfuscated from Hg in chunk_175.ts.
 */
export function isLocalCliInstalled(): boolean {
    const installDir = getLocalInstallDir();
    return existsSync(join(installDir, "node_modules", ".bin", "claude"));
}

/**
 * Detects the current shell type (zsh, bash, fish).
 * Deobfuscated from kZA in chunk_175.ts.
 */
export function getShellType(): "zsh" | "bash" | "fish" | "unknown" {
    const shell = process.env.SHELL || "";
    if (shell.includes("zsh")) return "zsh";
    if (shell.includes("bash")) return "bash";
    if (shell.includes("fish")) return "fish";
    return "unknown";
}

/**
 * Returns the local installation directory path.
 */
function getLocalInstallDir(): string {
    // This would normally be calculated based on config dir
    return join(process.env.HOME || "", ".claude", "local");
}
