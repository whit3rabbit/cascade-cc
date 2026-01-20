
import { spawn } from "node:child_process";

/**
 * Runs ripgrep with the specified arguments.
 * Deobfuscated from xx in chunk_380.ts
 */
export async function runRipgrep(
    paths: string[],
    query: string,
    signal?: AbortSignal,
    options: { args?: string[] } = {}
): Promise<string[]> {
    const combinedArgs = [...(options.args || [])];
    if (query) combinedArgs.push(query);
    combinedArgs.push(...paths);

    const cwd = paths[0] || process.cwd();

    return new Promise((resolve, reject) => {
        const child = spawn("rg", combinedArgs, {
            cwd,
            signal,
            env: { ...process.env, LANG: "en_US.UTF-8" }
        });

        let stdout = "";
        let stderr = "";

        child.stdout.on("data", data => {
            stdout += data;
        });

        child.stderr.on("data", data => {
            stderr += data;
        });

        child.on("close", code => {
            if (code === 0 || code === 1) {
                // code 1 means no matches
                resolve(stdout.trim().split("\n").filter(Boolean));
            } else {
                reject(new Error(`ripgrep failed with exit code ${code}: ${stderr}`));
            }
        });

        child.on("error", err => {
            if (err.name === "AbortError") {
                resolve([]);
            } else {
                reject(err);
            }
        });
    });
}

/**
 * Checks if ripgrep is available in the path.
 * Deobfuscated from tFB in chunk_219.ts
 */
export function isRipgrepAvailable(): boolean {
    try {
        const result = spawnSync("which", ["rg"], {
            stdio: "ignore",
            timeout: 1000
        });
        return result.status === 0;
    } catch {
        return false;
    }
}

// Helper to keep imports clean
import { spawnSync } from "node:child_process";
