
import * as fs from "fs";
import * as path from "path";
import { spawn } from "node:child_process";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";
import { PluginManifestSchema } from "../marketplace/MarketplaceSchemas.js";
import { z } from "zod";

// Logic from C2A in chunk_363.ts

function getPluginsCacheDir(): string {
    return path.join(getConfigDir(), "plugins", "cache");
}

function getSafePath(base: string, ...parts: string[]): string {
    const safeParts = parts.map(p => (p || "unknown").replace(/[^a-zA-Z0-9\-_.]/g, "-"));
    return path.join(base, ...safeParts);
}

function generateTempDirName(source: any): string {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 8);
    let sourceType = "unknown";

    if (typeof source === "string") sourceType = "local";
    else if (source.source) sourceType = source.source === "url" ? "git" : source.source;

    return `temp_${sourceType}_${timestamp}_${random}`;
}

async function runCommand(command: string, args: string[], cwd?: string, env?: any): Promise<{ code: number, stdout: string, stderr: string }> {
    return new Promise((resolve, reject) => {
        const p = spawn(command, args, {
            cwd,
            env: { ...process.env, ...env },
            shell: true
        });
        let stdout = "";
        let stderr = "";
        p.stdout.on("data", d => stdout += d.toString());
        p.stderr.on("data", d => stderr += d.toString());

        p.on("close", code => resolve({ code: code || 0, stdout, stderr }));
        p.on("error", reject);
    });
}
async function installNpmPackage(packageName: string, dest: string) {
    const npmCacheDir = path.join(getConfigDir(), "plugins", "npm-cache");
    if (!fs.existsSync(npmCacheDir)) fs.mkdirSync(npmCacheDir, { recursive: true });

    const packageDir = path.join(npmCacheDir, "node_modules", packageName);
    if (!fs.existsSync(packageDir)) {
        console.log(`Installing npm package ${packageName} to cache...`);
        const result = await runCommand("npm", ["install", packageName, "--prefix", npmCacheDir]);
        if (result.code !== 0) throw new Error(`Failed to install npm package: ${result.stderr}`);
    }

    copyRecursive(packageDir, dest);
}

async function cloneGitRepo(url: string, dest: string, ref?: string) {
    const args = ["clone", "--depth", "1"];
    if (ref) args.push("--branch", ref);
    args.push(url, dest);

    const result = await runCommand("git", args);
    if (result.code !== 0) throw new Error(`Failed to clone repository: ${result.stderr}`);
}

async function fetchFromGithub(repo: string, dest: string, ref?: string) {
    if (!/^[a-zA-Z0-9-_.]+\/[a-zA-Z0-9-_.]+$/.test(repo)) {
        throw new Error(`Invalid GitHub repository format: ${repo}. Expected format: owner/repo`);
    }
    const url = `https://github.com/${repo}.git`;
    await cloneGitRepo(url, dest, ref);
}

function copyRecursive(src: string, dest: string) {
    if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
    const entries = fs.readdirSync(src, { withFileTypes: true });
    for (const entry of entries) {
        const srcPath = path.join(src, entry.name);
        const destPath = path.join(dest, entry.name);
        if (entry.isDirectory()) {
            copyRecursive(srcPath, destPath);
        } else if (entry.isFile()) {
            fs.copyFileSync(srcPath, destPath);
        } else if (entry.isSymbolicLink()) {
            // Simplistic symlink handling - try to copy target if internal
            // For now, mirroring original logic approximately
            try {
                const target = fs.readlinkSync(srcPath);
                fs.symlinkSync(target, destPath);
            } catch (e) {
                // Ignore symlink errors
            }
        }
    }
}

async function fetchSourceToTemp(source: any, tempPath: string) {
    if (typeof source === "string") {
        if (!fs.existsSync(source)) throw new Error(`Source path does not exist: ${source}`);
        copyRecursive(source, tempPath);
        // remove .git
        const gitDir = path.join(tempPath, ".git");
        if (fs.existsSync(gitDir)) fs.rmSync(gitDir, { recursive: true, force: true });
    } else {
        switch (source.source) {
            case "npm":
                await installNpmPackage(source.package, tempPath);
                break;
            case "github":
                await fetchFromGithub(source.repo, tempPath, source.ref);
                break;
            case "url":
                await cloneGitRepo(source.url, tempPath, source.ref);
                break;
            case "pip":
                throw new Error("Python package plugins are not yet supported");
            default:
                throw new Error("Unsupported plugin source type");
        }
    }
}

export async function cachePluginToTemp(source: any, options: any = {}): Promise<{ path: string, manifest: any }> {
    const baseCacheDir = getPluginsCacheDir();
    if (!fs.existsSync(baseCacheDir)) fs.mkdirSync(baseCacheDir, { recursive: true });

    const tempDirName = generateTempDirName(source);
    const tempPath = path.join(baseCacheDir, tempDirName);
    let installed = false;

    try {
        await fetchSourceToTemp(source, tempPath);
        installed = true;

        // Find manifest
        const manifestResult = findManifest(tempPath);
        let manifest = manifestResult.manifest;
        if (!manifest) {
            // Fallback from options or default
            manifest = options.manifest || {
                name: typeof source === "string" ? path.basename(source) : (source.package || source.repo || "unknown"),
                description: `Plugin cached from ${typeof source === "string" ? source : source.source}`
            };
        }

        // Move to final versioned cache
        const version = manifest.version || (typeof source !== "string" && source.ref) || "unknown";
        const pluginName = manifest.name;

        // Construct final path: cacheDir / pluginName / version
        // The original logic uses a deeper nesting: cache / name / source_key / version approx
        const finalDest = getSafePath(baseCacheDir, pluginName, version);

        if (fs.existsSync(finalDest)) {
            fs.rmSync(finalDest, { recursive: true, force: true });
        }

        fs.mkdirSync(path.dirname(finalDest), { recursive: true });
        fs.renameSync(tempPath, finalDest);

        return {
            path: finalDest,
            manifest
        };

    } catch (error) {
        if (installed && fs.existsSync(tempPath)) {
            try {
                fs.rmSync(tempPath, { recursive: true, force: true });
            } catch (e) { }
        }
        throw error;
    }
}

function findManifest(pluginPath: string): { manifest?: any, error?: string } {
    const locations = [
        path.join(pluginPath, ".claude-plugin", "plugin.json"),
        path.join(pluginPath, "plugin.json")
    ];

    for (const loc of locations) {
        if (fs.existsSync(loc)) {
            try {
                const content = fs.readFileSync(loc, "utf-8");
                const json = JSON.parse(content);
                const parsed = PluginManifestSchema.safeParse(json);
                if (parsed.success) {
                    return { manifest: parsed.data };
                } else {
                    return { error: `Invalid manifest at ${loc}: ${parsed.error.toString()}` };
                }
            } catch (e: any) {
                return { error: `Failed to parse manifest at ${loc}: ${e.message}` };
            }
        }
    }
    return {};
}
