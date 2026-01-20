import * as fs from 'node:fs';
import * as path from 'node:path';
import * as crypto from 'node:crypto';
import { unzip } from 'fflate';
import axios from 'axios';
import { z } from 'zod';
import { fileSystem as jA } from '../../utils/file-system/fileUtils.js';

// --- Schemas from chunk_360.ts ---

const McpConfigSchema = z.object({
    command: z.string().optional(),
    args: z.array(z.string()).optional(),
    env: z.record(z.string(), z.string()).optional(),
    platform_overrides: z.record(z.string(), z.object({
        command: z.string().optional(),
        args: z.array(z.string()).optional(),
        env: z.record(z.string(), z.string()).optional()
    })).optional()
});

const AuthorSchema = z.object({
    name: z.string(),
    email: z.string().email().optional(),
    url: z.string().url().optional()
});

const RepositorySchema = z.object({
    type: z.string(),
    url: z.string().url()
});

const UserConfigSchema = z.object({
    type: z.enum(["string", "number", "boolean", "directory", "file"]),
    title: z.string(),
    description: z.string(),
    required: z.boolean().optional(),
    default: z.union([z.string(), z.number(), z.boolean(), z.array(z.string())]).optional(),
    multiple: z.boolean().optional(),
    sensitive: z.boolean().optional(),
    min: z.number().optional(),
    max: z.number().optional()
});

const McpBundleManifestSchema = z.object({
    $schema: z.string().optional(),
    manifest_version: z.string().optional(),
    dxt_version: z.string().optional(),
    name: z.string(),
    display_name: z.string().optional(),
    version: z.string(),
    description: z.string(),
    long_description: z.string().optional(),
    author: AuthorSchema,
    repository: RepositorySchema.optional(),
    homepage: z.string().url().optional(),
    documentation: z.string().url().optional(),
    support: z.string().url().optional(),
    icon: z.string().optional(),
    screenshots: z.array(z.string()).optional(),
    server: z.object({
        type: z.enum(["python", "node", "binary"]),
        entry_point: z.string(),
        mcp_config: McpConfigSchema.optional()
    }),
    tools: z.array(z.object({
        name: z.string(),
        description: z.string().optional()
    })).optional(),
    prompts: z.array(z.object({
        name: z.string(),
        description: z.string().optional(),
        arguments: z.array(z.string()).optional(),
        text: z.string()
    })).optional(),
    user_config: z.record(z.string(), UserConfigSchema).optional()
}).refine(m => m.manifest_version || m.dxt_version, {
    message: "manifest_version or dxt_version (deprecated) is required"
});

export type McpBundleManifest = z.infer<typeof McpBundleManifestSchema>;

// --- Variable Expansion (zWA and AKA in chunk_360.ts) ---

export function expandEnvVars(text: string): { expanded: string; missingVars: string[] } {
    const missingVars: string[] = [];
    const expanded = text.replace(/\$\{([^}]+)\}/g, (match, content) => {
        const [varName, defaultValue] = content.split(":-", 2);
        const value = process.env[varName];
        if (value !== undefined) return value;
        if (defaultValue !== undefined) return defaultValue;
        missingVars.push(varName);
        return match;
    });
    return { expanded, missingVars };
}

function expandPlaceholders(obj: any, replacements: Record<string, string>): any {
    if (typeof obj === "string") {
        let result = obj;
        for (const [key, val] of Object.entries(replacements)) {
            const pattern = new RegExp(`\\$\\{${key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\}`, "g");
            result = result.replace(pattern, val);
        }
        return result;
    }
    if (Array.isArray(obj)) {
        const result: any[] = [];
        for (const item of obj) {
            if (typeof item === "string" && item.match(/^\$\{user_config\.[^}]+\}$/)) {
                const key = item.match(/^\$\{([^}]+)\}$/)?.[1];
                if (key && replacements[key]) {
                    const val = replacements[key];
                    // If the value is an array or string that looks like one, we should ideally handle it,
                    // but most likely it's just a string here. Original code mG0 has some array logic.
                    result.push(val);
                } else {
                    result.push(item);
                }
            } else {
                result.push(expandPlaceholders(item, replacements));
            }
        }
        return result;
    }
    if (obj && typeof obj === "object") {
        const result: any = {};
        for (const [key, val] of Object.entries(obj)) {
            result[key] = expandPlaceholders(val, replacements);
        }
        return result;
    }
    return obj;
}

export async function generateMcpConfigFromManifest(params: {
    manifest: McpBundleManifest;
    extensionPath: string;
    systemDirs: Record<string, string>;
    userConfig: Record<string, any>;
    pathSeparator: string;
}) {
    const { manifest, extensionPath, systemDirs, userConfig, pathSeparator } = params;
    const baseConfig = manifest.server?.mcp_config;
    if (!baseConfig) return null;

    let config = { ...baseConfig };

    // Platform overrides
    if (config.platform_overrides && process.platform in config.platform_overrides) {
        const override = (config.platform_overrides as Record<string, any>)[process.platform];
        if (override.command) config.command = override.command;
        if (override.args) config.args = override.args;
        if (override.env) config.env = { ...config.env, ...override.env };
    }

    // Check required user config
    if (manifest.user_config) {
        for (const [key, schema] of Object.entries(manifest.user_config)) {
            const s = schema as any;
            if (s.required && (userConfig[key] === undefined || userConfig[key] === "")) {
                return null;
            }
        }
    }

    // Build replacements
    const replacements: Record<string, string> = {
        __dirname: extensionPath,
        pathSeparator,
        "/": pathSeparator,
        ...systemDirs
    };

    if (manifest.user_config) {
        for (const [key, schema] of Object.entries(manifest.user_config)) {
            const s = schema as any;
            const val = userConfig[key] ?? s.default;
            if (val !== undefined) {
                const rKey = `user_config.${key}`;
                if (Array.isArray(val)) replacements[rKey] = val.map(String).join(","); // Simple join
                else if (typeof val === "boolean") replacements[rKey] = val ? "true" : "false";
                else replacements[rKey] = String(val);
            }
        }
    }

    return expandPlaceholders(config, replacements);
}

// --- Internal Utilities (chunk_361.ts helpers) ---

function getCacheMetadataPath(cacheDir: string, source: string) {
    const hash = crypto.createHash("md5").update(source).digest("hex").substring(0, 8);
    return path.join(cacheDir, `${hash}.metadata.json`);
}

function getContentHash(data: Uint8Array) {
    return crypto.createHash("sha256").update(data).digest("hex").substring(0, 16);
}

// --- Main Loader Logic (zPA in chunk_361.ts) ---

export async function loadMcpBundle(
    source: string,
    workspacePath: string,
    pluginRepository: string,
    progressCallback?: (msg: string) => void,
    userConfigOverride?: any,
    forceRevalidate: boolean = false
): Promise<any> {
    const cacheDir = path.join(workspacePath, ".mcpb-cache");
    if (!fs.existsSync(cacheDir)) fs.mkdirSync(cacheDir, { recursive: true });

    const metadataPath = getCacheMetadataPath(cacheDir, source);
    let cachedMetadata: any = null;
    if (fs.existsSync(metadataPath)) {
        try {
            cachedMetadata = JSON.parse(fs.readFileSync(metadataPath, "utf-8"));
        } catch { }
    }

    // Check if cache is valid (H65)
    let useCache = false;
    if (cachedMetadata && !forceRevalidate) {
        if (fs.existsSync(cachedMetadata.extractedPath)) {
            if (source.startsWith("http://") || source.startsWith("https://")) {
                useCache = true;
            } else {
                const bundlePath = path.isAbsolute(source) ? source : path.join(workspacePath, source);
                if (fs.existsSync(bundlePath)) {
                    const stat = fs.statSync(bundlePath);
                    if (new Date(cachedMetadata.cachedAt).getTime() >= stat.mtimeMs) {
                        useCache = true;
                    }
                }
            }
        }
    }

    if (useCache && cachedMetadata) {
        const manifestPath = path.join(cachedMetadata.extractedPath, "manifest.json");
        if (fs.existsSync(manifestPath)) {
            const manifestData = fs.readFileSync(manifestPath);
            const manifest = McpBundleManifestSchema.parse(JSON.parse(new TextDecoder().decode(manifestData)));

            // Handle user config (P62 equivalent: read from settings)
            // For now we assume userConfigOverride or empty
            const userConfig = userConfigOverride || {};

            const mcpConfig = await generateMcpConfigFromManifest({
                manifest,
                extensionPath: cachedMetadata.extractedPath,
                systemDirs: getSystemDirs(),
                userConfig,
                pathSeparator: path.sep
            });

            if (mcpConfig) {
                return {
                    manifest,
                    mcpConfig,
                    extractedPath: cachedMetadata.extractedPath,
                    contentHash: cachedMetadata.contentHash
                };
            }

            // If mcpConfig is null, it might need config
            return {
                status: "needs-config",
                manifest,
                extractedPath: cachedMetadata.extractedPath,
                contentHash: cachedMetadata.contentHash,
                configSchema: manifest.user_config,
                existingConfig: userConfig
            };
        }
    }

    // Not using cache, load/download bundle
    let bundleData: Uint8Array;
    let bundleFile: string;

    if (source.startsWith("http://") || source.startsWith("https://")) {
        if (progressCallback) progressCallback(`Downloading ${source}...`);
        const response = await axios.get(source, { responseType: 'arraybuffer' });
        bundleData = new Uint8Array(response.data);
        const l = crypto.createHash("md5").update(source).digest("hex").substring(0, 8);
        bundleFile = path.join(cacheDir, `${l}.mcpb`);
        fs.writeFileSync(bundleFile, Buffer.from(bundleData));
    } else {
        const bundlePath = path.isAbsolute(source) ? source : path.join(workspacePath, source);
        if (!fs.existsSync(bundlePath)) throw new Error(`MCPB file not found: ${bundlePath}`);
        if (progressCallback) progressCallback(`Loading ${source}...`);
        bundleData = new Uint8Array(fs.readFileSync(bundlePath));
        bundleFile = bundlePath;
    }

    const contentHash = getContentHash(bundleData);
    if (progressCallback) progressCallback("Extracting MCPB archive...");

    // Extract ZIP (pG0)
    const files = await new Promise<Record<string, Uint8Array>>((resolve, reject) => {
        unzip(bundleData, (err, data) => {
            if (err) reject(err);
            else resolve(data);
        });
    });

    const manifestRaw = files["manifest.json"];
    if (!manifestRaw) throw new Error("No manifest.json found in MCPB file");

    const manifest = McpBundleManifestSchema.parse(JSON.parse(new TextDecoder().decode(manifestRaw)));
    const extractionPath = path.join(cacheDir, contentHash);

    // Extract to dir (V65)
    if (!fs.existsSync(extractionPath)) fs.mkdirSync(extractionPath, { recursive: true });
    for (const [name, data] of Object.entries(files)) {
        const fullPath = path.join(extractionPath, name);
        fs.mkdirSync(path.dirname(fullPath), { recursive: true });
        if (name.endsWith('.json') || name.endsWith('.js') || name.endsWith('.ts') || name.endsWith('.txt') || name.endsWith('.md')) {
            fs.writeFileSync(fullPath, new TextDecoder().decode(data));
        } else {
            fs.writeFileSync(fullPath, Buffer.from(data));
        }
    }

    const metadata = {
        source,
        contentHash,
        extractedPath: extractionPath,
        cachedAt: new Date().toISOString(),
        lastChecked: new Date().toISOString()
    };
    fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));

    const userConfig = userConfigOverride || {};
    const mcpConfig = await generateMcpConfigFromManifest({
        manifest,
        extensionPath: extractionPath,
        systemDirs: getSystemDirs(),
        userConfig,
        pathSeparator: path.sep
    });

    if (!mcpConfig) {
        return {
            status: "needs-config",
            manifest,
            extractedPath: extractionPath,
            contentHash,
            configSchema: manifest.user_config,
            existingConfig: userConfig
        };
    }

    return {
        manifest,
        mcpConfig,
        extractedPath: extractionPath,
        contentHash
    };
}

function getSystemDirs(): Record<string, string> {
    const home = process.env.HOME || process.env.USERPROFILE || "";
    return {
        HOME: home,
        DESKTOP: path.join(home, "Desktop"),
        DOCUMENTS: path.join(home, "Documents"),
        DOWNLOADS: path.join(home, "Downloads")
    };
}
