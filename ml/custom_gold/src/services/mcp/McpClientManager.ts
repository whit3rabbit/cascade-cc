import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { createMcpClient } from "./McpClientFactory.js";
import { McpServerManager } from "./McpServerManager.js";
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);
import { log } from '../logger/loggerService.js';

const logger = log('McpClientManager');
const clientCache = new Map<string, any>();

export async function connectAllMcpServers(callback: (clientWrapper: any) => void) {
    // Logic from DZ0
    const { servers } = await McpServerManager.getAllMcpServers();
    const batchSize = parseInt(process.env.MCP_SERVER_CONNECTION_BATCH_SIZE || "", 10) || 3;

    const entries = Object.entries(servers);

    for (let i = 0; i < entries.length; i += batchSize) {
        const batch = entries.slice(i, i + batchSize);
        await Promise.all(batch.map(async ([name, config]) => {
            // Check disabled logic
            const clientWrapper = await getMcpClient(name, config);
            callback(clientWrapper);
        }));
    }
}

export async function getMcpClient(name: string, config: any) {
    // Logic from Nm wrapper/memoization
    const key = `${name}-${JSON.stringify(config)}`;
    if (clientCache.has(key)) return clientCache.get(key);

    const client = await createMcpClient(name, config, {});
    clientCache.set(key, client);
    return client;
}

export async function disconnectMcpServer(name: string, config: any) {
    const key = `${name}-${JSON.stringify(config)}`;
    const client = clientCache.get(key);
    if (client && client.cleanup) await client.cleanup();
    clientCache.delete(key);
}


import { getSettings, updateSettings } from '../terminal/settings.js';

export async function getInstalledPlugins(): Promise<{ enabled: any[], disabled: any[] }> {
    const store = getPluginStore();
    const enabled: any[] = [];
    const disabled: any[] = [];

    // Get enabled plugins from all scopes
    const userSettings = getSettings('userSettings');
    const projectSettings = getSettings('projectSettings');
    const localSettings = getSettings('localSettings');

    const allEnabled = {
        ...userSettings?.enabledPlugins,
        ...projectSettings?.enabledPlugins,
        ...localSettings?.enabledPlugins
    };

    for (const [pluginId, installations] of Object.entries(store.plugins)) {
        if (!Array.isArray(installations)) continue;

        // Find if any installation of this plugin is active/relevant for current context
        // (Simplified for now: if it's in the store, we report it)
        const isEnabled = !!allEnabled[pluginId];

        // Take latest installation as the primary one for data display
        const latest = installations[installations.length - 1];
        const pluginData = {
            id: pluginId,
            name: pluginId.split('@')[0],
            version: latest.version,
            installPath: latest.installPath,
            scope: latest.scope,
            projectPath: latest.projectPath
        };

        if (isEnabled) {
            enabled.push(pluginData);
        } else {
            disabled.push(pluginData);
        }
    }

    return { enabled, disabled };
}

export function refreshPlugins() {
    clearInstalledPluginsCache();
    // In a real app, this would emit an event or call a callback
    // to notify the agent that tools/plugins have changed.
    logger.info("Plugins refreshed.");
}

export function clearInstalledPluginsCache() {
    clientCache.clear();
}

export async function installPluginInternal(pluginId: string, entry: any, scope: string, projectPath: string, installPath?: string): Promise<string> {
    // Logic from Dz in chunk_362.ts
    // 1. Download/Copy to temp
    // 2. Determine final install Path
    // 3. Move to final path
    // 4. Record installation

    // For now, if we have installPath (local), we use it.
    // If not, we download.

    let actualInstallPath = installPath;
    if (!actualInstallPath) {
        // download logic ...
        actualInstallPath = getPluginInstallPath(pluginId, entry.version || "1.0.0");
    }

    // Stub implementation that matches what's expected
    recordInstallation(pluginId, scope, projectPath || "", actualInstallPath, entry.version || "1.0.0");

    return actualInstallPath;
}

export function recordUninstallation(pluginId: string, scope: string, projectPath: string) {
    const store = getPluginStore();
    const pluginEntries = store.plugins[pluginId];
    if (!pluginEntries) return;

    const filtered = pluginEntries.filter((e: any) => !(e.scope === scope && e.projectPath === projectPath));
    if (filtered.length === 0) {
        delete store.plugins[pluginId];
    } else {
        store.plugins[pluginId] = filtered;
    }

    const filePath = path.join(os.homedir(), '.claude', 'installed_plugins.json');
    fs.writeFileSync(filePath, JSON.stringify(store, null, 2), { encoding: 'utf-8' });
}

export function removeDirectory(dirPath: string) {
    if (fs.existsSync(dirPath)) {
        fs.rmSync(dirPath, { recursive: true, force: true });
    }
}

export function isManagedScope(scope: string) {
    return scope === 'managed';
}

export function togglePluginInternal(pluginId: string, enable: boolean, entry: any, target: { scope: string, projectPath?: string }) {
    const scope = target.scope as any;
    const settings = getSettings(scope);
    const enabledPlugins = { ...settings?.enabledPlugins };

    if (enable) {
        enabledPlugins[pluginId] = true;
    } else {
        delete enabledPlugins[pluginId];
    }

    updateSettings(scope, { enabledPlugins });
}

export function getPluginStore() {
    const filePath = path.join(os.homedir(), '.claude', 'installed_plugins.json');
    if (fs.existsSync(filePath)) {
        try {
            return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        } catch { }
    }
    return {
        plugins: {} as Record<string, any[]>
    };
}

export async function downloadPlugin(source: any, options: any) {
    const tempDir = path.join(os.tmpdir(), `claude-plugin-${Date.now()}-${Math.random().toString(36).substring(2, 8)}`);
    fs.mkdirSync(tempDir, { recursive: true });

    try {
        if (typeof source === 'string') {
            // Assume it's a local path
            recursiveCopy(source, tempDir);
        } else if (source.source === 'npm') {
            await execFileAsync('npm', ['install', source.package, '--prefix', tempDir]);
            // npm install --prefix puts things in node_modules
            const pkgPath = path.join(tempDir, 'node_modules', source.package);
            const finalTemp = path.join(os.tmpdir(), `claude-plugin-extracted-${Date.now()}`);
            recursiveCopy(pkgPath, finalTemp);
            fs.rmSync(tempDir, { recursive: true, force: true });
            return { path: finalTemp, manifest: options?.manifest || {} };
        } else if (source.source === 'github') {
            const repoUrl = `https://github.com/${source.repo}.git`;
            const args = ['clone', '--depth', '1'];
            if (source.ref) args.push('--branch', source.ref);
            args.push(repoUrl, tempDir);
            await execFileAsync('git', args);
        } else if (source.source === 'url') {
            const args = ['clone', '--depth', '1'];
            if (source.ref) args.push('--branch', source.ref);
            args.push(source.url, tempDir);
            await execFileAsync('git', args);
        } else {
            throw new Error(`Unsupported plugin source: ${JSON.stringify(source)}`);
        }

        // Try to load manifest
        const manifestPath = path.join(tempDir, '.claude-plugin', 'plugin.json');
        let manifest = options?.manifest || {};
        if (fs.existsSync(manifestPath)) {
            manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf-8'));
        }

        return { path: tempDir, manifest };
    } catch (err) {
        if (fs.existsSync(tempDir)) fs.rmSync(tempDir, { recursive: true, force: true });
        throw err;
    }
}

export async function getLatestVersion(pluginId: string, _source: any, manifest: any, currentPath: string, providedVersion?: string) {
    if (manifest?.version) return manifest.version;
    if (providedVersion) return providedVersion;

    // Try git SHA
    try {
        const { stdout } = await execFileAsync('git', ['rev-parse', 'HEAD'], { cwd: currentPath });
        if (stdout) return stdout.trim().substring(0, 12);
    } catch { }

    return "unknown";
}

export function getPluginInstallPath(pluginId: string, version: string) {
    const baseDir = path.join(os.homedir(), '.claude', 'plugins', 'cache');
    const [name, marketplace] = pluginId.split('@');
    const safeMarketplace = (marketplace || 'unknown').replace(/[^a-zA-Z0-9\-_]/g, '-');
    const safeName = (name || pluginId).replace(/[^a-zA-Z0-9\-_]/g, '-');
    const safeVersion = version.replace(/[^a-zA-Z0-9\-_.]/g, '-');

    return path.join(baseDir, safeMarketplace, safeName, safeVersion);
}

function recursiveCopy(src: string, dest: string) {
    if (!fs.existsSync(dest)) fs.mkdirSync(dest, { recursive: true });
    const entries = fs.readdirSync(src, { withFileTypes: true });

    for (const entry of entries) {
        const srcPath = path.join(src, entry.name);
        const destPath = path.join(dest, entry.name);

        if (entry.isDirectory()) {
            recursiveCopy(srcPath, destPath);
        } else if (entry.isFile()) {
            fs.copyFileSync(srcPath, destPath);
        } else if (entry.isSymbolicLink()) {
            const target = fs.readlinkSync(srcPath);
            fs.symlinkSync(target, destPath);
        }
    }
}

export async function copyPluginToInstallPath(sourcePath: string, pluginId: string, version: string, _entry: any) {
    const dest = getPluginInstallPath(pluginId, version);
    if (fs.existsSync(dest)) {
        fs.rmSync(dest, { recursive: true, force: true });
    }
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    recursiveCopy(sourcePath, dest);

    // Cleanup .git if it exists
    const gitDir = path.join(dest, '.git');
    if (fs.existsSync(gitDir)) {
        fs.rmSync(gitDir, { recursive: true, force: true });
    }
}

export function recordInstallation(pluginId: string, scope: string, projectPath: string, installPath: string, version: string) {
    const store = getPluginStore();
    const installation = {
        scope: scope as any,
        installPath,
        version,
        installedAt: new Date().toISOString(),
        lastUpdated: new Date().toISOString(),
        projectPath
    };

    const pluginEntries = store.plugins[pluginId] || [];
    const index = pluginEntries.findIndex((e: any) => e.scope === scope && e.projectPath === projectPath);
    if (index >= 0) {
        pluginEntries[index] = installation;
    } else {
        pluginEntries.push(installation);
    }
    store.plugins[pluginId] = pluginEntries;

    // Save to disk (Logic from Q70)
    const filePath = path.join(os.homedir(), '.claude', 'installed_plugins.json');
    fs.writeFileSync(filePath, JSON.stringify(store, null, 2), { encoding: 'utf-8' });
}


export async function checkPluginForUpdateInternal(pluginId: string) {
    return null;
}

/**
 * Returns the currently connected IDE client, if any. (jw in chunk_368)
 */
export async function getConnectedIdeClient() {
    // This should check active client connections for an IDE type
    const clients = Array.from(clientCache.values());
    return clients.find(c => c.name === 'ide' || c.clientType === 'ide');
}

/**
 * Returns the path to the MCP CLI state directory. (Bp in chunk_589)
 */
export function getMcpCliDir() {
    return process.env.USE_MCP_CLI_DIR || path.join(os.tmpdir(), "claude-code-mcp-cli");
}

/**
 * Returns the path to the current session's MCP state file. (OD1 in chunk_589)
 */
export function getMcpStatePath(sessionId: string) {
    return path.join(getMcpCliDir(), `${sessionId}.json`);
}

/**
 * Saves the current state of MCP clients and tools to disk for the CLI to use.
 * (RK9 in chunk_589)
 */
export async function saveMcpState(sessionId: string, clients: any[], tools: any[], resources: any) {
    const dir = getMcpCliDir();
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }

    const configs: Record<string, any> = {};
    const normalizedNames: Record<string, string> = {};

    for (const client of clients) {
        if (client.name) {
            configs[client.name] = client.config;
            const normalized = client.name.replace(/[^a-zA-Z0-9_-]/g, "_");
            normalizedNames[normalized] = client.name;
        }
    }

    const state = {
        clients: clients.map(c => ({
            name: c.name,
            type: c.type,
            capabilities: c.capabilities
        })),
        configs,
        tools: tools.filter(t => t.isMcp).map(t => ({
            name: t.name,
            description: t.description,
            inputJSONSchema: t.inputJSONSchema,
            isMcp: t.isMcp,
            originalToolName: t.originalMcpToolName
        })),
        resources,
        normalizedNames
    };

    const filePath = getMcpStatePath(sessionId);
    fs.writeFileSync(filePath, JSON.stringify(state, null, 2));
}

/**
 * Loads the MCP state from disk. (ns in chunk_604)
 */
export function loadMcpState(sessionId: string) {
    const filePath = getMcpStatePath(sessionId);
    if (!fs.existsSync(filePath)) {
        throw new Error(`MCP state file not found at ${filePath}. Is Claude Code running?`);
    }
    return JSON.parse(fs.readFileSync(filePath, "utf-8"));
}
