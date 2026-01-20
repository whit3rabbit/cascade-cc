import { execSync, execFileSync } from "node:child_process";
import * as os from "os";
import * as path from "path";
import * as fs from "fs";
import { createConnection } from "net";
import semver from "semver";
import { IDE_REGISTRY, CLI_DISPLAY_NAMES, JETBRAINS_FOLDER_MAP } from "./IdeRegistry.js";
import { memoize } from "../../utils/shared/lodashLikeRuntimeAndEnv.js";

const CLAUDE_JETBRAINS_PLUGIN_ID = "claude-code-jetbrains-plugin";
const CLAUDE_VSCODE_EXTENSION_ID = "anthropic.claude-code";

export class WslPathConverter {
    constructor(private wslDistroName?: string) { }

    toLocalPath(wslPath: string): string {
        if (!wslPath) return wslPath;
        if (this.wslDistroName) {
            const match = wslPath.match(/^\\\\wsl(?:\.localhost|\$)\\([^\\]+)(.*)$/);
            if (match && match[1] !== this.wslDistroName) return wslPath;
        }
        try {
            return execFileSync("wslpath", ["-u", wslPath], { encoding: "utf8", stdio: ["pipe", "pipe", "ignore"] }).trim();
        } catch {
            return wslPath.replace(/\\/g, "/").replace(/^([A-Z]):/i, (_, drive) => `/mnt/${drive.toLowerCase()}`);
        }
    }

    toIDEPath(p: string): string {
        if (!p) return p;
        try {
            return execFileSync("wslpath", ["-w", p], { encoding: "utf8", stdio: ["pipe", "pipe", "ignore"] }).trim();
        } catch {
            return p;
        }
    }
}

export function isDistroMatch(path: string, distro: string): boolean {
    const match = path.match(/^\\\\wsl(?:\.localhost|\$)\\([^\\]+)(.*)$/);
    if (match) return match[1] === distro;
    return true;
}

export function isProcessRunning(pid: number): boolean {
    try {
        process.kill(pid, 0);
        return true;
    } catch {
        return false;
    }
}

export function isJetBrainsTerminal(): boolean {
    return Boolean(process.env.TERMINAL_EMULATOR === 'JetBrains-JediTerm');
}

export function isVscodeTerminal(): boolean {
    return Boolean(process.env.VSCODE_INJECTION || process.env.TERM_PROGRAM === 'vscode');
}

export function isIdeTerminal(): boolean {
    return isVscodeTerminal() || isJetBrainsTerminal() || Boolean(process.env.FORCE_CODE_TERMINAL);
}

export function isVscode(ide?: string): boolean {
    if (!ide) return false;
    return IDE_REGISTRY[ide]?.ideKind === 'vscode';
}

export function isJetBrains(ide?: string): boolean {
    if (!ide) return false;
    return IDE_REGISTRY[ide]?.ideKind === 'jetbrains';
}

export function getTerminalIde(): string | null {
    return isIdeTerminal() ? (process.env.TERMINAL_PROGRAM || null) : null;
}

export function getJetBrainsConfigDirs(ide: string): string[] {
    const home = os.homedir();
    const dirs: string[] = [];
    const folderNames = JETBRAINS_FOLDER_MAP[ide.toLowerCase()];
    if (!folderNames) return dirs;

    const roamingAppData = process.env.APPDATA || path.join(home, "AppData", "Roaming");
    const localAppData = process.env.LOCALAPPDATA || path.join(home, "AppData", "Local");

    switch (os.platform()) {
        case "darwin":
            dirs.push(path.join(home, "Library", "Application Support", "JetBrains"), path.join(home, "Library", "Application Support"));
            if (ide.toLowerCase() === "androidstudio") dirs.push(path.join(home, "Library", "Application Support", "Google"));
            break;
        case "win32":
            dirs.push(path.join(roamingAppData, "JetBrains"), path.join(localAppData, "JetBrains"), roamingAppData);
            if (ide.toLowerCase() === "androidstudio") dirs.push(path.join(localAppData, "Google"));
            break;
        case "linux":
            dirs.push(path.join(home, ".config", "JetBrains"), path.join(home, ".local", "share", "JetBrains"));
            for (const name of folderNames) dirs.push(path.join(home, "." + name));
            if (ide.toLowerCase() === "androidstudio") dirs.push(path.join(home, ".config", "Google"));
            break;
    }
    return dirs;
}

export function getJetBrainsPluginDirs(ide: string): string[] {
    const pluginDirs: string[] = [];
    const configDirs = getJetBrainsConfigDirs(ide);
    const folderNames = JETBRAINS_FOLDER_MAP[ide.toLowerCase()];
    if (!folderNames) return pluginDirs;

    for (const configDir of configDirs) {
        if (!fs.existsSync(configDir)) continue;
        for (const name of folderNames) {
            const regex = new RegExp("^" + name + ".*$");
            try {
                const entries = fs.readdirSync(configDir, { withFileTypes: true });
                const matches = entries
                    .filter(e => e.isDirectory() && regex.test(e.name))
                    .map(e => path.join(configDir, e.name));
                for (const match of matches) {
                    const pluginsPath = os.platform() === "linux" ? match : path.join(match, "plugins");
                    if (fs.existsSync(pluginsPath)) pluginDirs.push(pluginsPath);
                }
            } catch { continue; }
        }
    }
    return [...new Set(pluginDirs)];
}

export function hasClaudeJetBrainsPlugin(ide: string): boolean {
    const pluginDirs = getJetBrainsPluginDirs(ide);
    for (const dir of pluginDirs) {
        if (fs.existsSync(path.join(dir, CLAUDE_JETBRAINS_PLUGIN_ID))) return true;
    }
    return false;
}

export const isClaudeJetBrainsPluginInstalled = memoize(hasClaudeJetBrainsPlugin);

export function getIdeDisplayName(ide?: string): string {
    if (!ide) return "IDE";
    const registryEntry = IDE_REGISTRY[ide];
    if (registryEntry) return registryEntry.displayName;

    const normalized = ide.toLowerCase().trim();
    if (CLI_DISPLAY_NAMES[normalized]) return CLI_DISPLAY_NAMES[normalized];

    const firstWord = ide.split(" ")[0];
    const base = firstWord ? path.basename(firstWord).toLowerCase() : null;
    if (base && CLI_DISPLAY_NAMES[base]) return CLI_DISPLAY_NAMES[base];

    return ide;
}

export function getVscodeExePath(ide: string): string | null {
    const pathFromPs = getVscodeBinPathFromPs();
    if (pathFromPs && fs.existsSync(pathFromPs)) return pathFromPs;

    switch (ide) {
        case "vscode": return "code";
        case "cursor": return "cursor";
        case "windsurf": return "windsurf";
        default: return null;
    }
}

export function getVscodeBinPathFromPs(): string | null {
    try {
        if (os.platform() !== "darwin") return null;
        let ppid: number | null = process.ppid;
        for (let i = 0; i < 10; i++) {
            if (!ppid || ppid === 0 || ppid === 1) break;
            const cmd = execSync(`ps -o command= -p ${ppid}`, { encoding: "utf8" }).trim();
            if (cmd) {
                const appMap: Record<string, string> = {
                    "Visual Studio Code.app": "code",
                    "Cursor.app": "cursor",
                    "Windsurf.app": "windsurf",
                    "Visual Studio Code - Insiders.app": "code",
                    "VSCodium.app": "codium"
                };
                const electronSuffix = "/Contents/MacOS/Electron";
                for (const [app, bin] of Object.entries(appMap)) {
                    const idx = cmd.indexOf(app + electronSuffix);
                    if (idx !== -1) {
                        return cmd.substring(0, idx + app.length) + "/Contents/Resources/app/bin/" + bin;
                    }
                }
            }
            const ppidStr = execSync(`ps -o ppid= -p ${ppid}`, { encoding: "utf8" }).trim();
            if (!ppidStr) break;
            ppid = parseInt(ppidStr);
        }
    } catch { }
    return null;
}

export async function detectRunningIdes(): Promise<string[]> {
    const runningIdes: string[] = [];
    const platform = os.platform();
    try {
        if (platform === "darwin") {
            const output = execSync('ps aux | grep -E "Visual Studio Code|Code Helper|Cursor Helper|Windsurf Helper|IntelliJ IDEA|PyCharm|WebStorm|PhpStorm|RubyMine|CLion|GoLand|Rider|DataGrip|AppCode|DataSpell|Aqua|Gateway|Fleet|Android Studio" | grep -v grep', { encoding: "utf8" });
            for (const [id, def] of Object.entries(IDE_REGISTRY)) {
                if (def.processKeywordsMac.some(k => output.includes(k))) runningIdes.push(id);
            }
        } else if (platform === "win32") {
            const output = execSync('tasklist', { encoding: "utf8" }).toLowerCase();
            for (const [id, def] of Object.entries(IDE_REGISTRY)) {
                if (def.processKeywordsWindows.some(k => output.includes(k.toLowerCase()))) runningIdes.push(id);
            }
        } else if (platform === "linux") {
            const output = execSync('ps aux | grep -E "code|cursor|windsurf|idea|pycharm|webstorm|phpstorm|rubymine|clion|goland|rider|datagrip|dataspell|aqua|gateway|fleet|android-studio" | grep -v grep', { encoding: "utf8" }).toLowerCase();
            for (const [id, def] of Object.entries(IDE_REGISTRY)) {
                if (def.processKeywordsLinux.some(k => output.includes(k))) {
                    if (id === "vscode") {
                        if (!output.includes("cursor") && !output.includes("appcode")) runningIdes.push(id);
                    } else {
                        runningIdes.push(id);
                    }
                }
            }
        }
    } catch { }
    return [...new Set(runningIdes)];
}

export interface IdeConnectionInfo {
    workspaceFolders: string[];
    port: number;
    pid?: number;
    ideName?: string;
    useWebSocket?: boolean;
    runningInWindows?: boolean;
    authToken?: string;
}

export function getIdeLockFilePaths(): string[] {
    const ideDirs: string[] = [];
    const ideDirDefault = path.join(os.homedir(), ".claude", "ide");
    if (fs.existsSync(ideDirDefault)) ideDirs.push(ideDirDefault);

    if (os.platform() === 'linux' && process.env.WSL_DISTRO_NAME) {
        // Logic for WSL to find Window side lock files would go here
        // Gs3 in chunk_332.ts has more complex logic for this
    }

    const paths: string[] = [];
    for (const d of ideDirs) {
        try {
            const files = fs.readdirSync(d, { withFileTypes: true })
                .filter(f => f.isFile() && f.name.endsWith(".lock"))
                .map(f => {
                    const fullPath = path.join(d, f.name);
                    return { path: fullPath, mtime: fs.statSync(fullPath).mtime };
                });
            files.sort((a, b) => b.mtime.getTime() - a.mtime.getTime());
            paths.push(...files.map(f => f.path));
        } catch { }
    }
    return paths;
}

export function parseIdeLockFile(filePath: string): IdeConnectionInfo | null {
    try {
        const content = fs.readFileSync(filePath, "utf8");
        let info: any;
        try {
            info = JSON.parse(content);
        } catch {
            info = { workspaceFolders: content.split("\n").map(l => l.trim()).filter(Boolean) };
        }
        const fileName = path.basename(filePath);
        const port = parseInt(fileName.replace(".lock", ""));
        return {
            workspaceFolders: info.workspaceFolders || [],
            port,
            pid: info.pid,
            ideName: info.ideName,
            useWebSocket: info.transport === "ws",
            runningInWindows: info.runningInWindows === true,
            authToken: info.authToken
        };
    } catch {
        return null;
    }
}

export async function isPortOpen(host: string, port: number, timeout = 500): Promise<boolean> {
    return new Promise((resolve) => {
        const socket = createConnection({ host, port, timeout });
        socket.on("connect", () => {
            socket.destroy();
            resolve(true);
        });
        socket.on("error", () => resolve(false));
        socket.on("timeout", () => {
            socket.destroy();
            resolve(false);
        });
    });
}

export interface IdeConnection {
    url: string;
    name: string;
    workspaceFolders: string[];
    port: number;
    isValid: boolean;
    authToken?: string;
    ideRunningInWindows?: boolean;
}

export async function getRunningIdeConnections(all = false): Promise<IdeConnection[]> {
    const connections: IdeConnection[] = [];
    const lockFiles = getIdeLockFilePaths();
    const currentCwd = process.cwd();
    const ssePort = process.env.CLAUDE_CODE_SSE_PORT ? parseInt(process.env.CLAUDE_CODE_SSE_PORT) : null;

    for (const file of lockFiles) {
        const info = parseIdeLockFile(file);
        if (!info) continue;

        if (info.pid && !isProcessRunning(info.pid)) {
            // Basic check, might need more for WSL
            try { fs.unlinkSync(file); } catch { }
            continue;
        }

        let isValid = process.env.CLAUDE_CODE_IDE_SKIP_VALID_CHECK === "true" || info.port === ssePort;
        if (!isValid) {
            isValid = info.workspaceFolders.some(folder => {
                if (!folder) return false;
                const normalizedFolder = path.resolve(folder);
                const normalizedCwd = path.resolve(currentCwd);
                if (os.platform() === 'win32') {
                    const c = normalizedCwd.replace(/^[a-zA-Z]:/, (m) => m.toUpperCase());
                    const f = normalizedFolder.replace(/^[a-zA-Z]:/, (m) => m.toUpperCase());
                    return c === f || c.startsWith(f + path.sep);
                }
                return normalizedCwd === normalizedFolder || normalizedCwd.startsWith(normalizedFolder + path.sep);
            });
        }

        if (!isValid && !all) continue;

        const name = info.ideName || getIdeDisplayName(getTerminalIde() || "IDE");
        const host = "127.0.0.1"; // Host detection from gA2 in chunk_332.ts omitted for brevity
        const url = info.useWebSocket ? `ws://${host}:${info.port}` : `http://${host}:${info.port}/sse`;

        connections.push({
            url,
            name,
            workspaceFolders: info.workspaceFolders,
            port: info.port,
            isValid,
            authToken: info.authToken,
            ideRunningInWindows: info.runningInWindows
        });
    }
    return connections;
}

export function cleanupIdeLockFiles() {
    const lockFiles = getIdeLockFilePaths();
    for (const file of lockFiles) {
        const info = parseIdeLockFile(file);
        if (info && info.pid && !isProcessRunning(info.pid)) {
            try { fs.unlinkSync(file); } catch { }
        }
    }
}

let connectionAbortController: AbortController | null = null;

export async function waitForIdeConnection(): Promise<IdeConnection | null> {
    if (connectionAbortController) connectionAbortController.abort();
    connectionAbortController = new AbortController();
    const { signal } = connectionAbortController;

    cleanupIdeLockFiles();
    const start = Date.now();
    while (Date.now() - start < 30000 && !signal.aborted) {
        const connections = await getRunningIdeConnections(false);
        if (signal.aborted) return null;
        if (connections.length === 1) return connections[0];
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    return null;
}

export async function closeAllDiffTabs(ideConnection: any) {
    try {
        // This would call the IDE RPC method 'closeAllDiffTabs'
        // Implementation depends on the RPC client used
    } catch { }
}

export function getClaudeCodeVersion(): string {
    return "2.0.76"; // Matching chunk_332.ts RA2()
}

export async function hasConnectedIde(connections: IdeConnection[]): Promise<boolean> {
    return connections.some(c => c.isValid);
}
