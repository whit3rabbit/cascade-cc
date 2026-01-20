import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { execFileSync, execSync } from 'node:child_process';
import { createConnection } from 'node:net';
import { log } from '../logger/loggerService.js';
import { getSettings, updateSettings } from '../terminal/settings.js';

const logger = log("ide-manager");

export interface IdeInstance {
    url: string;
    name: string;
    workspaceFolders: string[];
    port: number;
    pid?: number;
    isValid: boolean;
    authToken?: string;
    ideRunningInWindows: boolean;
}

const JETBRAINS_IDEs: Record<string, string[]> = {
    pycharm: ["PyCharm"],
    intellij: ["IntelliJIdea", "IdeaIC"],
    webstorm: ["WebStorm"],
    phpstorm: ["PhpStorm"],
    rubymine: ["RubyMine"],
    clion: ["CLion"],
    goland: ["GoLand"],
    rider: ["Rider"],
    datagrip: ["DataGrip"],
    appcode: ["AppCode"],
    dataspell: ["DataSpell"],
    aqua: ["Aqua"],
    gateway: ["Gateway"],
    fleet: ["Fleet"],
    androidstudio: ["AndroidStudio"]
};

const JETBRAINS_PLUGIN_NAME = "claude-code-jetbrains-plugin";

export class IdeManager {
    static getIdeLockFiles(): string[] {
        const lockDirs = this.getIdeLockDirs();
        const lockFiles: { path: string, mtime: Date }[] = [];

        for (const dir of lockDirs) {
            if (!fs.existsSync(dir)) continue;
            try {
                const files = fs.readdirSync(dir);
                for (const file of files) {
                    if (file.endsWith(".lock")) {
                        const fullPath = path.join(dir, file);
                        const stats = fs.statSync(fullPath);
                        lockFiles.push({ path: fullPath, mtime: stats.mtime });
                    }
                }
            } catch (err) {
                logger.error(new Error(`Failed to read IDE lock dir ${dir}: ${err}`));
            }
        }

        return lockFiles.sort((a, b) => b.mtime.getTime() - a.mtime.getTime()).map(f => f.path);
    }

    static getIdeLockDirs(): string[] {
        const dirs: string[] = [];
        const home = os.homedir();

        // Local path
        const localIdeDir = path.join(home, ".claude", "ide");
        if (fs.existsSync(localIdeDir)) dirs.push(localIdeDir);

        // WSL specifics
        if (process.env.WSL_DISTRO_NAME) {
            // ... (porting WSL logic if needed)
        }

        return dirs;
    }

    static parseLockFile(filePath: string): IdeInstance | null {
        try {
            const content = fs.readFileSync(filePath, "utf-8");
            let data: any;
            try {
                data = JSON.parse(content);
            } catch {
                // Legacy format: one folder per line
                data = { workspaceFolders: content.split("\n").map(l => l.trim()).filter(Boolean) };
            }

            const fileName = path.basename(filePath);
            const port = parseInt(fileName.replace(".lock", ""), 10);

            return {
                workspaceFolders: data.workspaceFolders || [],
                port: port || data.port,
                pid: data.pid,
                ideName: data.ideName || "IDE",
                url: data.transport === "ws" ? `ws://localhost:${port}` : `http://localhost:${port}/sse`,
                useWebSocket: data.transport === "ws",
                runningInWindows: !!data.runningInWindows,
                authToken: data.authToken,
                isValid: true, // Will be verified
                name: data.ideName || "IDE"
            } as any;
        } catch (err) {
            logger.error(new Error(`Failed to parse IDE lock file ${filePath}: ${err}`));
            return null;
        }
    }

    static async isPortOpen(port: number, timeout = 500): Promise<boolean> {
        return new Promise((resolve) => {
            const socket = createConnection({ port, host: "localhost", timeout });
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

    static async getConnectedIdes(onlyValidForCwd = true): Promise<IdeInstance[]> {
        const lockFiles = this.getIdeLockFiles();
        const ides: IdeInstance[] = [];
        const cwd = process.cwd();

        for (const lockFile of lockFiles) {
            const instance = this.parseLockFile(lockFile);
            if (!instance) continue;

            const isOpen = await this.isPortOpen(instance.port);
            if (!isOpen) {
                try { fs.unlinkSync(lockFile); } catch { }
                continue;
            }

            if (onlyValidForCwd) {
                const isValid = instance.workspaceFolders.some(folder => {
                    const abs = path.resolve(folder);
                    return cwd === abs || cwd.startsWith(abs + path.sep);
                });
                if (!isValid) continue;
            }

            ides.push(instance);
        }

        return ides;
    }

    static async getActiveIde(): Promise<IdeInstance | null> {
        const ides = await this.getConnectedIdes(true);
        return ides.length > 0 ? ides[0] : null;
    }
}
