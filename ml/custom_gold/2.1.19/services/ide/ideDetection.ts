/**
 * File: src/services/ide/ideDetection.ts
 * Role: Service for detecting installed JetBrains IDEs and Claude plugins.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

const CLAUDE_PLUGIN_DIR_NAME = "claude-code-jetbrains-plugin";

const IDE_TO_DIR_NAME: Record<string, string[]> = {
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

import { EnvService } from '../config/EnvService.js';

/**
 * Finds potential installation paths for an IDE based on the platform.
 */
export function findIdeInstallationPaths(ideName: string): string[] {
    const homeDir = os.homedir();
    const installPaths: string[] = [];
    const directoryNames = IDE_TO_DIR_NAME[ideName.toLowerCase()];

    if (!directoryNames) return installPaths;

    const appData = EnvService.get("APPDATA") || path.join(homeDir, "AppData", "Roaming");
    const localAppData = EnvService.get("LOCALAPPDATA") || path.join(homeDir, "AppData", "Local");

    switch (os.platform()) {
        case "darwin":
            installPaths.push(
                path.join(homeDir, "Library", "Application Support", "JetBrains"),
                path.join(homeDir, "Library", "Application Support")
            );
            if (ideName.toLowerCase() === "androidstudio") {
                installPaths.push(path.join(homeDir, "Library", "Application Support", "Google"));
            }
            break;
        case "win32":
            installPaths.push(
                path.join(appData, "JetBrains"),
                path.join(localAppData, "JetBrains"),
                appData
            );
            if (ideName.toLowerCase() === "androidstudio") {
                installPaths.push(path.join(localAppData, "Google"));
            }
            break;
        case "linux":
            installPaths.push(
                path.join(homeDir, ".config", "JetBrains"),
                path.join(homeDir, ".local", "share", "JetBrains")
            );
            for (const dirName of directoryNames) {
                installPaths.push(path.join(homeDir, "." + dirName));
            }
            if (ideName.toLowerCase() === "androidstudio") {
                installPaths.push(path.join(homeDir, ".config", "Google"));
            }
            break;
    }
    return installPaths;
}

/**
 * Scans for plugin installation paths for a specific IDE.
 */
export function findPluginInstallationPaths(ideName: string): string[] {
    const pluginPaths: string[] = [];
    const ideInstallPaths = findIdeInstallationPaths(ideName);
    const directoryNames = IDE_TO_DIR_NAME[ideName.toLowerCase()];

    if (!directoryNames) return pluginPaths;

    for (const installPath of ideInstallPaths) {
        try {
            if (!fs.existsSync(installPath)) continue;

            for (const dirName of directoryNames) {
                const pluginRegex = new RegExp("^" + dirName + ".*$");
                const pluginDirectories = fs.readdirSync(installPath, { withFileTypes: true })
                    .filter(dirent => pluginRegex.test(dirent.name) && dirent.isDirectory())
                    .map(dirent => path.join(installPath, dirent.name));

                for (const pluginDir of pluginDirectories) {
                    const pluginsDir = os.platform() === "linux" ? pluginDir : path.join(pluginDir, "plugins");
                    if (fs.existsSync(pluginsDir)) {
                        pluginPaths.push(pluginsDir);
                    }
                }
            }
        } catch {
            continue;
        }
    }

    return Array.from(new Set(pluginPaths));
}

/**
 * Checks if the Claude Code plugin is installed in a given IDE.
 */
export function isClaudePluginInstalled(ideName: string): boolean {
    const pluginInstallPaths = findPluginInstallationPaths(ideName);
    for (const pluginPath of pluginInstallPaths) {
        const claudePluginPath = path.join(pluginPath, CLAUDE_PLUGIN_DIR_NAME);
        if (fs.existsSync(claudePluginPath)) {
            return true;
        }
    }
    return false;
}
