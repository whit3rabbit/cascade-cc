import { existsSync, readFileSync, writeFileSync, copyFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { homedir, platform } from 'node:os';
import { randomBytes } from 'node:crypto';
import { updateSettings } from '../../services/config/SettingsService.js';

export function isRemoteSession(): boolean {
    const askPass = process.env.VSCODE_GIT_ASKPASS_MAIN || "";
    const path = process.env.PATH || "";
    return askPass.includes(".vscode-server") ||
        askPass.includes(".cursor-server") ||
        askPass.includes(".windsurf-server") ||
        path.includes(".vscode-server") ||
        path.includes(".cursor-server") ||
        path.includes(".windsurf-server");
}

export function getTerminalName(): string | null {
    if (process.env.TERM_PROGRAM) return process.env.TERM_PROGRAM;
    if (process.env.ITERM_SESSION_ID) return "iterm2";
    if (process.env.WARP_COMPLETION_STRATEGY) return "warp";
    if (process.env.TERM && process.env.TERM.includes("apple")) return "apple_terminal";
    // Fallback checks could go here
    return "generic_terminal";
}

export interface InstallResult {
    success: boolean;
    message: string;
}

export function installKeybindings(terminalName: string): InstallResult {
    if (isRemoteSession()) {
        return {
            success: false,
            message: `Cannot install keybindings from a remote ${terminalName} session.\nPlease install them on your local machine.`
        };
    }

    const appName = terminalName === "vscode" ? "Code" :
        terminalName === "cursor" ? "Cursor" :
            terminalName === "windsurf" ? "Windsurf" : terminalName;

    if (!["Code", "Cursor", "Windsurf"].includes(appName)) {
        return {
            success: false,
            message: `Automatic keybinding installation is not supported for ${terminalName}.`
        };
    }

    let configDir;
    if (platform() === "win32") {
        configDir = join(homedir(), "AppData", "Roaming", appName, "User");
    } else if (platform() === "darwin") {
        configDir = join(homedir(), "Library", "Application Support", appName, "User");
    } else {
        configDir = join(homedir(), ".config", appName, "User");
    }

    const keybindingsPath = join(configDir, "keybindings.json");

    try {
        if (!existsSync(configDir)) {
            mkdirSync(configDir, { recursive: true });
        }

        let keybindings: any[] = [];
        let content = "[]";

        if (existsSync(keybindingsPath)) {
            content = readFileSync(keybindingsPath, "utf-8");
            // Improved JSON parse - strip comments and trailing commas for VS Code compatibility
            try {
                const stripped = content
                    .replace(/\/\/.*|\/\*[\s\S]*?\*\//g, "") // Strip comments
                    .replace(/,(\s*[\]}])/g, "$1"); // Strip trailing commas
                keybindings = JSON.parse(stripped);
            } catch {
                console.warn("Failed to parse keybindings.json, treating as empty array.");
            }

            // Backup
            const id = randomBytes(4).toString("hex");
            try {
                copyFileSync(keybindingsPath, `${keybindingsPath}.${id}.bak`);
            } catch {
                return {
                    success: false,
                    message: "Failed to backup existing keybindings. Aborting."
                };
            }
        }

        const hasBinding = keybindings.find((k: any) =>
            k.key === "shift+enter" &&
            k.command === "workbench.action.terminal.sendSequence" &&
            k.when === "terminalFocus"
        );

        if (hasBinding) {
            updateSettings(s => ({ ...s, shiftEnterKeyBindingInstalled: true }));
            return {
                success: true,
                message: `Shift+Enter keybinding is already installed for ${appName}.`
            };
        }

        const newBinding = {
            "key": "shift+enter",
            "command": "workbench.action.terminal.sendSequence",
            "args": { "text": "\u001b\r" },
            "when": "terminalFocus"
        };

        keybindings.push(newBinding);
        writeFileSync(keybindingsPath, JSON.stringify(keybindings, null, 2), "utf-8");

        updateSettings(s => ({ ...s, shiftEnterKeyBindingInstalled: true }));

        return {
            success: true,
            message: `Successfully installed Shift+Enter keybinding for ${appName}.`
        };

    } catch (error: any) {
        return {
            success: false,
            message: `Failed to install keybindings: ${error.message}`
        };
    }
}
