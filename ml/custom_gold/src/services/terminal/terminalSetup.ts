import { execSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

/**
 * Terminal setup and backup utilities for macOS.
 * Deobfuscated from pDB, cDB, and related in chunk_215.ts.
 */

function getAppleTerminalPlistPath(): string {
    return path.join(os.homedir(), "Library", "Preferences", "com.apple.Terminal.plist");
}

function getIterm2PlistPath(): string {
    return path.join(os.homedir(), "Library", "Preferences", "com.googlecode.iterm2.plist");
}

/**
 * Backs up Apple Terminal settings.
 */
export async function backupAppleTerminalSettings(): Promise<string | null> {
    const plistPath = getAppleTerminalPlistPath();
    const backupPath = `${plistPath}.bak`;
    try {
        // Export current settings to ensure we have the latest
        execSync(`defaults export com.apple.Terminal "${plistPath}"`);
        if (fs.existsSync(plistPath)) {
            fs.copyFileSync(plistPath, backupPath);
            return backupPath;
        }
        return null;
    } catch (e) {
        return null;
    }
}

/**
 * Backs up iTerm2 settings.
 */
export async function backupIterm2Settings(): Promise<string | null> {
    const plistPath = getIterm2PlistPath();
    const backupPath = `${plistPath}.bak`;
    try {
        execSync(`defaults export com.googlecode.iterm2 "${plistPath}"`);
        if (fs.existsSync(plistPath)) {
            fs.copyFileSync(plistPath, backupPath);
            return backupPath;
        }
        return null;
    } catch (e) {
        return null;
    }
}

/**
 * Checks if the current terminal is supported for advanced styling features.
 */
export function isSupportedTerminal(): boolean {
    const term = process.env.TERM_PROGRAM;
    if (os.platform() !== "darwin") return false;

    const supported = [
        "iTerm.app",
        "Apple_Terminal",
        "vscode",
        "cursor",
        "windsurf",
        "ghostty",
        "WezTerm",
        "kitty",
        "alacritty",
        "WarpTerminal",
        "zed"
    ];

    return supported.includes(term || "");
}

/**
 * Configures the terminal prompt or theme (stub for complex logic).
 */
export function setupTerminalPrompt() {
    // Logic from Si1 would go here - configuring specific terminal apps
    // to ensure they handle Shift+Enter or Option as Meta correctly.
}
