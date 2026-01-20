import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as crypto from "node:crypto";
import { backupAppleTerminalSettings, backupIterm2Settings } from "./terminalSetup.js";

/**
 * Terminal configuration installers for Shift+Enter support.
 * Deobfuscated from UB3, wB3, qB3, Ti1, etc. in chunk_216.ts.
 */

/**
 * Modifies .wezterm.lua to add Shift+Enter binding.
 */
export async function installWezTermKeybinding(): Promise<string> {
    const configPath = path.join(os.homedir(), ".wezterm.lua");
    try {
        let content = "";
        let exists = false;
        if (fs.existsSync(configPath)) {
            exists = true;
            content = fs.readFileSync(configPath, "utf-8");
            if (content.includes('key="Enter"') && content.includes('mods="SHIFT"')) {
                return `Found existing WezTerm Shift+Enter binding in ${configPath}. Remove it to continue.`;
            }
            const backup = `${configPath}.${crypto.randomBytes(4).toString("hex")}.bak`;
            fs.copyFileSync(configPath, backup);
        }

        if (!exists) {
            content = `local wezterm = require 'wezterm'
local config = wezterm.config_builder()

config.keys = {
  {key="Enter", mods="SHIFT", action=wezterm.action{SendString="\\x1b\\r"}},
}

return config
`;
        } else {
            // Heuristic to insert into existing config.keys
            const keysMatch = content.match(/config\.keys\s*=\s*\{([\s\S]*?)\}/);
            if (keysMatch) {
                const existing = keysMatch[1].trim();
                const newValue = existing ? `${existing},\n  {key="Enter", mods="SHIFT", action=wezterm.action{SendString="\\x1b\\r"}},` : `\n  {key="Enter", mods="SHIFT", action=wezterm.action{SendString="\\x1b\\r"}},\n`;
                content = content.replace(/config\.keys\s*=\s*\{[\s\S]*?\}/, `config.keys = {${newValue}}`);
            } else if (content.match(/return\s+config/)) {
                content = content.replace(/return\s+config/, `config.keys = {\n  {key="Enter", mods="SHIFT", action=wezterm.action{SendString="\\x1b\\r"}},\n}\n\nreturn config`);
            } else {
                content += `\nconfig.keys = {\n  {key="Enter", mods="SHIFT", action=wezterm.action{SendString="\\x1b\\r"}},\n}\n`;
            }
        }

        fs.writeFileSync(configPath, content);
        return `Installed WezTerm Shift+Enter key binding. Restart WezTerm to apply. See ${configPath}`;
    } catch (e) {
        throw new Error(`Failed to install WezTerm binding: ${e}`);
    }
}

/**
 * Modifies Ghostty config for Shift+Enter.
 */
export async function installGhosttyKeybinding(): Promise<string> {
    const paths = [
        path.join(process.env.XDG_CONFIG_HOME || path.join(os.homedir(), ".config"), "ghostty", "config"),
        ...(os.platform() === "darwin" ? [path.join(os.homedir(), "Library", "Application Support", "com.mitchellh.ghostty", "config")] : [])
    ];

    let configPath = paths.find(p => fs.existsSync(p)) || paths[0];
    try {
        let content = "";
        if (fs.existsSync(configPath)) {
            content = fs.readFileSync(configPath, "utf-8");
            if (content.includes("shift+enter")) {
                return `Found existing Ghostty Shift+Enter binding in ${configPath}.`;
            }
            fs.copyFileSync(configPath, `${configPath}.bak`);
        } else {
            fs.mkdirSync(path.dirname(configPath), { recursive: true });
        }

        content += (content && !content.endsWith("\n") ? "\n" : "") + "keybind = shift+enter=text:\\x1b\\r\n";
        fs.writeFileSync(configPath, content);
        return `Installed Ghostty Shift+Enter binding. See ${configPath}`;
    } catch (e) {
        throw new Error(`Failed to install Ghostty binding: ${e}`);
    }
}

/**
 * Modifies VSCode/Cursor terminal keybindings.
 */
export async function installIdeKeybinding(app: "VSCode" | "Cursor" | "Windsurf" = "VSCode"): Promise<string> {
    const appName = app === "VSCode" ? "Code" : app;
    const platform = os.platform();
    const baseDir = platform === "win32"
        ? path.join(os.homedir(), "AppData", "Roaming", appName, "User")
        : platform === "darwin"
            ? path.join(os.homedir(), "Library", "Application Support", appName, "User")
            : path.join(os.homedir(), ".config", appName, "User");

    const configPath = path.join(baseDir, "keybindings.json");
    try {
        if (!fs.existsSync(baseDir)) fs.mkdirSync(baseDir, { recursive: true });

        let bindings: any[] = [];
        if (fs.existsSync(configPath)) {
            fs.copyFileSync(configPath, `${configPath}.bak`);
            try {
                bindings = JSON.parse(fs.readFileSync(configPath, "utf-8"));
            } catch (e) {
                bindings = [];
            }
        }

        if (bindings.some(b => b.key === "shift+enter" && b.command === "workbench.action.terminal.sendSequence")) {
            return `Found existing Shift+Enter binding in ${app}.`;
        }

        bindings.push({
            key: "shift+enter",
            command: "workbench.action.terminal.sendSequence",
            args: { text: "\x1B\r" },
            when: "terminalFocus"
        });

        fs.writeFileSync(configPath, JSON.stringify(bindings, null, 2));
        return `Installed ${app} terminal Shift+Enter binding. See ${configPath}`;
    } catch (e) {
        throw new Error(`Failed to install ${app} binding: ${e}`);
    }
}

/**
 * Higher-level installers for iTerm and Apple Terminal would go here,
 * calling backup functions and using 'defaults write' commands as seen in chunk 216.
 */
