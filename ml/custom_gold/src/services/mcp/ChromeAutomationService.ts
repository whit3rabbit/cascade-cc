import { writeFile, mkdir, chmod, readFile, readdir, access } from 'node:fs/promises';
import { homedir } from 'node:os';
import * as path from 'node:path';
import { log } from '../logger/loggerService.js';
import { runProcess } from '../../utils/shared/runProcess.js';

const logger = log("chrome-automation");

const EXTENSION_ID = "com.anthropic.claude_code_browser_extension";
const EXTENSION_ORIGIN = "chrome-extension://fcoeoabgfenejglbffodgkkbkcdhcgfn/";
const RECONNECT_URL = "https://clau.de/chrome/reconnect";

function getManifestPath() {
    const home = homedir();
    switch (process.platform) {
        case "darwin":
            return path.join(home, "Library", "Application Support", "Google", "Chrome", "NativeMessagingHosts");
        case "linux":
            return path.join(home, ".config", "google-chrome", "NativeMessagingHosts");
        case "win32": {
            const appData = process.env.APPDATA || path.join(home, "AppData", "Local");
            return path.join(appData, "Claude Code", "ChromeNativeHost");
        }
        default:
            return null;
    }
}

function getChromeUserDataDir() {
    const home = homedir();
    switch (process.platform) {
        case "darwin":
            return path.join(home, "Library", "Application Support", "Google", "Chrome");
        case "win32":
            return path.join(home, "AppData", "Local", "Google", "Chrome", "User Data");
        case "linux":
            return path.join(home, ".config", "google-chrome");
        default:
            return null;
    }
}

async function isExtensionInstalled(): Promise<boolean> {
    const userDataDir = getChromeUserDataDir();
    if (!userDataDir) return false;

    try {
        await access(userDataDir);
        const entries = await readdir(userDataDir, { withFileTypes: true });
        const profiles = entries
            .filter(e => e.isDirectory() && (e.name === "Default" || e.name.startsWith("Profile ")))
            .map(e => e.name);

        const extensionIds = ["fcoeoabgfenejglbffodgkkbkcdhcgfn"];

        for (const profile of profiles) {
            for (const id of extensionIds) {
                const extPath = path.join(userDataDir, profile, "Extensions", id);
                try {
                    await access(extPath);
                    return true;
                } catch { }
            }
        }
    } catch (err) {
        logger.warn(`Failed to check for extension installation: ${err}`);
    }
    return false;
}

export async function installChromeExtensionBridge() {
    logger.info("Installing Chrome Native Host manifest components...");
    const manifestDir = getManifestPath();
    if (!manifestDir) {
        throw new Error(`Chrome Native Host not supported on platform: ${process.platform}`);
    }

    // 1. Create wrapper script
    const wrapperScriptPath = await createWrapperScript();

    // 2. Write Manifest
    const manifestFile = path.join(manifestDir, `${EXTENSION_ID}.json`);
    const manifestContent = {
        name: EXTENSION_ID,
        description: "Claude Code Browser Extension Native Host",
        path: wrapperScriptPath,
        type: "stdio",
        allowed_origins: [EXTENSION_ORIGIN]
    };

    const manifestJson = JSON.stringify(manifestContent, null, 2);

    // Check if update needed
    try {
        const existing = await readFile(manifestFile, "utf-8");
        if (existing === manifestJson) {
            logger.info("Manifest up to date.");
        } else {
            throw new Error("Update needed");
        }
    } catch {
        // Write it
        await mkdir(manifestDir, { recursive: true });
        await writeFile(manifestFile, manifestJson);
        if (process.platform === "win32") {
            await registerWindowsRegistry(manifestFile);
        }
        logger.info(`Installed Native Host manifest at ${manifestFile}`);
    }

    // Check if extension is installed to advise user
    const installed = await isExtensionInstalled();
    if (installed) {
        logger.info("Extension detected. Please open Chrome to use.");
        // We could open a URL to trigger reconnect if supported
        // open(RECONNECT_URL); 
    } else {
        logger.warn("Chrome extension not detected. Please install it from https://claude.ai/chrome");
    }
}

async function createWrapperScript() {
    const isWin = process.platform === "win32";
    // We need to point to the `claude` binary or `node cli.js` entrypoint
    // This logic mirrors `bw0` / `se2` in chunk_519.ts
    // For now we'll assume we are running from the source or dist

    let command = process.execPath;
    let args: string[] = [];

    // Heuristics to determine how to invoke ourselves as the native host
    // If running from node directly
    if (process.argv[1] && (process.argv[1].endsWith(".js") || process.argv[1].endsWith(".ts"))) {
        command = process.execPath;
        args = [process.argv[1], "--chrome-native-host"]; // Using --chrome-native-host as trigger
    } else {
        // Binary case
        command = process.execPath;
        args = ["--chrome-native-host"];
    }

    // In local dev, we might be `npm run start -- --chrome-native-host` equivalent
    // Ideally we assume the `claude` binary is in path or use absolute path
    const cliPath = path.resolve(process.argv[1]);
    const scriptDir = path.join(path.dirname(cliPath), 'chrome'); // .claude/chrome or similar
    await mkdir(scriptDir, { recursive: true });

    const wrapperName = isWin ? "chrome-native-host.bat" : "chrome-native-host";
    const wrapperPath = path.join(scriptDir, wrapperName);

    const fullCommand = `"${command}" ${args.map(a => `"${a}"`).join(" ")}`;

    const scriptContent = isWin
        ? `@echo off\nREM Chrome native host wrapper\n${fullCommand}\n`
        : `#!/bin/bash\n# Chrome native host wrapper\nexec ${fullCommand}\n`;

    try {
        const existing = await readFile(wrapperPath, "utf-8");
        if (existing === scriptContent) return wrapperPath;
    } catch { }

    await writeFile(wrapperPath, scriptContent);
    if (!isWin) await chmod(wrapperPath, 0o755);

    return wrapperPath;
}

async function registerWindowsRegistry(manifestPath: string) {
    const keyPath = `HKCU\\Software\\Google\\Chrome\\NativeMessagingHosts\\${EXTENSION_ID}`;
    try {
        await runProcess("reg", ["add", keyPath, "/ve", "/t", "REG_SZ", "/d", manifestPath, "/f"]);
    } catch (err) {
        logger.error(`Failed to register registry key: ${err}`);
    }
}

export function createMcpSocketBridge(serverName: string) {
    // Logic to create a bridge that the `ChromeNativeHost` (via NativeMessageStream) 
    // can talk to. 
    // This is effectively handled by `ChromeNativeHost` in `main.ts` managing the socket 
    // and `McpSocketClient` connecting to it.
    // This function might just verify the setup or return config.
    return {
        type: "socket",
        socketPath: process.platform === "win32" ? "\\\\.\\pipe\\claude-code" : "/tmp/claude-code.sock"
    };
}

export async function autoInstallOfficialPlugins() {
    // Logic from YA9 - installing plugins from github
    // This seems to be generic plugin installation logic
    logger.info("Checking official plugins...");
    // Stub for now, as specific git logic is complex
}
