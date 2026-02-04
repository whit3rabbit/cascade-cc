/**
 * File: src/utils/shared/runtimeAndEnv.ts
 * Role: Centralized process, environment, and runtime state management.
 */

import { join } from 'node:path';
import { homedir } from 'node:os';
import { readFileSync, existsSync } from 'node:fs';
import { getProductName, getSystemUser } from './product.js';
import { KeychainService } from '../../services/auth/KeychainService.js';
import { EnvService } from '../../services/config/EnvService.js';

export { getProductName, getSystemUser, KeychainService };

/**
 * Returns the content of the configured JS environment loading script.
 * Checks CLAUDE_ENV_FILE and optionally session-env hooks (simplified here).
 */
export function getSessionEnvScript(): string | null {
    const envFile = EnvService.get('CLAUDE_ENV_FILE');
    if (envFile && existsSync(envFile)) {
        try {
            const content = readFileSync(envFile, 'utf8').trim();
            if (content) return content;
        } catch (error) {
            console.error(`Failed to read CLAUDE_ENV_FILE: ${envFile}`, error);
        }
    }
    return null;
}

/**
 * Normalizes a value to a boolean.
 */
export function toBoolean(value: any): boolean {
    if (typeof value === 'boolean') return value;
    if (typeof value === 'string') {
        const v = value.toLowerCase();
        return v === 'true' || v === '1' || v === 'yes';
    }
    return !!value;
}

/**
 * Returns the base configuration directory.
 */
export function getBaseConfigDir(): string {
    return EnvService.get('CLAUDE_CONFIG_DIR');
}

/**
 * Returns the directory for chat history.
 */
export function getChatHistoryDir(): string {
    return join(getBaseConfigDir(), 'chat');
}

/**
 * Checks if running in a CI environment.
 */
export function isCI(): boolean {
    return EnvService.isTruthy("CI") || EnvService.isTruthy("GITHUB_ACTIONS") || !!process.env.TRAVIS || !!process.env.CIRCLECI;
}

/**
 * Returns the current session ID.
 */
export function getSessionId(): string {
    return EnvService.get("CLAUDE_SESSION_ID");
}

/**
 * Returns the parent session ID if applicable.
 */
export function getParentSessionId(): string | undefined {
    return EnvService.get("CLAUDE_PARENT_SESSION_ID");
}

/**
 * Returns the agent context.
 */
export function getAgentContext(): any {
    return {
        agentId: EnvService.get("CLAUDE_AGENT_ID"),
        parentSessionId: getParentSessionId(),
        agentType: isTeammate() ? "teammate" : (EnvService.get("CLAUDE_AGENT_ID") ? "standalone" : "cli")
    };
}

/**
 * Checks if the user is a teammate (via env flag).
 */
export function isTeammate(): boolean {
    return EnvService.isTruthy("CLAUDE_TEAMMATE");
}

/**
 * Returns the agent ID.
 */
export function getAgentId(): string | undefined {
    return EnvService.get("CLAUDE_AGENT_ID");
}

/**
 * Returns beta flags for the given model.
 */
export function getBetaFlags(model?: string): string[] {
    // Returns empty or comma-separated list of flags from env
    const flags = EnvService.get("CLAUDE_BETA_FLAGS") || "";
    return flags.split(',').filter(Boolean);
}

/**
 * Returns the entrypoint model name.
 */
export function getEntrypoint(): string {
    return EnvService.get("CLAUDE_CODE_ENTRYPOINT");
}

/**
 * Returns general environment context.
 */
export async function getEnvContext(): Promise<any> {
    return {
        platform: process.platform,
        arch: process.arch,
        nodeVersion: process.version,
        isCI: isCI()
    };
}

/**
 * Returns the client type for Claude Code.
 */
export function getClaudeCodeClientType(): string {
    return EnvService.get("CLAUDE_CODE_CLIENT_TYPE");
}

/**
 * Returns the platform-specific data directory.
 */
export function getDataDir(): string {
    const home = homedir();
    switch (process.platform) {
        case 'darwin': return join(home, 'Library', 'Application Support');
        case 'win32': return EnvService.get("APPDATA") || join(home, 'AppData', 'Roaming');
        default: return EnvService.get("XDG_DATA_HOME") || join(home, '.local', 'share');
    }
}

/**
 * Returns the platform-specific cache directory.
 */
export function getCacheDir(): string {
    const home = homedir();
    switch (process.platform) {
        case 'darwin': return join(home, 'Library', 'Caches');
        case 'win32': return EnvService.get("LOCALAPPDATA") || join(home, 'AppData', 'Local');
        default: return EnvService.get("XDG_CACHE_HOME") || join(home, '.cache');
    }
}

/**
 * Returns the platform-specific configuration directory.
 */
export function getConfigDir(): string {
    const home = homedir();
    switch (process.platform) {
        case 'darwin': return join(home, 'Library', 'Application Support');
        case 'win32': return EnvService.get("APPDATA") || join(home, 'AppData', 'Roaming');
        default: return EnvService.get("XDG_CONFIG_HOME") || join(home, '.config');
    }
}

/**
 * Returns the platform-specific directory for enterprise/policy settings.
 */
export function getPolicySettingsDirectory(): string {
    switch (process.platform) {
        case 'darwin':
            return '/Library/Application Support/ClaudeCode';
        case 'win32':
            // Check both potential locations on Windows
            return EnvService.get("PROGRAMFILES")
                ? join(EnvService.get("PROGRAMFILES"), 'ClaudeCode')
                : 'C:\\ProgramData\\ClaudeCode';
        default:
            return '/etc/claude-code';
    }
}


/**
 * Returns the path to the enterprise MCP configuration file.
 */
export function getEnterpriseMcpConfigPath(): string {
    return join(getPolicySettingsDirectory(), 'mcp.json');
}

/**
 * Returns the directory for managed rules/instructions.
 */
export function getManagedRulesDirectory(): string {
    return join(getPolicySettingsDirectory(), 'rules');
}

/**
 * Returns the platform-specific host identifier (e.g., 'darwin-arm64').
 */
export function getHostPlatform(): string {
    const arch = process.arch === "x64" ? "x64" : process.arch === "arm64" ? "arm64" : process.arch;
    return `${process.platform}-${arch}`;
}

/**
 * Returns the paths for various Claude-related system files.
 */
export function getClaudePaths() {
    const platformName = getHostPlatform();
    const executableName = process.platform === "win32" ? "claude.exe" : "claude";

    return {
        versions: join(getDataDir(), "claude", "versions"),
        staging: join(getCacheDir(), "claude", "staging"),
        locks: join(getConfigDir(), "claude", "locks"),
        executable: join(getDataDir(), "claude", "bin", executableName),
    };
}

/**
 * Returns the installation method.
 */
export function getInstallMethod(): string {
    if (EnvService.get("CLAUDE_CODE_INSTALL_METHOD")) return EnvService.get("CLAUDE_CODE_INSTALL_METHOD");
    const mainFile = require.main?.filename;
    if (mainFile?.includes('node_modules')) return 'npm';
    return 'native';
}

/**
 * Gets a string from an environment variable.
 */
export function getStringFromEnv(key: string): string | undefined {
    return EnvService.get(key);
}

/**
 * Gets a parsed number from an environment variable.
 */
export function getNumberFromEnv(key: string): number | undefined {
    const val = EnvService.get(key);
    if (!val) return undefined;
    const num = Number(val);
    return isNaN(num) ? undefined : num;
}

/**
 * Parses a comma-separated key=value string into a record.
 */
export function parseKeyPairsIntoRecord(input: string | undefined): Record<string, string> {
    if (!input) return {};
    const record: Record<string, string> = {};
    input.split(',').forEach(pair => {
        const [key, value] = pair.split('=').map(s => s.trim());
        if (key && value) {
            record[key] = value;
        }
    });
    return record;
}

/**
 * Parsing logic for baggage metadata.
 */
export function baggageEntryMetadataFromString(str: string): any {
    return { toString: () => str };
}

/**
 * Creates a baggage object.
 */
export function createBaggage(entries: Record<string, any>): any {
    return {
        getAllEntries: () => Object.entries(entries).map(([k, v]) => [k, v]),
        getEntry: (key: string) => entries[key]
    };
}

/**
 * Propagation stub.
 */
export const propagation = {
    getBaggage: (context: any) => context.baggage,
    setBaggage: (context: any, baggage: any) => ({ ...context, baggage })
};

/**
 * Checks if running inside a Bubblewrap sandbox.
 */
export function isBubblewrapSandbox(): boolean {
    return process.platform === "linux" && EnvService.get("CLAUDE_CODE_BUBBLEWRAP") === "1";
}

/**
 * Checks if running inside a Docker container.
 */
export function isDocker(): boolean {
    if (existsSync("/.dockerenv")) {
        return true;
    }
    return false;
}

/**
 * Checks if running in a Musl environment (Alpine, etc.).
 */
export function isMuslEnvironment(): boolean {
    if (process.platform !== "linux") {
        return false;
    }
    try {
        if (existsSync("/lib/libc.musl-x86_64.so.1") || existsSync("/lib/libc.musl-aarch64.so.1")) {
            return true;
        }
        // Fallback: check ldd output if available (simplified check)
        return false;
    } catch {
        return false;
    }
}
