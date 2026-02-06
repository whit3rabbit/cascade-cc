import { join } from 'node:path';
import { homedir, tmpdir } from 'node:os';

/**
 * Service for managing environment variables with validation and types.
 */
export interface EnvConfig {
    // Claude Specific
    CLAUDE_BASE_URL: string;
    CLAUDE_CONFIG_DIR: string;
    CLAUDE_LOG_LEVEL: string;
    CLAUDE_CHECK_FOR_UPDATES: boolean;
    CLAUDE_SESSION_ID: string;
    CLAUDE_PARENT_SESSION_ID: string;
    CLAUDE_AGENT_ID: string;
    CLAUDE_TEAMMATE: boolean;
    CLAUDE_BETA_FLAGS: string;
    CLAUDE_CODE_ENTRYPOINT: string;
    CLAUDE_CODE_CLIENT_TYPE: string;
    CLAUDE_CODE_DEMO: boolean;
    CLAUDE_CODE_INSTALL_METHOD: string;
    CLAUDE_API_KEY: string;
    CLAUDE_CODE_INTERACTIVE: boolean;
    CLAUDE_AGENT_SDK_VERSION: string;
    CLAUDE_CODE_BUBBLEWRAP: string;
    CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD: string;
    CLAUDE_CODE_API_KEY_HELPER_TTL_MS: number;
    CLAUDE_CODE_CLIENT_CERT: string;
    CLAUDE_CODE_CLIENT_KEY_PASSPHRASE: string;
    CLAUDE_CODE_CLIENT_KEY: string;
    CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS: boolean;
    CLAUDE_CODE_DISABLE_BACKGROUND_TASKS: boolean;
    CLAUDE_CODE_EXIT_AFTER_STOP_DELAY: number;
    CLAUDE_CODE_PROXY_RESOLVES_HOSTS: boolean;
    CLAUDE_CODE_TASK_LIST_ID: string;
    CLAUDE_CODE_TMPDIR: string;
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: boolean;
    CLAUDE_CODE_DISABLE_TERMINAL_TITLE: boolean;
    CLAUDE_CODE_ENABLE_TASKS: boolean;
    CLAUDE_CODE_ENABLE_TELEMETRY: boolean;
    CLAUDE_CODE_HIDE_ACCOUNT_INFO: boolean;
    CLAUDE_CODE_IDE_SKIP_AUTO_INSTALL: boolean;
    CLAUDE_CODE_MAX_OUTPUT_TOKENS: number;
    CLAUDE_CODE_SHELL: string;
    CLAUDE_CODE_SHELL_PREFIX: string;
    CLAUDE_CODE_SKIP_BEDROCK_AUTH: boolean;
    CLAUDE_CODE_SKIP_FOUNDRY_AUTH: boolean;
    CLAUDE_CODE_SKIP_VERTEX_AUTH: boolean;
    CLAUDE_CODE_USE_BEDROCK: boolean;
    CLAUDE_CODE_USE_FOUNDRY: boolean;
    CLAUDE_CODE_USE_VERTEX: boolean;
    IS_DEMO: boolean;
    USE_BUILTIN_RIPGREP: boolean;

    // MCP
    USE_MCP_CLI_DIR: string;
    MCP_TIMEOUT: number;
    MCP_TOOL_TIMEOUT: number;
    SLASH_COMMAND_TOOL_CHAR_BUDGET: number;
    ENABLE_TOOL_SEARCH: string; // "auto", "auto:N", "true", "false"

    // Anthropic Specific
    ANTHROPIC_BASE_URL: string;
    ANTHROPIC_API_KEY: string;
    ANTHROPIC_AUTH_TOKEN: string;
    ANTHROPIC_CUSTOM_HEADERS: string;
    ANTHROPIC_FOUNDRY_API_KEY: string;
    ANTHROPIC_FOUNDRY_BASE_URL: string;
    ANTHROPIC_FOUNDRY_RESOURCE: string;
    ANTHROPIC_MODEL: string;
    ANTHROPIC_SMALL_FAST_MODEL: string;
    ANTHROPIC_SMALL_FAST_MODEL_AWS_REGION: string;
    AWS_BEARER_TOKEN_BEDROCK: string;

    // System / Shell / NPM
    SHELL: string;
    TERM_PROGRAM: string;
    TERM: string;
    TERMINAL_EMULATOR: string;
    CI: boolean;
    GITHUB_ACTIONS: boolean;
    TMUX: string;
    ITERM_SESSION_ID: string;
    WSL_DISTRO_NAME: string;
    WSL_INTEROP: string;
    npm_package_version: string;
    BASH_DEFAULT_TIMEOUT_MS: number;
    BASH_MAX_OUTPUT_LENGTH: number;
    BASH_MAX_TIMEOUT_MS: number;
    HTTP_PROXY: string;
    HTTPS_PROXY: string;
    NO_PROXY: string;

    // IDEs
    PYCHARM_VM_OPTIONS: string;
    WEBIDE_VM_OPTIONS: string;
    IDEA_VM_OPTIONS: string;

    // SWE-Bench
    SWE_BENCH_RUN_ID: string;
    SWE_BENCH_INSTANCE_ID: string;
    SWE_BENCH_TASK_ID: string;

    // Debugging / Testing / Disabled Features
    DEBUG_TELEMETRY: boolean;
    DEBUG_UPDATER: boolean;
    CLAUDE_CODE_USE_FIXTURES: boolean;
    CLAUDE_CODE_TEST_FIXTURES_ROOT: string;
    CLAUDE_CODE_SKIP_PROMPT_HISTORY: boolean;
    CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING: boolean;
    CLAUDE_CODE_DISABLE_FILE_CHECKPOINTING: boolean;
    CLAUDE_CODE_TEAMMATE: boolean;
    CLAUDE_CODE_TEAM_NAME: string;
    CLAUDE_CODE_AGENT_NAME: string;
    CLAUDE_CODE_DONT_INHERIT_ENV: boolean;
    NODE_ENV: string;
    DISABLE_AUTOUPDATER: boolean;
    DISABLE_BUG_COMMAND: boolean;
    DISABLE_COST_WARNINGS: boolean;
    DISABLE_ERROR_REPORTING: boolean;
    DISABLE_INSTALLATION_CHECKS: boolean;
    DISABLE_NON_ESSENTIAL_MODEL_CALLS: boolean;
    DISABLE_TELEMETRY: boolean;
    FORCE_AUTOUPDATE_PLUGINS: boolean;

    // Model & Cache Configuration
    ANTHROPIC_DEFAULT_HAIKU_MODEL: string;
    ANTHROPIC_DEFAULT_SONNET_MODEL: string;
    ANTHROPIC_DEFAULT_OPUS_MODEL: string;
    CLAUDE_CODE_SUBAGENT_MODEL: string;
    VERTEX_REGION_CLAUDE_3_5_HAIKU: string;
    VERTEX_REGION_CLAUDE_3_7_SONNET: string;
    VERTEX_REGION_CLAUDE_4_0_OPUS: string;
    VERTEX_REGION_CLAUDE_4_0_SONNET: string;
    VERTEX_REGION_CLAUDE_4_1_OPUS: string;

    DISABLE_PROMPT_CACHING: boolean;
    DISABLE_PROMPT_CACHING_HAIKU: boolean;
    DISABLE_PROMPT_CACHING_SONNET: boolean;
    DISABLE_PROMPT_CACHING_OPUS: boolean;

    // Shadow List
    CLAUDE_AUTOCOMPACT_PCT_OVERRIDE: number;
    CLAUDE_CODE_OTEL_HEADERS_HELPER_DEBOUNCE_MS: number;
    CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS: number;
    MAX_MCP_OUTPUT_TOKENS: number;
    MAX_THINKING_TOKENS: number;

    // Paths
    PATH: string;
    APPDATA: string;
    LOCALAPPDATA: string;
    XDG_DATA_HOME: string;
    XDG_CACHE_HOME: string;
    XDG_CONFIG_HOME: string;

    // General purpose for deobfuscated code
    [key: string]: any;
}

export class EnvService {
    private static config: Partial<EnvConfig> = {};

    static get(key: keyof EnvConfig | string): any {
        // @ts-ignore
        if (this.config[key] !== undefined) return this.config[key];

        const value = process.env[key];

        // Handle defaults and fallbacks for Claude/Anthropic keys
        if (value === undefined || value === '') {
            switch (key) {
                case 'CLAUDE_BASE_URL':
                case 'ANTHROPIC_BASE_URL':
                    return 'https://api.anthropic.com';
                case 'CLAUDE_CONFIG_DIR':
                    return process.env.CLAUDE_CONFIG_DIR || join(homedir(), '.claude');
                case 'CLAUDE_LOG_LEVEL':
                    return 'info';
                case 'SHELL':
                    return process.env.SHELL || (process.platform === 'win32' ? 'cmd.exe' : '/bin/bash');
                case 'CLAUDE_CODE_ENTRYPOINT':
                    return 'claude-3-5-sonnet-latest';
                case 'CLAUDE_CODE_CLIENT_TYPE':
                    return 'cli';
                case 'CLAUDE_SESSION_ID':
                    return `session_${process.pid}_${Date.now()}`;
                case 'CLAUDE_CODE_TEST_FIXTURES_ROOT':
                    return join(process.cwd(), "tests", "fixtures");
                case 'TERM':
                    return process.env.TERM || 'xterm-256color';
                case 'USE_MCP_CLI_DIR':
                    return join(tmpdir(), "claude-code-mcp-cli");
                case 'CLAUDE_CODE_ENABLE_TASKS':
                    return true;
                case 'CLAUDE_CODE_MAX_OUTPUT_TOKENS':
                    return 32000;
                case 'CLAUDE_CODE_OTEL_HEADERS_HELPER_DEBOUNCE_MS':
                    return 1740000; // 29 minutes
                case 'ENABLE_TOOL_SEARCH':
                    return 'auto';
                case 'MAX_MCP_OUTPUT_TOKENS':
                    return 25000;
                case 'MAX_THINKING_TOKENS':
                    return 31999;
                case 'SLASH_COMMAND_TOOL_CHAR_BUDGET':
                    return 15000;
                case 'BASH_MAX_OUTPUT_LENGTH':
                    return 600000;
                default:
                    return undefined;
            }
        }

        // Type conversion for booleans
        const booleanKeys = [
            'CLAUDE_CHECK_FOR_UPDATES',
            'CLAUDE_TEAMMATE',
            'CLAUDE_CODE_DEMO',
            'CLAUDE_CODE_INTERACTIVE',
            'DEBUG_TELEMETRY',
            'DEBUG_UPDATER',
            'CLAUDE_CODE_USE_FIXTURES',
            'CI',
            'GITHUB_ACTIONS',
            'CLAUDE_CODE_SKIP_PROMPT_HISTORY',
            'CLAUDE_CODE_DONT_INHERIT_ENV',
            'CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING',
            'CLAUDE_CODE_DISABLE_FILE_CHECKPOINTING',
            'DISABLE_PROMPT_CACHING',
            'DISABLE_PROMPT_CACHING_HAIKU',
            'DISABLE_PROMPT_CACHING_SONNET',
            'DISABLE_PROMPT_CACHING_OPUS',
            'CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS',
            'CLAUDE_CODE_DISABLE_BACKGROUND_TASKS',
            'CLAUDE_CODE_PROXY_RESOLVES_HOSTS',
            'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC',
            'CLAUDE_CODE_DISABLE_TERMINAL_TITLE',
            'CLAUDE_CODE_ENABLE_TASKS',
            'CLAUDE_CODE_ENABLE_TELEMETRY',
            'CLAUDE_CODE_HIDE_ACCOUNT_INFO',
            'CLAUDE_CODE_IDE_SKIP_AUTO_INSTALL',
            'CLAUDE_CODE_SKIP_BEDROCK_AUTH',
            'CLAUDE_CODE_SKIP_FOUNDRY_AUTH',
            'CLAUDE_CODE_SKIP_VERTEX_AUTH',
            'CLAUDE_CODE_USE_BEDROCK',
            'CLAUDE_CODE_USE_FOUNDRY',
            'CLAUDE_CODE_USE_VERTEX',
            'IS_DEMO',
            'USE_BUILTIN_RIPGREP',
            'DISABLE_AUTOUPDATER',
            'DISABLE_BUG_COMMAND',
            'DISABLE_COST_WARNINGS',
            'DISABLE_ERROR_REPORTING',
            'DISABLE_INSTALLATION_CHECKS',
            'DISABLE_NON_ESSENTIAL_MODEL_CALLS',
            'DISABLE_TELEMETRY',
            'FORCE_AUTOUPDATE_PLUGINS',
        ];

        if (booleanKeys.includes(key as string)) {
            return value === 'true' || value === '1' || value === 'yes';
        }

        return value;
    }

    static getRequired(key: keyof EnvConfig | string): string {
        const val = this.get(key);
        if (val === undefined || val === '') {
            throw new Error(`Environment variable ${key} is required but not set.`);
        }
        return val as string;
    }

    static isTruthy(key: keyof EnvConfig | string): boolean {
        const val = this.get(key);
        return val === true || val === 'true' || val === '1' || val === 'yes';
    }

    /**
     * Set a value in the environment (updates process.env as well).
     */
    static set(key: string, value: any): void {
        // @ts-ignore
        this.config[key] = value;
        if (value === undefined) {
            delete process.env[key];
        } else {
            process.env[key] = String(value);
        }
    }

    /**
     * Parse the ENABLE_TOOL_SEARCH setting and return the percentage if it's in auto:N format.
     */
    static getToolSearchAutoPercentage(value: string): number | null {
        if (!value.startsWith("auto:")) {
            return null;
        }
        const percentageStr = value.slice(5);
        const percentage = parseInt(percentageStr, 10);
        if (isNaN(percentage)) {
            console.warn(`[Config] Invalid ENABLE_TOOL_SEARCH value "${value}": expected auto:N where N is a number.`);
            return null;
        }
        return Math.max(0, Math.min(100, percentage));
    }

    /**
     * Returns the token threshold for tool search based on context window.
     */
    static getToolSearchThreshold(contextWindow: number): number {
        const setting = this.get('ENABLE_TOOL_SEARCH') || 'auto';
        if (setting === 'auto') {
            return Math.floor(contextWindow * 0.1); // Default 10%
        }
        const percentage = this.getToolSearchAutoPercentage(setting);
        if (percentage !== null) {
            return Math.floor(contextWindow * (percentage / 100));
        }
        return Math.floor(contextWindow * 0.1); // Fallback to 10%
    }

    /**
     * Returns the character threshold for tool search based on token threshold.
     */
    static getToolSearchCharThreshold(tokenThreshold: number): number {
        // Based on reference code discovery: nW2 = 2.5
        return Math.floor(tokenThreshold * 2.5);
    }
}

