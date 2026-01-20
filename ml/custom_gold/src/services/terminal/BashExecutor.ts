
import { execFile, spawn } from 'node:child_process';
import { existsSync, mkdirSync, realpathSync, writeFileSync, unlinkSync, statSync } from 'node:fs';
import { join, dirname, isAbsolute, resolve } from 'node:path';
import { homedir, tmpdir } from 'node:os';
import { setCwd } from './sessionService.js';
import { logTelemetryEvent } from '../telemetry/telemetryInit.js';
import { getConfigDir } from '../../utils/shared/pathUtils.js';

const SNAPSHOT_TIMEOUT_MS = 10000;
const MAX_SNAPSHOT_BUFFER = 1024 * 1024; // 1MB

/**
 * Robust shell detection.
 * Matches Gs5 from chunk_496.ts.
 */
function getShellExecutable(): string {
    const envShell = process.env.CLAUDE_CODE_SHELL;
    if (envShell && existsSync(envShell)) return envShell;

    const shell = process.env.SHELL;
    if (shell && (shell.includes("bash") || shell.includes("zsh")) && existsSync(shell)) {
        return shell;
    }

    const fallbacks = ["/bin/zsh", "/bin/bash", "/usr/bin/zsh", "/usr/bin/bash", "/bin/sh"];
    for (const shell of fallbacks) {
        if (existsSync(shell)) return shell;
    }
    return "bash";
}

/**
 * Gets the path to the shell config file (~/.zshrc or ~/.bashrc).
 * Matches R$0 from chunk_495.ts.
 */
function getShellConfigPath(shellExecutable: string): string {
    const configName = shellExecutable.includes("zsh") ? ".zshrc" : shellExecutable.includes("bash") ? ".bashrc" : ".profile";
    return join(homedir(), configName);
}

/**
 * Generates the script to dump aliases, functions and options.
 * Matches dr5 from chunk_495.ts.
 */
function getFunctionAndOptionDumpingScript(shellExecutable: string, configPath: string): string {
    const isZsh = configPath.endsWith(".zshrc");
    let script = "";

    if (isZsh) {
        script += `
            echo "# Functions" >> "$SNAPSHOT_FILE"
            # Force autoload all functions first
            typeset -f > /dev/null 2>&1
            # Get user function names - filter system ones
            typeset +f | grep -vE '^(_|__)' | while read func; do
                typeset -f "$func" >> "$SNAPSHOT_FILE"
            done
        `;
    } else {
        script += `
            echo "# Functions" >> "$SNAPSHOT_FILE"
            # Force autoload all functions first
            declare -f > /dev/null 2>&1
            # Get user function names - filter system ones and use base64 for safety
            declare -F | cut -d' ' -f3 | grep -vE '^(_|__)' | while read func; do
                encoded_func=$(declare -f "$func" | base64)
                echo "eval \\"\\$(echo '$encoded_func' | base64 -d)\\" > /dev/null 2>&1" >> "$SNAPSHOT_FILE"
            done
        `;
    }

    if (isZsh) {
        script += `
            echo "# Shell Options" >> "$SNAPSHOT_FILE"
            setopt | sed 's/^/setopt /' | head -n 1000 >> "$SNAPSHOT_FILE"
        `;
    } else {
        script += `
            echo "# Shell Options" >> "$SNAPSHOT_FILE"
            shopt -p | head -n 1000 >> "$SNAPSHOT_FILE"
            set -o | grep "on" | awk '{print "set -o " $1}' | head -n 1000 >> "$SNAPSHOT_FILE"
            echo "shopt -s expand_aliases" >> "$SNAPSHOT_FILE"
        `;
    }

    script += `
        echo "# Aliases" >> "$SNAPSHOT_FILE"
        if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
            alias | grep -v "='winpty " | sed 's/^alias //g' | sed 's/^/alias -- /' | head -n 1000 >> "$SNAPSHOT_FILE"
        else
            alias | sed 's/^alias //g' | sed 's/^/alias -- /' | head -n 1000 >> "$SNAPSHOT_FILE"
        fi
    `;

    return script;
}

/**
 * Detects utility binary paths for snapshots.
 * Matches cr5 from chunk_495.ts.
 */
function getUtilityAliasesScript(): string {
    // These should ideally come from actual detection services
    const rgPath = "/usr/local/bin/rg"; // Placeholder for mr5()
    const mcpCliPath = process.execPath; // Placeholder for pr5()

    let script = `
        # Check for rg availability
        echo "# Check for rg availability" >> "$SNAPSHOT_FILE"
        echo "if ! command -v rg >/dev/null 2>&1; then" >> "$SNAPSHOT_FILE"
        echo "  alias rg='${rgPath}'" >> "$SNAPSHOT_FILE"
        echo "fi" >> "$SNAPSHOT_FILE"

        # Check for mcp-cli availability
        echo "# Check for mcp-cli availability" >> "$SNAPSHOT_FILE"
        echo "if ! command -v mcp-cli >/dev/null 2>&1; then" >> "$SNAPSHOT_FILE"
        echo "  alias mcp-cli='${mcpCliPath} --mcp-cli'" >> "$SNAPSHOT_FILE"
        echo "fi" >> "$SNAPSHOT_FILE"

        # Add PATH to the file
        echo "export PATH=\\"$PATH\\"" >> "$SNAPSHOT_FILE"
    `;

    return script;
}

/**
 * Generates the full snapshot generation script.
 * Matches lr5 from chunk_496.ts.
 */
function generateSnapshotScript(shellExecutable: string, snapshotFile: string, userConfigExists: boolean): string {
    const configPath = getShellConfigPath(shellExecutable);
    const dumpScript = userConfigExists ? getFunctionAndOptionDumpingScript(shellExecutable, configPath) :
        (shellExecutable.includes("zsh") ? "" : 'echo "shopt -s expand_aliases" >> "$SNAPSHOT_FILE"');
    const utilityScript = getUtilityAliasesScript();

    return `
        SNAPSHOT_FILE="${snapshotFile}"
        ${userConfigExists ? `source "${configPath}" < /dev/null` : "# No user config file to source"}

        # First, create/clear the snapshot file
        echo "# Snapshot file" >| "$SNAPSHOT_FILE"

        # When this file is sourced, we first unalias to avoid conflicts
        echo "# Unset all aliases to avoid conflicts with functions" >> "$SNAPSHOT_FILE"
        echo "unalias -a 2>/dev/null || true" >> "$SNAPSHOT_FILE"

        ${dumpScript}
        ${utilityScript}

        # Exit silently on success
        if [ ! -f "$SNAPSHOT_FILE" ]; then
            echo "Error: Snapshot file was not created at $SNAPSHOT_FILE" >&2
            exit 1
        fi
    `;
}

/**
 * Creates a shell snapshot for the current user environment.
 * Matches dc2 from chunk_496.ts.
 */
export async function createShellSnapshot(shellExecutable: string): Promise<string | undefined> {
    const configPath = getShellConfigPath(shellExecutable);
    const configExists = existsSync(configPath);
    const timestamp = Date.now();
    const randomId = Math.random().toString(36).substring(2, 8);
    const snapshotDir = join(getConfigDir(), "shell-snapshots");
    const snapshotFile = join(snapshotDir, `snapshot-${timestamp}-${randomId}.sh`);

    if (!existsSync(snapshotDir)) {
        mkdirSync(snapshotDir, { recursive: true });
    }

    const script = generateSnapshotScript(shellExecutable, snapshotFile, configExists);

    return new Promise((resolve) => {
        execFile(shellExecutable, ["-c", "-l", script], {
            env: {
                ...process.env,
                SHELL: shellExecutable,
                CLAUDECODE: "1"
            },
            timeout: SNAPSHOT_TIMEOUT_MS,
            maxBuffer: MAX_SNAPSHOT_BUFFER,
            encoding: "utf8"
        }, (error) => {
            if (error) {
                logTelemetryEvent("tengu_shell_snapshot_failed", { error: error.message });
                resolve(undefined);
            } else if (existsSync(snapshotFile)) {
                // Return path, but it should be cleaned up later
                resolve(snapshotFile);
            } else {
                resolve(undefined);
            }
        });
    });
}

/**
 * Memoized shell info.
 */
let cachedShellInfo: { binShell: string; snapshotFilePath?: string } | null = null;
async function getShellInfo() {
    if (cachedShellInfo) return cachedShellInfo;
    const binShell = getShellExecutable();
    const snapshotFilePath = await createShellSnapshot(binShell);
    cachedShellInfo = { binShell, snapshotFilePath };
    return cachedShellInfo;
}

/**
 * Executes a bash command in a persistent shell environment.
 * Matches dW1 from chunk_496.ts.
 */
export async function runBashCommand(commandText: string, options: any = {}) {
    const { binShell, snapshotFilePath } = await getShellInfo();
    const shell = options.shellExecutable || binShell;

    const randomId = Math.floor(Math.random() * 65536).toString(16).padStart(4, "0");
    const cwdFile = join(tmpdir(), `claude-${randomId}-cwd`);

    const bootScripts: string[] = [];
    if (snapshotFilePath && existsSync(snapshotFilePath)) {
        bootScripts.push(`source "${snapshotFilePath}"`);
    }

    // Ensure aliases are enabled for non-interactive shells
    if (shell.includes("bash")) {
        bootScripts.push("shopt -s expand_aliases 2>/dev/null || true");
    }

    bootScripts.push(`eval "${commandText.replace(/"/g, '\\"')}"`);
    bootScripts.push(`pwd -P >| "${cwdFile}"`);

    const fullScript = bootScripts.join(" && ");
    const args = ["-c", "-l", fullScript];

    const child = spawn(shell, args, {
        cwd: process.cwd(),
        env: {
            ...process.env,
            SHELL: shell,
            CLAUDECODE: "1"
        },
        detached: true
    });

    // In a real implementation we'd handle output streams and promise resolution
    // For brevity, we just manage the process
    return {
        child,
        result: new Promise((resolve) => {
            let stdout = "";
            let stderr = "";
            let totalLines = 0;

            child.stdout?.on('data', d => {
                const chunk = d.toString();
                stdout += chunk;
                totalLines += chunk.split('\n').length - 1;
                if (options.onOutput) {
                    options.onOutput(chunk, stdout, totalLines);
                }
            });
            child.stderr?.on('data', d => {
                const chunk = d.toString();
                stderr += chunk;
                if (options.onOutput) {
                    options.onOutput(chunk, stdout, totalLines);
                }
            });
            child.on('close', (code) => {
                let currentCwd = process.cwd();
                if (existsSync(cwdFile)) {
                    try {
                        currentCwd = statSync(cwdFile).size > 0 ? realpathSync(cwdFile) : currentCwd;
                        setCwd(currentCwd);
                        unlinkSync(cwdFile);
                    } catch { }
                }
                // Final update
                if (options.onOutput) {
                    options.onOutput("", stdout, totalLines);
                }
                resolve({ code, stdout, stderr, interrupted: false });
            });
            child.on('error', (err) => {
                resolve({ code: 1, stdout, stderr: err.message, interrupted: false });
            });
        })
    };
}
