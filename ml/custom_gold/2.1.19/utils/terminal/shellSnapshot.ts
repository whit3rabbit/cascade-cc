/**
 * File: src/utils/terminal/shellSnapshot.ts
 * Role: UTILITY_HELPER
 * Generates and manages shell configuration snapshots.
 */

import { existsSync, statSync, mkdirSync, realpathSync, unlinkSync, readdirSync } from "node:fs";
import { join as joinPath } from "node:path";
import * as os from "node:os";
import { fileExists, getHomeDir, getRelativeFilePath, resolvePathFromHome } from "../fs/paths.js";
import { onCleanup } from "../cleanup.js";
import { getPlatform, isWsl } from "../platform/detector.js";
import { getGitBashPath } from "../platform/shell.js";
import { shellQuote as R4 } from "./shellQuote.js";
import { terminalLog, errorLog, m1 as getCwd } from "../shared/runtime.js";
import { spawnBashCommand, executeBashCommand } from "../shared/bashUtils.js";
import { EnvService } from "../../services/config/EnvService.js";

const SHELL_SNAPSHOT_TIMEOUT = 10000;
const base64EncodedString = "\\\\";

/**
 * Gets the rip-grep alias or function definition.
 */
function getRipgrepAliasOrFunction() {
    // In gold, this queries a ripgrep config provider. 
    // For now, we'll check common paths and environment.
    const useBuiltin = EnvService.get("CLAUDE_CODE_USE_BUILTIN_RIPGREP") === "true";
    const rgPath = useBuiltin ? "rg" : "/usr/bin/rg";

    // Check if it's actually bundled or system
    const isBundled = useBuiltin;

    if (isBundled) {
        return {
            type: "function",
            snippet: [
                "function rg {",
                "  if [[ -n $ZSH_VERSION ]]; then",
                `    ARGV0=rg ${R4([rgPath])} "$@"`,
                "  elif [[ $BASHPID != $$ ]]; then",
                `    exec -a rg ${R4([rgPath])} "$@"`,
                "  else",
                `    (exec -a rg ${R4([rgPath])} "$@")`,
                "  fi",
                "}"
            ].join('\n')
        };
    }

    return {
        type: "alias",
        snippet: R4([rgPath]),
    };
}

/**
 * Determines the shell configuration file path based on the shell type.
 */
function getShellConfigFilePath(shellPath: string): string {
    const shellName = shellPath.includes("zsh") ? ".zshrc" : shellPath.includes("bash") ? ".bashrc" : ".profile";
    return joinPath(os.homedir(), shellName);
}

/**
 * Constructs the shell script to set up environment variables and aliases.
 */
async function buildShellSetupScript(shellPath: string) {
    let envPath = EnvService.get("PATH") || "";
    if (getPlatform() === "windows") {
        const result = await executeBashCommand("echo $PATH", {
            shell: "bash.exe",
        } as any);
        if (result.exitCode === 0 && result.stdout) {
            envPath = result.stdout.trim();
        }
    }

    const ripgrep = getRipgrepAliasOrFunction();
    const cliPath = getCliPath();

    let script = `
      # Check for rg availability
      echo "# Check for rg availability" >> "$SNAPSHOT_FILE"
      echo "if ! (unalias rg 2>/dev/null; command -v rg) >/dev/null 2>&1; then" >> "$SNAPSHOT_FILE"
  `;

    if (ripgrep.type === "function") {
        script += `
      cat >> "$SNAPSHOT_FILE" << 'RIPGREP_FUNC_END'
${ripgrep.snippet}
RIPGREP_FUNC_END
    `;
    } else {
        script += `
      echo '  alias rg='"'${ripgrep.snippet.replace(/'/g, "'\\''")}'" >> "$SNAPSHOT_FILE"
    `;
    }

    script += `
      echo "fi" >> "$SNAPSHOT_FILE"
  `;

    if (cliPath) {
        const escapedCliPath = R4([cliPath.cliPath]);
        const escapedArgs = cliPath.args.map(arg => R4([arg])).join(" ");
        const mcpCliAlias = `${escapedCliPath} ${escapedArgs}`;

        script += `
      # Check for mcp-cli availability
      echo "# Check for mcp-cli availability" >> "$SNAPSHOT_FILE"
      echo "if ! command -v mcp-cli >/dev/null 2>&1; then" >> "$SNAPSHOT_FILE"
      echo '  alias mcp-cli='"'${mcpCliAlias.replace(/'/g, "'\\''")}'" >> "$SNAPSHOT_FILE"
      echo "fi" >> "$SNAPSHOT_FILE"
    `;
    }

    script += `
      # Add PATH to the file
      echo "export PATH=${R4([envPath])}" >> "$SNAPSHOT_FILE"
  `;

    // Add function and alias harvesting logic from gold
    const isZsh = shellPath.includes("zsh");
    if (isZsh) {
        script += `
      echo "# Functions" >> "$SNAPSHOT_FILE"
      typeset -f > /dev/null 2>&1
      typeset +f | grep -vE '^(_|__)' | while read func; do
        typeset -f "$func" >> "$SNAPSHOT_FILE"
      done
      echo "# Shell Options" >> "$SNAPSHOT_FILE"
      setopt | sed 's/^/setopt /' | head -n 1000 >> "$SNAPSHOT_FILE"
      echo "# Aliases" >> "$SNAPSHOT_FILE"
      alias | sed 's/^alias //g' | sed 's/^/alias -- /' | head -n 1000 >> "$SNAPSHOT_FILE"
    `;
    } else {
        script += `
      echo "# Functions" >> "$SNAPSHOT_FILE"
      declare -f > /dev/null 2>&1
      declare -F | cut -d' ' -f3 | grep -vE '^(_|__)' | while read func; do
        declare -f "$func" >> "$SNAPSHOT_FILE"
      done
      echo "# Shell Options" >> "$SNAPSHOT_FILE"
      shopt -p | head -n 1000 >> "$SNAPSHOT_FILE"
      set -o | grep "on" | awk '{print "set -o " $1}' | head -n 1000 >> "$SNAPSHOT_FILE"
      echo "shopt -s expand_aliases" >> "$SNAPSHOT_FILE"
      echo "# Aliases" >> "$SNAPSHOT_FILE"
      if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        alias | grep -v "='winpty " | sed 's/^alias //g' | sed 's/^/alias -- /' | head -n 1000 >> "$SNAPSHOT_FILE"
      else
        alias | sed 's/^alias //g' | sed 's/^/alias -- /' | head -n 1000 >> "$SNAPSHOT_FILE"
      fi
    `;
    }

    return script;
}

/**
 * Determines the cli path for the current process.
 */
function getCliPath() {
    if (!isWsl()) return null;
    try {
        const gitBashPath = getGitBashPath();
        let cliPath = gitBashPath ? process.execPath : process.argv[1];
        if (!cliPath) return null;
        try {
            cliPath = realpathSync(cliPath);
        } catch { }
        if (getPlatform() === "windows") {
            cliPath = resolvePathFromHome(cliPath);
        }
        return { cliPath, args: ["--mcp-cli"] };
    } catch (error) {
        errorLog(error instanceof Error ? error : Error(String(error)));
        return null;
    }
}

/**
 * Builds the full snapshot script.
 */
async function buildShellSnapshotScript(shellPath: string, snapshotFilePath: string, shouldSourceConfig: boolean) {
    const shellConfigFile = getShellConfigFilePath(shellPath);
    const isZsh = shellConfigFile.endsWith(".zshrc");
    const sourceCommand = shouldSourceConfig ? `source "${shellConfigFile}" < /dev/null` : "# No user config file to source";
    const shellSetupScript = await buildShellSetupScript(shellPath);

    return `
    SNAPSHOT_FILE=${R4([snapshotFilePath])}
    ${sourceCommand}

    # First, create/clear the snapshot file
    echo "# Snapshot file" >| "$SNAPSHOT_FILE"

    # When this file is sourced, we first unalias to avoid conflicts with functions
    echo "# Unset all aliases to avoid conflicts with functions" >> "$SNAPSHOT_FILE"
    echo "unalias -a 2>/dev/null || true" >> "$SNAPSHOT_FILE"

    ${shouldSourceConfig && !isZsh ? 'echo "shopt -s expand_aliases" >> "$SNAPSHOT_FILE"' : ""}

    ${shellSetupScript}

    # Exit silently on success, only report errors
    if [ ! -f "$SNAPSHOT_FILE" ]; then
      echo "Error: Snapshot file was not created at $SNAPSHOT_FILE" >&2
      exit 1
    fi
  `;
}

/**
 * Creates a shell snapshot, capturing shell configuration and environment.
 */
export async function createShellSnapshot(shellPath: string): Promise<string | undefined> {
    const shellType = shellPath.includes("zsh") ? "zsh" : shellPath.includes("bash") ? "bash" : "sh";
    terminalLog(`Creating shell snapshot for ${shellType} (${shellPath})`);

    return new Promise(async (resolve) => {
        try {
            const shellConfigFile = getShellConfigFilePath(shellPath);
            const shouldSourceConfig = fileExists(shellConfigFile);

            const timestamp = Date.now();
            const randomString = Math.random().toString(36).substring(2, 8);
            const snapshotsDir = getRelativeFilePath(getHomeDir(), "shell-snapshots");
            const snapshotFilePath = getRelativeFilePath(snapshotsDir, `snapshot-${shellType}-${timestamp}-${randomString}.sh`);

            mkdirSync(snapshotsDir, { recursive: true });

            const snapshotScript = await buildShellSnapshotScript(shellPath, snapshotFilePath, shouldSourceConfig);

            spawnBashCommand(shellPath, ["-c", "-l", snapshotScript], {
                env: {
                    ...(EnvService.isTruthy("CLAUDE_CODE_DONT_INHERIT_ENV") ? {} : process.env),
                    SHELL: shellPath,
                    GIT_EDITOR: "true",
                    CLAUDECODE: "1",
                },
                timeout: SHELL_SNAPSHOT_TIMEOUT,
                maxBuffer: 1048576,
                encoding: "utf8",
            }, (error: any, stdout: string | Buffer, stderr: string | Buffer) => {
                if (error) {
                    const msg = error instanceof Error ? error.message : String(error);
                    terminalLog(`Shell snapshot creation failed: ${msg}`);
                    errorLog(Error(`Failed to create shell snapshot: ${msg}`));
                    resolve(undefined);
                } else {
                    if (fileExists(snapshotFilePath)) {
                        const fileSize = statSync(snapshotFilePath).size;
                        terminalLog(`Shell snapshot created successfully (${fileSize} bytes)`);

                        onCleanup(async () => {
                            try {
                                if (fileExists(snapshotFilePath)) {
                                    unlinkSync(snapshotFilePath);
                                    terminalLog(`Cleaned up session snapshot: ${snapshotFilePath}`);
                                }
                            } catch (cleanupError) {
                                terminalLog(`Error cleaning up session snapshot: ${cleanupError}`);
                            }
                        });
                        resolve(snapshotFilePath);
                    } else {
                        terminalLog(`Shell snapshot file not found after creation: ${snapshotFilePath}`);
                        resolve(undefined);
                    }
                }
            });
        } catch (unexpectedError) {
            terminalLog(`Unexpected error during snapshot creation: ${unexpectedError}`);
            errorLog(unexpectedError instanceof Error ? unexpectedError : Error(String(unexpectedError)));
            resolve(undefined);
        }
    });
}
