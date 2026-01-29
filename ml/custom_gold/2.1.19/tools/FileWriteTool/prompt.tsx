// @ts-nocheck
/**
 * File: src/tools/FileWriteTool/prompt.js
 * Role: CLI_COMMAND
 */

import { existsSync, statSync, mkdirSync, realpathSync } from "node:fs";
import { execFile } from "node:child_process";
import { join as joinPath } from "node:path";
import * as os from "node:os";
import { fileExists, getHomeDir, getRelativeFilePath, readEnvVar } from "../../utils/fs/paths";
import { cleanupOnExit } from "../../utils/cleanup";
import { getPlatform, isWsl } from "../../utils/platform/detector";
import { resolvePathFromHome } from "../../utils/fs/paths";
import { getGitBashPath } from "../../utils/platform/shell";
import { stringWidth as calculateStringWidth } from "../../utils/text/ansi"; // Use ansi directly
import * as ansi from "../../utils/text/ansi";
import { terminalLog, errorLog, infoLog, m1 as getCwd, getAppEnv } from "../../utils/shared/runtime";
import { truncateString } from "../../utils/text/ansi"; // obtain from ansi
import { executeBashCommand, spawnBashCommand } from "../../utils/shared/bashUtils";
import { setBashHistory } from "../../hooks/useHistory";
import { signals } from "../../utils/shared/constants";

// Constants
const base64EncodedString = "\\";
const SHELL_SNAPSHOT_TIMEOUT = 10000; // 10 seconds

/**
 * Gets the rip-grep alias or function definition.
 * @returns {object} An object containing the type and snippet.
 */
function getRipgrepAliasOrFunctioinfoLog() {
  const ripGrepConfig = readRipGrepConfig();
  const rgPath = ripGrepConfig.rgPath;

  if (!rgPath) {
    return { type: "alias", snippet: "rg" };
  }

  const rgArgs = ripGrepConfig.rgArgs || [];
  const escapedArgs = rgArgs.map(arg => truncateString([arg])); // Assuming truncateString escapes the arguments
  const rgAlias = escapedArgs.join(" ");

  if (ripGrepConfig.argv0) {
    return {
      type: "function",
      snippet: `
function rg {
  if [[ -n $ZSH_VERSION ]]; then
    ARGV0=rg ${truncateString([rgPath])} "$@"
  elif [[ $BASHPID != $$ ]]; then
    exec -a rg ${truncateString([rgPath])} "$@"
  else
    (exec -a rg ${truncateString([rgPath])} "$@")
  fi
}
`.trim(),
    };
  }

  return {
    type: "alias",
    snippet: rgArgs.length > 0 ? `${truncateString([rgPath])} ${rgAlias}` : truncateString([rgPath]),
  };
}

/**
 * Determines the shell configuration file path based on the shell type.
 * @param {string} shellPath - The shell path (e.g., /bin/bash).
 * @returns {string} The path to the shell configuration file.
 */
function getShellConfigFilePath(shellPath) {
  const shellName = shellPath.includes("zsh") ? ".zshrc" : shellPath.includes("bash") ? ".bashrc" : ".profile";
  return joinPath(os.homedir(), shellName);
}

/**
 * Generates a shell configuration snapshot.  This extracts functions, shell options, and aliases and
 * writes them to a file. Used to recreate the shell environment as accurately as possible.
 * @param {string} shellConfigFile The path to the shell configuration file.
 * @returns {string} A string containing the shell configuration snapshot script.
 */
function generateShellConfigurationSnapshot(shellConfigFile) {
  const isZsh = shellConfigFile.endsWith(".zshrc");
  let snapshotScript = "";

  if (isZsh) {
    snapshotScript += `
      echo "# Functions" >> "$SNAPSHOT_FILE"

      # Force autoload all functions first
      typeset -f > /dev/null 2>&1

      # Now get user function names - filter system ones and write directly to file
      typeset +f | grep -vE '^(_|__)' | while read func; do
        typeset -f "$func" >> "$SNAPSHOT_FILE"
      done
    `;
  } else {
    snapshotScript += `
      echo "# Functions" >> "$SNAPSHOT_FILE"

      # Force autoload all functions first
      declare -f > /dev/null 2>&1

      # Now get user function names - filter system ones and give the rest to eval in b64 encoding
      declare -F | cut -d' ' -f3 | grep -vE '^(_|__)' | while read func; do
        # Encode the function to base64, preserving all special characters
        encoded_func=$(declare -f "$func" | base64 )
        # Write the function definition to the snapshot
        echo "eval ${base64EncodedString}${base64EncodedString}$(echo '$encoded_func' | base64 -d)${base64EncodedString}" > /dev/null 2>&1" >> "$SNAPSHOT_FILE"
      done
    `;
  }

  if (isZsh) {
    snapshotScript += `
      echo "# Shell Options" >> "$SNAPSHOT_FILE"
      setopt | sed 's/^/setopt /' | head -n 1000 >> "$SNAPSHOT_FILE"
    `;
  } else {
    snapshotScript += `
      echo "# Shell Options" >> "$SNAPSHOT_FILE"
      shopt -p | head -n 1000 >> "$SNAPSHOT_FILE"
      set -o | grep "on" | awk '{print "set -o " $1}' | head -n 1000 >> "$SNAPSHOT_FILE"
      echo "shopt -s expand_aliases" >> "$SNAPSHOT_FILE"
    `;
  }

  snapshotScript += `
      echo "# Aliases" >> "$SNAPSHOT_FILE"
      # Filter out winpty aliases on Windows to avoid "stdin is not a tty" errors
      # Git Bash automatically creates aliases like "alias node='winpty node.exe'" for
      # programs that need Win32 Console in mintty, but winpty fails when there's no TTY
      if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        alias | grep -v "='winpty " | sed 's/^alias //g' | sed 's/^/alias -- /' | head -n 1000 >> "$SNAPSHOT_FILE"
      else
        alias | sed 's/^alias //g' | sed 's/^/alias -- /' | head -n 1000 >> "$SNAPSHOT_FILE"
      fi
  `;

  return snapshotScript;
}

/**
 * Determines the cli path for the current process, used in mcp-cli alias detection.
 * @returns {object | null} An object containing the cliPath and arguments or null.
 */
function getCliPath() {
  if (!isWsl()) {
    return null;
  }
  try {
    let cliPath = getGitBashPath() ? process.execPath : process.argv[1];

    if (!cliPath) {
      return null;
    }
    try {
      cliPath = realpathSync(cliPath);
    } catch { }

    if (getPlatform() === "windows") {
      cliPath = resolvePathFromHome(cliPath);
    }

    return {
      cliPath: cliPath,
      args: ["--mcp-cli"],
    };
  } catch (error) {
    errorLog(error instanceof Error ? error : Error(String(error)));
    return null;
  }
}

/**
 * Constructs the shell script to set up environment variables and aliases.
 * @returns {Promise<string>} The shell script as a string.
 */
async function buildShellSetupScript() {
  let envPath = process.env.PATH;
  if (getPlatform() === "windows") {
    const result = await executeBashCommand("echo $PATH", {
      shell: true,
      reject: false,
    });
    if (result.exitCode === 0 && result.stdout) {
      envPath = result.stdout.trim();
    }
  }

  const ripgrep = getRipgrepAliasOrFunctioinfoLog();
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
    const escapedSnippet = ripgrep.snippet.replace(/'/g, "'\\''");
    script += `
      echo '  alias rg='"'${escapedSnippet}'" >> "$SNAPSHOT_FILE"
    `;
  }

  script += `
      echo "fi" >> "$SNAPSHOT_FILE"
  `;

  if (cliPath) {
    const escapedCliPath = truncateString([cliPath.cliPath]);
    const escapedArgs = cliPath.args.map(arg => truncateString([arg])).join(" ");
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
      echo "export PATH=${truncateString([envPath || ""])}" >> "$SNAPSHOT_FILE"
  `;

  return script;
}

/**
 * Builds the shell snapshot script.
 * @param {string} shellPath - The path to the shell executable.
 * @param {string} snapshotFilePath - The path to the snapshot file.
 * @param {boolean} shouldSourceConfig - Whether to source the shell config file.
 * @returns {Promise<string>} The full snapshot script.
 */
async function buildShellSnapshotScript(shellPath, snapshotFilePath, shouldSourceConfig) {
  const shellConfigFile = getShellConfigFilePath(shellPath);
  const isZsh = shellConfigFile.endsWith(".zshrc");
  const sourceCommand = shouldSourceConfig ? `source "${shellConfigFile}" < /dev/null` : "# No user config file to source";
  const shellSetupScript = await buildShellSetupScript();

  // Build the core commands
  const snapshotScript = `
    SNAPSHOT_FILE=${truncateString([snapshotFilePath])}
    ${sourceCommand}

    # First, create/clear the snapshot file
    echo "# Snapshot file" >| "$SNAPSHOT_FILE"

    # When this file is sourced, we first unalias to avoid conflicts with functions
    # This is necessary because aliases get "frozen" inside function definitions at definition time,
    # which can cause unexpected behavior when functions use commands that conflict with aliases
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

  return snapshotScript;
}

/**
 * Creates a shell snapshot, capturing shell configuration and environment for later use.
 * This function handles creating a temporary file, executing the necessary shell commands
 * to capture the environment, and then cleans up the temporary file.
 * @param {string} shellPath - The full path to the shell executable (e.g., /bin/bash).
 * @returns {Promise<string | undefined>} - The path to the snapshot file if successful, otherwise undefined.
 */
async function createShellSnapshot(shellPath) {
  const shellType = shellPath.includes("zsh") ? "zsh" : shellPath.includes("bash") ? "bash" : "sh";
  terminalLog(`Creating shell snapshot for ${shellType} (${shellPath})`);

  return new Promise(async (resolve) => {
    try {
      const shellConfigFile = getShellConfigFilePath(shellPath);
      terminalLog(`Looking for shell config file: ${shellConfigFile}`);

      const shouldSourceConfig = fileExists(shellConfigFile);

      if (!shouldSourceConfig) {
        terminalLog(`Shell config file not found: ${shellConfigFile}, creating snapshot with Claude Code defaults only`);
      }

      const timestamp = Date.now();
      const randomString = Math.random().toString(36).substring(2, 8);
      const snapshotsDir = getRelativeFilePath(getHomeDir(), "shell-snapshots");
      terminalLog(`Snapshots directory: ${snapshotsDir}`);
      const snapshotFilePath = getRelativeFilePath(snapshotsDir, `snapshot-${shellType}-${timestamp}-${randomString}.sh`);
      mkdirSync(snapshotsDir, { recursive: true });

      const snapshotScript = await buildShellSnapshotScript(shellPath, snapshotFilePath, shouldSourceConfig);
      terminalLog(`Creating snapshot at: ${snapshotFilePath}`);
      terminalLog(`Shell binary exists: ${fileExists(shellPath)}`);
      terminalLog(`Execution timeout: ${SHELL_SNAPSHOT_TIMEOUT}ms`);

      spawnBashCommand(shellPath, ["-c", "-l", snapshotScript], {
        env: {
          ...(process.env.CLAUDE_CODE_DONT_INHERIT_ENV ? {} : process.env),
          SHELL: shellPath,
          GIT_EDITOR: "true",
          CLAUDECODE: "1",
        },
        timeout: SHELL_SNAPSHOT_TIMEOUT,
        maxBuffer: 1048576,
        encoding: "utf8",
      }, async (error, stdout, stderr) => {
        if (error) {
          const errorMessage = error;
          terminalLog(`Shell snapshot creation failed: ${errorMessage.message}`);
          terminalLog("Error details:");
          terminalLog(`  - Error code: ${errorMessage?.code}`);
          terminalLog(`  - Error signal: ${errorMessage?.signal}`);
          terminalLog(`  - Error killed: ${errorMessage?.killed}`);
          terminalLog(`  - Shell path: ${shellPath}`);
          terminalLog(`  - Config file: ${shellConfigFile}`);
          terminalLog(`  - Config file exists: ${shouldSourceConfig}`);
          terminalLog(`  - Working directory: ${getCwd()}`);
          terminalLog(`  - Claude home: ${getHomeDir()}`);
          terminalLog(`Full snapshot script:
${snapshotScript}`);
          if (stdout) {
            terminalLog(`stdout output (${stdout.length} chars):
${stdout}`);
          } else {
            terminalLog("No stdout output captured");
          }
          if (stderr) {
            terminalLog(`stderr output (${stderr.length} chars): ${stderr}`);
          } else {
            terminalLog("No stderr output captured");
          }
          errorLog(Error(`Failed to create shell snapshot: ${errorMessage.message}`));
          const signalNumber = errorMessage?.signal ? signals[errorMessage.signal] : void 0;
          infoLog("tengu_shell_snapshot_failed", {
            stderr_length: stderr?.length || 0,
            has_error_code: !!errorMessage?.code,
            error_signal_number: signalNumber,
            error_killed: errorMessage?.killed,
          });
          resolve(undefined);
        } else {
          if (fileExists(snapshotFilePath)) {
            const fileSize = statSync(snapshotFilePath).size;
            terminalLog(`Shell snapshot created successfully (${fileSize} bytes)`);

            cleanupOnExit(async () => {
              try {
                if (fileExists(snapshotFilePath)) {
                  require("node:fs").unlinkSync(snapshotFilePath);
                  terminalLog(`Cleaned up session snapshot: ${snapshotFilePath}`);
                }
              } catch (cleanupError) {
                terminalLog(`Error cleaning up session snapshot: ${cleanupError}`);
              }
            });
            resolve(snapshotFilePath);
          } else {
            terminalLog(`Shell snapshot file not found after creation: ${snapshotFilePath}`);
            terminalLog(`Checking if parent directory still exists: ${snapshotsDir}`);
            const directoryExists = fileExists(snapshotsDir);
            terminalLog(`Parent directory exists: ${directoryExists}`);
            if (directoryExists) {
              try {
                const filesInDirectory = require("node:fs").readdirSync(snapshotsDir);
                terminalLog(`Directory contains ${filesInDirectory.length} files`);
              } catch (readDirError) {
                terminalLog(`Could not read directory contents: ${readDirError}`);
              }
            }
            infoLog("tengu_shell_unknown_error", {});
            resolve(undefined);
          }
        }
      });
    } catch (unexpectedError) {
      terminalLog(`Unexpected error during snapshot creation: ${unexpectedError}`);
      if (unexpectedError instanceof Error) {
        terminalLog(`Error stack trace: ${unexpectedError.stack}`);
      }
      errorLog(unexpectedError instanceof Error ? unexpectedError : Error(String(unexpectedError)));
      infoLog("tengu_shell_snapshot_error", {});
      resolve(undefined);
    }
  });
}

// Placeholder for missing functions - replace with actual implementations as needed.
function readRipGrepConfig() {
  return {
    rgPath: "/usr/bin/rg",
    rgArgs: [],
    argv0: false
  };
}

function fileExistsWrapper(Y) {
  // Assuming fileExistsWrapper checks for the possibility of sourcing the file
  return existsSync(Y);
}