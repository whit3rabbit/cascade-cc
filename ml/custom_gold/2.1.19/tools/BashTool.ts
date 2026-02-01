/**
 * File: src/tools/BashTool.ts
 * Role: Executes bash commands.
 */

import { executeBashCommand, cleanCwdResetWarning, cleanSandboxViolations } from '../utils/shared/bashUtils.js';
import { isSandboxEnabled, isUrlAllowed, areUnsandboxedCommandsAllowed } from '../services/sandbox/SandboxSettings.js';
import { EnvService } from '../services/config/EnvService.js';
import { getSessionEnvScript } from '../utils/shared/runtimeAndEnv.js';
import { parseCommandWithRedirections } from '../utils/shared/commandStringProcessing.js';
import { interceptCommand } from '../services/sandbox/SandboxInterceptor.js';

export interface BashToolInput {
    command: string;
    timeout?: number;
    delay?: number;
    dangerouslyDisableSandbox?: boolean;
}

export const BashTool = {
    name: "Bash",
    description: "Run scripts or commands on the local machine.",
    async call(input: BashToolInput, context: any) {
        const { command, timeout } = input;
        const cwd = context.cwd || process.cwd();

        // 1. Redirection Safety & URL validation
        const { commandWithoutRedirections, hasDangerousRedirection } = parseCommandWithRedirections(command);
        if (hasDangerousRedirection) {
            return {
                is_error: true,
                content: `<sandbox_violations>\nBlocked potentially dangerous redirection in command.\n</sandbox_violations>`
            };
        }

        if (isSandboxEnabled()) {
            // ... URL check logic would go here
            if (input.dangerouslyDisableSandbox && !areUnsandboxedCommandsAllowed()) {
                return {
                    is_error: true,
                    content: "Sandbox violation: dangerous command execution blocked by policy."
                };
            }
        }

        try {
            let finalCommand = interceptCommand(command);
            const shellSnapshotPath = context.options?.shellSnapshotPath;
            const shell = EnvService.get("SHELL");
            const shellPrefix = EnvService.get("CLAUDE_CODE_SHELL_PREFIX") || "";

            // Apply CLAUDE_ENV_FILE if present
            const sessionEnvScript = getSessionEnvScript();
            let envPrefix = "";
            if (sessionEnvScript) {
                // We can source it if it's a file, or eval it if it's content.
                // getSessionEnvScript returns content. We might need to write it to a temp file or use process.env directly.
                // But wait, chunks suggest sourcing a file.
                // Let's assume we just want to source the file path stored in env var for simplicity if getSessionEnvScript returned null,
                // but we implemented getSessionEnvScript to return content.
                // Re-reading chunk431: "Session environment loaded from CLAUDE_ENV_FILE...".
                // It seems simpler to just check the env var path here for sourcing in bash.
                const envFilePath = EnvService.get('CLAUDE_ENV_FILE');
                if (envFilePath) {
                    envPrefix = `source "${envFilePath}"; `;
                }

                if (EnvService.get('CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR') === 'true') {
                    // Placeholder for stateful CWD logic
                }
            }

            // Construct final command with optional prefix
            // If prefix is "log_wrapper.sh ", it becomes "log_wrapper.sh <shell> -c 'command'" ??
            // Or does it wrap the command string inside?
            // "Command prefix to wrap all bash commands (for example, for logging or auditing). Example: /path/to/logger.sh will execute /path/to/logger.sh <command>"
            // So it should be `shellPrefix + " " + command`?
            // Wait, command is executed via `shell -c`.
            // If shellPrefix is set, maybe we prepend it to the whole execution string?
            // "Example: /path/to/logger.sh will execute /path/to/logger.sh <command>"
            // This sounds like it replaces the direct execution.
            // But we are running `shell -c '...'`.
            // Let's assume it prepends inside the shell execution if it's a logger, or wraps the shell itself.
            // Let's interpret strictly: "/path/to/logger.sh <command>"
            // If we are doing `bash -c '...'`, then the "command" is the whole thing?
            // "CLAUDE_CODE_SHELL_PREFIX ... Example: /path/to/logger.sh"
            // If I have `ls -la`, maybe it runs `/path/to/logger.sh ls -la`?
            // But complex commands with redirects/pipes need shell.
            // So probably: `/path/to/logger.sh bash -c 'ls -la'` ?
            // Or `bash -c '/path/to/logger.sh ls -la'`?
            // Given the description "wrap all bash commands", let's assume it prefixes the raw command string before shell processing?
            // Actually, for safety and consistency with "Example: /path/to/logger.sh <command>", it likely means the outer executor.
            // But `executeBashCommand` usually spawns a shell.
            // If I look at `executeBashCommand` implementation (not visible here but standard usually), it usually does `spawn(shell, ['-c', command])`.
            // If we use a prefix, we might need to adjust what we pass to `executeBashCommand`.
            // Providing simple concatenation inside the shell string seems safest for now:
            // finalCommand = `${shellPrefix} ${command}`;
            // But wait, we construct `finalCommand` below.

            let commandToExecute = command;
            const maintainCwd = EnvService.get('CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR') === 'true';

            if (maintainCwd) {
                // Append a marker with the current CWD after execution
                commandToExecute = `${command}; printf "\\n<cwd>%s</cwd>\\n" "$PWD"`;
            }

            if (shellPrefix) {
                commandToExecute = `${shellPrefix} ${commandToExecute}`;
            }

            if (shellSnapshotPath) {
                // Wrap command to source the snapshot
                finalCommand = `${shell} -c '${envPrefix}source "${shellSnapshotPath}"; ${commandToExecute.replace(/'/g, "'\\''")}'`;
            } else if (envPrefix) {
                finalCommand = `${shell} -c '${envPrefix}${commandToExecute.replace(/'/g, "'\\''")}'`;
            } else {
                finalCommand = `${shell} -c '${commandToExecute.replace(/'/g, "'\\''")}'`;
            }

            const { exitCode, stdout, stderr } = await executeBashCommand(finalCommand, {
                cwd,
                timeout: timeout || Number(EnvService.get('BASH_DEFAULT_TIMEOUT_MS')) || 600000 // 10 min default
            });

            // Post-processing
            const { cleanedStderr, cwdResetWarning } = cleanCwdResetWarning(stderr);
            const { cleanedStderr: finalStderr } = cleanSandboxViolations(cleanedStderr);

            // Handle BASH_MAX_OUTPUT_LENGTH
            const maxOutputLength = Number(EnvService.get('BASH_MAX_OUTPUT_LENGTH')) || 600000; // 600k default from chunks

            let displayStdout = stdout;
            let newCwd = cwd;

            if (maintainCwd) {
                const cwdMatch = displayStdout.match(/<cwd>(.*)<\/cwd>/);
                if (cwdMatch) {
                    newCwd = cwdMatch[1].trim();
                    displayStdout = displayStdout.replace(/<cwd>.*<\/cwd>/, "").trim();
                }
            }

            if (displayStdout.length > maxOutputLength) {
                const half = Math.floor(maxOutputLength / 2);
                displayStdout = displayStdout.slice(0, half) + `\n... [Output truncated (${displayStdout.length - maxOutputLength} chars)] ...\n` + displayStdout.slice(-half);
            }

            let displayStderr = finalStderr;
            if (displayStderr.length > maxOutputLength) {
                const half = Math.floor(maxOutputLength / 2);
                displayStderr = displayStderr.slice(0, half) + `\n... [Stderr truncated (${displayStderr.length - maxOutputLength} chars)] ...\n` + displayStderr.slice(-half);
            }

            // Construct output
            if (exitCode !== 0) {
                return {
                    is_error: true,
                    content: [
                        displayStdout ? `Stdout:\n${displayStdout}` : "",
                        displayStderr ? `Stderr:\n${displayStderr}` : "",
                        `Exit code: ${exitCode}`
                    ].filter(Boolean).join("\n")
                };
            }

            return {
                is_error: false,
                content: [
                    cwdResetWarning ? `Warning: ${cwdResetWarning}` : "",
                    displayStdout,
                    displayStderr ? `Stderr:\n${displayStderr}` : ""
                ].filter(Boolean).join("\n"),
                metadata: {
                    cwd: newCwd
                }
            };

        } catch (error: any) {
            return {
                is_error: true,
                content: `Execution failed: ${error.message}`
            };
        }
    }
};
