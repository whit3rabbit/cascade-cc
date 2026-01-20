import { sandboxService } from '../../services/sandbox/sandboxService.js';

export function getBashToolPrompt() {
    const isSandboxing = sandboxService.isEnabled();

    const sandboxDescription = isSandboxing ? `
    - Commands run in a sandbox by default with the following restrictions:
      - Filesystem: Restricted to project directory and specific allowed paths.
      - Network: Restricted to allowed hosts.
    - IMPORTANT: For temporary files, use \`/tmp/claude/\` as your temporary directory
      - The TMPDIR environment variable is automatically set to \`/tmp/claude\` when running in sandbox mode
      - Do NOT use \`/tmp\` directly - use \`/tmp/claude/\` or rely on TMPDIR instead
    ` : `
    - CRITICAL: All commands MUST run in sandbox mode - the \`dangerouslyDisableSandbox\` parameter is disabled by policy
    - If a command fails due to sandbox restrictions, work with the user to adjust sandbox settings instead`;

    return `Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.

IMPORTANT: This tool is for terminal operations like git, npm, docker, etc. DO NOT use it for file operations (reading, writing, editing, searching, finding files) - use the specialized tools for this instead.

Before executing the command, please follow these steps:

1. Directory Verification:
   - If the command will create new directories or files, first use \`ls\` to verify the parent directory exists and is the correct location
   - For example, before running "mkdir foo/bar", first use \`ls foo\` to check that "foo" exists and is the intended parent directory

2. Command Execution:
   - Always quote file paths that contain spaces with double quotes (e.g., cd "path with spaces/file.txt")
   - After ensuring proper quoting, execute the command.
   - Capture the output of the command.

Usage notes:
  - The command argument is required.
  - You can specify an optional timeout in milliseconds.
  - It is very helpful if you write a clear, concise description of what this command does in 5-10 words.
  - You can use the \`run_in_background\` parameter to run the command in the background.
  ${sandboxDescription}
  - Avoid using Bash with the \`find\`, \`grep\`, \`cat\`, \`head\`, \`tail\`, \`sed\`, \`awk\`, or \`echo\` commands, unless explicitly instructed.
  - Try to maintain your current working directory throughout the session by using absolute paths.
`;
}

export const BashToolPrompt = {
    description: getBashToolPrompt()
};
