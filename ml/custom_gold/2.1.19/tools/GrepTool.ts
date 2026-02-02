
import { executeBashCommand } from '../utils/shared/bashUtils.js';
import { createRequire } from 'module';

const require = createRequire(import.meta.url);
let rgPath = 'rg'; // Default to system rg

try {
    const vscodeRg = require('vscode-ripgrep');
    if (vscodeRg.rgPath) {
        rgPath = vscodeRg.rgPath;
    }
} catch (e) {
    // vscode-ripgrep not found, rely on system rg
}

export interface GrepInput {
    pattern: string;
    path?: string;
    glob?: string;
    output_mode?: "content" | "files_with_matches" | "count";
    "-B"?: number;
    "-A"?: number;
    "-C"?: number;
    "-n"?: boolean;
    "-i"?: boolean;
    type?: string;
    head_limit?: number;
    offset?: number;
    multiline?: boolean;
}

export const GrepTool = {
    name: "Grep",
    description: "Search for patterns in files using ripgrep (rg).",
    isConcurrencySafe: true,
    async call(input: GrepInput) {
        const { pattern, path = ".", glob, output_mode = "content", type, multiline, head_limit = 0, offset = 0 } = input;

        // Build rg command
        const args = [rgPath, '--color=never'];

        // Output mode
        if (output_mode === 'files_with_matches') {
            args.push('--files-with-matches');
        } else if (output_mode === 'count') {
            args.push('--count');
        } else {
            // content
            // -n is default for content, unless explicitly disabled?
            // Schema say "Show line numbers ... Defaults to true."
            if (input["-n"] !== false) {
                args.push('--line-number');
            }
            if (input["-B"]) args.push(`-B ${input["-B"]}`);
            if (input["-A"]) args.push(`-A ${input["-A"]}`);
            if (input["-C"]) args.push(`-C ${input["-C"]}`);
        }

        if (input["-i"]) {
            args.push('--ignore-case');
        } else {
            // Schema doesn't say smart-case default, but rg usually implies it? 
            // We'll stick to explicit flags from input.
        }

        if (multiline) args.push('--multiline');
        if (type) args.push(`--type ${type}`);
        if (glob) args.push(`--glob '${glob}'`);

        // Pattern - ensure safe quoting!
        // Using strict quoting for the pattern
        const safePattern = pattern.replace(/'/g, "'\\''");
        args.push(`'${safePattern}'`);

        // Path
        args.push(`"${path}"`);

        // Construct full pipeline for head/tail limits
        let command = args.join(' ');

        if (offset > 0) {
            command += ` | tail -n +${offset + 1}`; // tail is 1-indexed for +N
        }
        if (head_limit > 0) {
            command += ` | head -n ${head_limit}`;
        }

        // Execute
        try {
            // 60s timeout seems reasonable for grep
            const { exitCode, stdout, stderr } = await executeBashCommand(command, { timeout: 60000 });

            // rg returns 1 if no matches found, which isn't an "error" in tool sense
            if (exitCode === 1 && stdout === "") {
                return {
                    is_error: false,
                    content: "No matches found."
                };
            }

            if (exitCode !== 0 && exitCode !== 1) { // 1 is no match, >1 is error
                return {
                    is_error: true,
                    content: `Grep failed (exit code ${exitCode}):\n${stderr}`
                };
            }

            return {
                is_error: false,
                content: stdout || "No matches found."
            };

        } catch (error: any) {
            return {
                is_error: true,
                content: `Execution failed: ${error.message}`
            };
        }
    }
};
