/**
 * File: src/entrypoints/cli.tsx
 * Role: Main entry point for the Claude Code CLI. Handles argument parsing and session initialization.
 */

import { Command, Option } from "commander";
import chalk from "chalk";
import { ConversationService } from "../services/conversation/ConversationService.js";
import { initializeApp } from "../services/terminal/AppInitializer.js";
import { terminalLog } from "../utils/shared/runtime.js";
import { commandRegistry } from "../services/terminal/CommandRegistry.js";

async function main() {
    const program = new Command();

    program
        .name("claude")
        .description("Claude Code - starts an interactive session by default, use -p/--print for non-interactive output")
        .version("2.0.76", "-v, --version", "Output the version number")
        .argument("[prompt]", "Your prompt")
        .option("--add-dir <directories...>", "Additional directories to allow tool access to")
        .option("--agent <agent>", "Agent for the current session. Overrides the 'agent' setting.")
        .option("--agents <json>", "JSON object defining custom agents (e.g. '{\"reviewer\": {\"description\": \"Reviews code\", \"prompt\": \"You are a code reviewer\"}}')")
        .option("--allow-dangerously-skip-permissions", "Enable bypassing all permission checks as an option, without it being enabled by default. Recommended only for sandboxes with no internet access.")
        .option("--allowedTools, --allowed-tools <tools...>", "Comma or space-separated list of tool names to allow (e.g. \"Bash(git:*) Edit\")")
        .option("--append-system-prompt <prompt>", "Append a system prompt to the default system prompt")
        .option("--betas <betas...>", "Beta headers to include in API requests (API key users only)")
        .option("--chrome", "Enable Claude in Chrome integration")
        .option("-c, --continue", "Continue the most recent conversation in the current directory")
        .option("--dangerously-skip-permissions", "Bypass all permission checks. Recommended only for sandboxes with no internet access.")
        .option("-d, --debug [filter]", "Enable debug mode with optional category filtering (e.g., \"api,hooks\" or \"!statsig,!file\")")
        .option("--debug-file <path>", "Write debug logs to a specific file path (implicitly enables debug mode)")
        .option("--disable-slash-commands", "Disable all skills")
        .option("--disallowedTools, --disallowed-tools <tools...>", "Comma or space-separated list of tool names to deny (e.g. \"Bash(git:*) Edit\")")
        .option("--fallback-model <model>", "Enable automatic fallback to specified model when default model is overloaded (only works with --print)")
        .option("--file <specs...>", "File resources to download at startup. Format: file_id:relative_path (e.g., --file file_abc:doc.txt file_def:img.png)")
        .option("--fork-session", "When resuming, create a new session ID instead of reusing the original (use with --resume or --continue)")
        .option("--ide", "Automatically connect to IDE on startup if exactly one valid IDE is available")
        .option("--include-partial-messages", "Include partial message chunks as they arrive (only works with --print and --output-format=stream-json)")
        .addOption(new Option("--input-format <format>", 'Input format (only works with --print): "text" (default), or "stream-json" (realtime streaming input)').choices(["text", "stream-json"]))
        .option("--json-schema <schema>", "JSON Schema for structured output validation. Example: {\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"}},\"required\":[\"name\"]}")
        .option("--max-budget-usd <amount>", "Maximum dollar amount to spend on API calls (only works with --print)")
        .option("--mcp-config <configs...>", "Load MCP servers from JSON files or strings (space-separated)")
        .option("--mcp-debug", "[DEPRECATED. Use --debug instead] Enable MCP debug mode (shows MCP server errors)")
        .option("--model <model>", "Model for the current session. Provide an alias for the latest model (e.g. 'sonnet' or 'opus') or a model's full name (e.g. 'claude-sonnet-4-5-20250929').")
        .option("--no-chrome", "Disable Claude in Chrome integration")
        .option("--no-session-persistence", "Disable session persistence - sessions will not be saved to disk and cannot be resumed (only works with --print)")
        .addOption(new Option("--output-format <format>", 'Output format (only works with --print): "text" (default), "json" (single result), or "stream-json" (realtime streaming)').choices(["text", "json", "stream-json"]))
        .addOption(new Option("--permission-mode <mode>", "Permission mode to use for the session").choices(["acceptEdits", "bypassPermissions", "default", "delegate", "dontAsk", "plan"]))
        .option("--plugin-dir <paths...>", "Load plugins from directories for this session only (repeatable)")
        .option("-p, --print", "Print response and exit (useful for pipes). Note: The workspace trust dialog is skipped when Claude is run with the -p mode. Only use this flag in directories you trust.")
        .option("--replay-user-messages", "Re-emit user messages from stdin back on stdout for acknowledgment (only works with --input-format=stream-json and --output-format=stream-json)")
        .option("-r, --resume [value]", "Resume a conversation by session ID, or open interactive picker with optional search term")
        .option("--session-id <uuid>", "Use a specific session ID for the conversation (must be a valid UUID)")
        .option("--setting-sources <sources>", "Comma-separated list of setting sources to load (user, project, local).")
        .option("--settings <file-or-json>", "Path to a settings JSON file or a JSON string to load additional settings from")
        .option("--strict-mcp-config", "Only use MCP servers from --mcp-config, ignoring all other MCP configurations")
        .option("--system-prompt <prompt>", "System prompt to use for the session")
        .option("--tools <tools...>", "Specify the list of available tools from the built-in set. Use \"\" to disable all tools, \"default\" to use all tools, or specify tool names (e.g. \"Bash,Edit,Read\").")
        .option("--auto-join <requestId>", "Automatically send a join request to a team leader on startup")
        .option("--verbose", "Override verbose mode setting from config")
        .action(async (prompt, options) => {
            terminalLog("Claude Code starting...");

            // 1. Initialize services
            const { isFirstRun } = await initializeApp();

            // 2. Handle auto-join if requested
            if (options.autoJoin) {
                const { TeammateTool } = await import("../tools/TeammateTool.js");
                const match = options.autoJoin.match(/@([^@]+)$/);
                const teamName = match?.[1];
                if (teamName) {
                    terminalLog(`Auto-joining team "${teamName}"...`);
                    try {
                        await TeammateTool.call({
                            operation: "requestJoin",
                            team_name: teamName,
                            proposed_name: options.agent || "teammate",
                            request_id: options.autoJoin,
                            timeout_ms: 10000
                        });
                    } catch (err) {
                        terminalLog(`Auto-join failed: ${err instanceof Error ? err.message : String(err)}`, "error");
                    }
                }
            }

            // 3. Decide flow: Interactive or Print
            if (options.print || prompt) {
                await runPrintMode(prompt, options);
            } else {
                await runInteractiveMode(options, isFirstRun);
            }
        });

    program.command('doctor')
        .description('Check the health of your Claude Code auto-updater')
        .action(() => {
            console.log("Doctor command initialized in skeleton mode.");
        });

    program.command('install [target]')
        .description('Install Claude Code native build. Use [target] to specify version (stable, latest, or specific version)')
        .action((target, options) => {
            console.log(`Install command initialized with target: ${target}`);
        });

    program.command('mcp')
        .description('Configure and manage MCP servers')
        .action(() => {
            console.log("MCP command initialized.");
        });

    program.command('plugin')
        .description('Manage Claude Code plugins')
        .action(() => {
            console.log("Plugin command initialized.");
        });

    program.command('setup-token')
        .description('Set up a long-lived authentication token (requires Claude subscription)')
        .action(() => {
            console.log("Setup-token command initialized.");
        });

    program.command('update')
        .description('Check for updates and install if available')
        .action(async () => {
            const { UpdaterService } = await import("../services/updater/UpdaterService.js");
            console.log("Checking for updates...");
            const update = await UpdaterService.checkForUpdates();

            if (update && update.hasUpdate) {
                console.log(chalk.green(`Update available: ${update.currentVersion} -> ${update.latestVersion}`));
                console.log(`Run ${chalk.cyan('npm install -g @anthropic-ai/claude-code')} to update.`);
                // Optionally execute the install command if permissions allow
                // import { execSync } from 'child_process';
                // execSync('npm install -g @anthropic-ai/claude-code', { stdio: 'inherit' });
            } else {
                console.log(chalk.green("You are using the latest version."));
            }
        });

    await program.parseAsync(process.argv);
}

/**
 * Non-interactive "print" mode.
 * Corresponds to the bifurcated flow using $bK in deobfuscation.md.
 */
async function runPrintMode(prompt: string, options: any) {
    terminalLog(`Running in print mode with prompt: ${prompt || "standard input"}`);

    const generator = ConversationService.startConversation(prompt, {
        commands: commandRegistry.getAllCommands(),
        tools: [],
        mcpClients: [],
        cwd: process.cwd(),
        verbose: options.verbose,
        model: options.model
    });

    for await (const chunk of generator) {
        if (chunk.type === "assistant") {
            const content = chunk.message.content;
            if (Array.isArray(content)) {
                const text = content
                    .filter((b: any) => b.type === 'text')
                    .map((b: any) => b.text)
                    .join('');
                process.stdout.write(text);
            } else {
                process.stdout.write(content);
            }
        } else if (chunk.type === "result") {
            console.log(chalk.green(`\n\nSession completed in ${chunk.num_turns} turns.`));
        }
    }
}

import { render } from "ink";
import { REPL } from "../components/terminal/REPL.js";

// ...

/**
 * Interactive REPL mode.
 * Uses REPL.tsx components via Ink render.
 */
async function runInteractiveMode(options: any, isFirstRun: boolean = false) {
    terminalLog("Entering interactive mode...");

    const { unmount } = render(
        <REPL
            initialPrompt={options.prompt}
            verbose={options.verbose}
            model={options.model}
            agent={options.agent}
            isFirstRun={isFirstRun}
        />
    );

    // Keep process alive if needed, or handle exit via REPL
    await new Promise(() => { }); // Infinite wait for now, REPL handles exit
}

main().catch(err => {
    console.error(chalk.red("Fatal error:"), err);
    process.exit(1);
});
