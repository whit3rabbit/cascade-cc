
import { createServer, Socket } from "node:net";
import { platform } from "node:os";
import * as fs from "node:fs";
import { log } from "../services/logger/loggerService.js";

const logger = log("entrypoints");

/**
 * Handles communication with the Claude Chrome extension via Native Messaging.
 * Based on chunk_604.ts (wV9)
 */
export class ChromeNativeHost {
    private mcpClients = new Map<number, { socket: Socket, buffer: Buffer }>();
    private nextClientId = 1;
    private server: any = null;
    private running = false;
    private socketPath: string;

    constructor() {
        // This should probably come from a central path utility
        this.socketPath = platform() === "win32"
            ? "\\\\.\\pipe\\claude-code"
            : "/tmp/claude-code.sock";
    }

    async start() {
        if (this.running) return;

        logger("native-host").info(`Starting Native Host on ${this.socketPath}`);

        if (platform() !== "win32" && fs.existsSync(this.socketPath)) {
            try {
                if (fs.statSync(this.socketPath).isSocket()) {
                    fs.unlinkSync(this.socketPath);
                }
            } catch (err) {
                logger("native-host").error("Failed to cleanup socket", err);
            }
        }

        this.server = createServer((socket) => this.handleMcpClient(socket));

        return new Promise<void>((resolve, reject) => {
            this.server.listen(this.socketPath, () => {
                logger("native-host").info("Socket server listening");
                if (platform() !== "win32") {
                    try {
                        fs.chmodSync(this.socketPath, 0o600);
                    } catch (err) {
                        logger("native-host").error("Failed to set socket permissions", err);
                    }
                }
                this.running = true;
                resolve();
            });

            this.server.on("error", (err: any) => {
                logger("native-host").error("Socket server error", err);
                reject(err);
            });

            this.server.on("close", () => {
                if (platform() !== "win32" && fs.existsSync(this.socketPath)) {
                    try {
                        fs.unlinkSync(this.socketPath);
                    } catch (err) {
                        logger("native-host").error("Failed to cleanup socket on close", err);
                    }
                }
            });
        });
    }

    private handleMcpClient(socket: Socket) {
        const id = this.nextClientId++;
        const client = { id, socket, buffer: Buffer.alloc(0) };
        this.mcpClients.set(id, client);

        logger("native-host").info(`MCP client ${id} connected`);

        this.sendToBrowser({ type: "mcp_connected" });

        socket.on("data", (data) => {
            client.buffer = Buffer.concat([client.buffer, data as any]);
            this.processMcpBuffer(client);
        });

        socket.on("close", () => {
            logger("native-host").info(`MCP client ${id} disconnected`);
            this.mcpClients.delete(id);
            this.sendToBrowser({ type: "mcp_disconnected" });
        });

        socket.on("error", (err) => {
            logger("native-host").error(`MCP client ${id} error`, err);
        });
    }

    private processMcpBuffer(client: any) {
        while (client.buffer.length >= 4) {
            const len = client.buffer.readUInt32LE(0);
            if (len === 0 || len > 1024 * 1024) {
                client.socket.destroy();
                return;
            }
            if (client.buffer.length < 4 + len) break;

            const payload = client.buffer.subarray(4, 4 + len);
            client.buffer = client.buffer.subarray(4 + len);

            try {
                const msg = JSON.parse(payload.toString("utf-8"));
                this.sendToBrowser({
                    type: "tool_request",
                    method: msg.method,
                    params: msg.params
                });
            } catch (err) {
                logger("native-host").error("Failed to parse MCP message", err);
            }
        }
    }

    private sendToBrowser(msg: any) {
        const payload = Buffer.from(JSON.stringify(msg));
        const len = Buffer.alloc(4);
        len.writeUInt32LE(payload.length, 0);
        process.stdout.write(len as any);
        process.stdout.write(payload as any);
    }

    async handleBrowserMessage(data: string) {
        try {
            const msg = JSON.parse(data);
            switch (msg.type) {
                case "ping":
                    this.sendToBrowser({ type: "pong", timestamp: Date.now() });
                    break;
                case "tool_response":
                case "notification":
                    if (this.mcpClients.size > 0) {
                        const { type, ...body } = msg;
                        const payload = Buffer.from(JSON.stringify(body));
                        const len = Buffer.alloc(4);
                        len.writeUInt32LE(payload.length, 0);
                        const frame = Buffer.concat([len, payload]);
                        for (const client of Array.from(this.mcpClients.values())) {
                            client.socket.write(frame);
                        }
                    }
                    break;
            }
        } catch (err) {
            logger("native-host").error("Failed to handle browser message", err);
        }
    }
}

/**
 * Main entrance for --chrome-native-host
 */
export async function startChromeNativeHost() {
    const host = new ChromeNativeHost();
    const reader = new ChromeNativeMessagingReader();
    await host.start();

    while (true) {
        const msg = await reader.read();
        if (msg === null) break;
        await host.handleBrowserMessage(msg);
    }
}

class ChromeNativeMessagingReader {
    private buffer = Buffer.alloc(0);
    private pendingResolve: ((val: string | null) => void) | null = null;
    private closed = false;

    constructor() {
        process.stdin.on("data", (chunk) => {
            this.buffer = Buffer.concat([this.buffer, chunk as any]);
            this.tryResolve();
        });
        process.stdin.on("end", () => {
            this.closed = true;
            if (this.pendingResolve) this.pendingResolve(null);
        });
    }

    private tryResolve() {
        if (!this.pendingResolve || this.buffer.length < 4) return;
        const len = this.buffer.readUInt32LE(0);
        if (len === 0 || len > 1024 * 1024) {
            this.pendingResolve(null);
            return;
        }
        if (this.buffer.length < 4 + len) return;

        const payload = this.buffer.subarray(4, 4 + len).toString("utf-8");
        this.buffer = this.buffer.subarray(4 + len);
        const resolve = this.pendingResolve;
        this.pendingResolve = null;
        resolve(payload);
    }

    async read(): Promise<string | null> {
        if (this.closed) return null;
        return new Promise((resolve) => {
            this.pendingResolve = resolve;
            this.tryResolve();
        });
    }
}

/**
 * Main application entry point.
 * Based on chunk_604.ts (ZV7)
 */
import { Command, Option } from "commander";

/**
 * Main application entry point.
 * Based on chunk_603.ts (ZV7/QV7)
 */
export async function cliMain() {
    const program = new Command();

    program
        .name("claude")
        .description("Claude Code - starts an interactive session by default, use -p/--print for non-interactive output")
        .version("2.0.76 (Claude Code)", "-v, --version", "Output the version number")
        .helpOption("-h, --help", "Display help for command")
        .argument("[prompt]", "Your prompt")
        .option("--add-dir <directories...>", "Additional directories to allow tool access to")
        .option("--agent <agent>", "Agent for the current session. Overrides the 'agent' setting.")
        .option("--agents <json>", "JSON object defining custom agents")
        .option("--allow-dangerously-skip-permissions", "Enable bypassing all permission checks as an option")
        .option("--allowedTools, --allowed-tools <tools...>", "Comma or space-separated list of tool names to allow")
        .option("--append-system-prompt <prompt>", "Append a system prompt to the default system prompt")
        .option("--betas <betas...>", "Beta headers to include in API requests")
        .option("--chrome", "Enable Claude in Chrome integration")
        .option("-c, --continue", "Continue the most recent conversation in the current directory")
        .option("--dangerously-skip-permissions", "Bypass all permission checks")
        .option("-d, --debug [filter]", "Enable debug mode with optional category filtering")
        .addOption(new Option("--debug-to-stderr", "Enable debug mode (to stderr)").hideHelp())
        .option("--disable-slash-commands", "Disable all slash commands")
        .option("--disallowedTools, --disallowed-tools <tools...>", "Comma or space-separated list of tool names to deny")
        .addOption(new Option("--enable-auth-status", "Enable auth status messages in SDK mode").default(false).hideHelp())
        .option("--fallback-model <model>", "Enable automatic fallback to specified model (only works with --print)")
        .option("--fork-session", "When resuming, create a new session ID instead of reusing the original")
        .option("--ide", "Automatically connect to IDE on startup")
        .option("--include-partial-messages", "Include partial message chunks as they arrive")
        .option("--input-format <format>", 'Input format (only works with --print): "text" (default), or "stream-json"', "text")
        .option("--json-schema <schema>", "JSON Schema for structured output validation")
        .addOption(new Option("--max-thinking-tokens <tokens>", "Maximum number of thinking tokens (only works with --print)").hideHelp())
        .addOption(new Option("--max-turns <turns>", "Maximum number of agentic turns in non-interactive mode (only works with --print)").hideHelp())
        .option("--max-budget-usd <amount>", "Maximum dollar amount to spend on API calls")
        .option("--mcp-config <config>", "Load MCP servers from JSON file or string (can be specified multiple times)")
        .option("--mcp-debug", "[DEPRECATED. Use --debug instead] Enable MCP debug mode")
        .option("--model <model>", "Model for the current session")
        .option("--no-chrome", "Disable Claude in Chrome integration")
        .option("--no-session-persistence", "Disable session persistence (only works with --print)")
        .option("--output-format <format>", 'Output format (only works with --print): "text" (default), "json", or "stream-json"', "text")
        .option("--permission-mode <mode>", "Permission mode to use (acceptEdits, bypassPermissions, default, delegate, dontAsk, plan)")
        .addOption(new Option("--permission-prompt-tool <tool>", "MCP tool to use for permission prompts (only works with --print)").hideHelp())
        .option("--plugin-dir <paths...>", "Load plugins from directories for this session only")
        .option("-p, --print", "Print response and exit")
        .option("--remote <description>", "Create a remote session with the given description")
        .option("--replay-user-messages", "Re-emit user messages from stdin back on stdout")
        .option("-r, --resume [value]", "Resume a conversation by session ID, or open interactive picker")
        .addOption(new Option("--resume-session-at <message id>", "When resuming, only messages up to and including the assistant message with <message.id>").hideHelp())
        .addOption(new Option("--rewind-files <user-message-id>", "Restore files to state at the specified user message and exit").hideHelp())
        .option("--session-id <uuid>", "Use a specific session ID for the conversation")
        .option("--setting-sources <sources>", "Comma-separated list of setting sources to load")
        .option("--settings <file-or-json>", "Path to a settings JSON file or a JSON string")
        .option("--strict-mcp-config", "Only use MCP servers from --mcp-config")
        .option("--system-prompt <prompt>", "System prompt to use for the session")
        .option("--system-prompt-file <file>", "Read system prompt from a file")
        .option("--append-system-prompt-file <file>", "Read system prompt from a file and append")
        .addOption(new Option("--teleport [session]", "Resume a teleport session, optionally specify session ID").hideHelp())
        .option("--tools <tools...>", "Specify the list of available tools from the built-in set")
        .option("--verbose", "Override verbose mode setting from config")
        .option("--chrome-native-host", "Internal: Start the Chrome Native Messaging host")
        .option("--mcp-cli", "Internal: Run MCP CLI commands")
        .option("--ripgrep", "Internal: Run ripgrep directly");

    // Subcommands
    program.command("doctor")
        .description("Check the health of your Claude Code auto-updater")
        .action(async () => {
            const React = await import("react");
            const { render } = await import("ink");
            const { DiagnosticsView } = await import("../components/terminal/DiagnosticsView.js");
            const { AppStateProvider } = await import("../contexts/AppStateContext.js");

            await new Promise<void>((resolve) => {
                const { unmount } = render(
                    React.default.createElement(AppStateProvider, null,
                        React.default.createElement(DiagnosticsView, {
                            onDone: () => { unmount(); resolve(); }
                        })
                    )
                );
                // In a real TUI, we might need more complex key handling but for 'doctor' this is a start
                process.stdin.once("data", () => {
                    unmount();
                    resolve();
                });
            });
            process.exit(0);
        });

    program.command("install [options] [target]")
        .description("Install Claude Code native build. Use [target] to specify version (stable, latest, or specific version)")
        .option("-f, --force", "Force installation even if already installed")
        .action(async (target, cmdOptions) => {
            const React = await import("react");
            const { render } = await import("ink");
            const { InstallationView } = await import("../components/terminal/InstallationView.js");
            const { AppStateProvider } = await import("../contexts/AppStateContext.js");

            await new Promise<void>((resolve) => {
                const { unmount } = render(
                    React.default.createElement(AppStateProvider, null,
                        React.default.createElement(InstallationView, {
                            force: cmdOptions.force,
                            target: target,
                            onDone: (msg) => {
                                console.log(msg);
                                unmount();
                                resolve();
                            }
                        })
                    )
                );
            });
            process.exit(0);
        });

    // MCP Subcommands
    const mcp = program.command("mcp").description("Configure and manage MCP servers");
    mcp.command("list")
        .description("List configured MCP servers")
        .option("--json", "Output in JSON format")
        .action(async (cmdOptions) => {
            const { McpCliAdmin } = await import("../services/mcp/McpCliAdmin.js");
            const result = await McpCliAdmin.listServers();
            if (cmdOptions.json) {
                console.log(JSON.stringify(result, null, 2));
            } else {
                if (result.errors.length > 0) {
                    console.error("Errors found:", result.errors);
                }
                if (result.servers.length === 0) {
                    console.log("No MCP servers configured.");
                } else {
                    result.servers.forEach(s => {
                        console.log(`${s.name.padEnd(20)} (${s.scope.padEnd(10)}) ${s.status}`);
                    });
                }
            }
            process.exit(0);
        });

    mcp.command("add <name> [commandOrUrl] [args...]")
        .description("Add an MCP server")
        .option("-t, --transport <transport>", "Transport type (stdio, sse, http)", "stdio")
        .option("-s, --scope <scope>", "Scope (user/project/local)", "user")
        .option("-e, --env <env...>", "Environment variables (KEY=VALUE)")
        .option("-H, --header <header...>", "HTTP headers (Name: Value)")
        .action(async (name, commandOrUrl, args, cmdOptions) => {
            const { McpCliAdmin } = await import("../services/mcp/McpCliAdmin.js");
            const env: Record<string, string> = {};
            if (cmdOptions.env) {
                cmdOptions.env.forEach((e: string) => {
                    const [k, v] = e.split("=");
                    if (k && v) env[k] = v;
                });
            }
            const headers: Record<string, string> = {};
            if (cmdOptions.header) {
                cmdOptions.header.forEach((h: string) => {
                    const [k, v] = h.split(":");
                    if (k && v) headers[k.trim()] = v.trim();
                });
            }

            let config: any;
            if (cmdOptions.transport === "sse" || cmdOptions.transport === "http") {
                config = { type: cmdOptions.transport, url: commandOrUrl, headers };
            } else {
                config = { type: "stdio", command: commandOrUrl, args: args || [], env };
            }

            await McpCliAdmin.addServer(name, config, cmdOptions.scope);
            console.log(`Added MCP server: ${name}`);
            process.exit(0);
        });

    mcp.command("add-json <name> <json>")
        .description("Add an MCP server via JSON string")
        .option("-s, --scope <scope>", "Scope (user/project/local)", "user")
        .action(async (name, json, cmdOptions) => {
            const { McpCliAdmin } = await import("../services/mcp/McpCliAdmin.js");
            const config = JSON.parse(json);
            await McpCliAdmin.addServer(name, config, cmdOptions.scope);
            console.log(`Added MCP server: ${name}`);
            process.exit(0);
        });

    mcp.command("get <name>")
        .description("Get details about an MCP server")
        .action(async (name) => {
            const { McpServerManager } = await import("../services/mcp/McpServerManager.js");
            const { servers } = await McpServerManager.getAllMcpServers();
            const server = servers[name];
            if (!server) {
                console.error(`No MCP server found with name: ${name}`);
                process.exit(1);
            }
            console.log(JSON.stringify(server, null, 2));
            process.exit(0);
        });

    mcp.command("serve")
        .description("Start the Claude Code MCP server")
        .action(async () => {
            // Logic from gH9 in chunk_603
            console.log("Claude Code MCP server starting...");
            // This would involve starting a server that exposes handles to the current session
            process.exit(0);
        });

    mcp.command("remove <name>")
        .description("Remove an MCP server")
        .option("-s, --scope <scope>", "Scope (user/project/local)", "user")
        .action(async (name, cmdOptions) => {
            const { McpCliAdmin } = await import("../services/mcp/McpCliAdmin.js");
            await McpCliAdmin.removeServer(name, cmdOptions.scope);
            console.log(`Removed MCP server: ${name}`);
            process.exit(0);
        });

    const plugin = program.command("plugin").description("Manage Claude Code plugins");

    const marketplace = plugin.command("marketplace").description("Manage Claude Code marketplaces");
    marketplace.command("add <source>")
        .description("Add a marketplace from a URL, path, or GitHub repo")
        .action(async (source) => {
            const { MarketplaceService } = await import("../services/marketplace/MarketplaceService.js");
            try {
                const result = await MarketplaceService.addMarketplace(source, (msg) => console.log(msg));
                console.log(`Successfully added marketplace: ${result.name}`);
            } catch (err: any) {
                console.error(`Failed to add marketplace: ${err.message}`);
                process.exit(1);
            }
            process.exit(0);
        });

    marketplace.command("list")
        .description("List all configured marketplaces")
        .action(async () => {
            const { readMarketplaceConfigFile } = await import("../services/marketplace/MarketplaceConfig.js");
            const config = await readMarketplaceConfigFile();
            if (Object.keys(config).length === 0) {
                console.log("No marketplaces configured.");
            } else {
                console.log("Configured Marketplaces:");
                for (const [name, entry] of Object.entries(config)) {
                    console.log(`- ${name.padEnd(20)} [${entry.source.source}] ${entry.lastUpdated || ''}`);
                }
            }
            process.exit(0);
        });

    marketplace.command("remove <name>")
        .alias("rm")
        .description("Remove a configured marketplace")
        .action(async (name) => {
            const { MarketplaceService } = await import("../services/marketplace/MarketplaceService.js");
            try {
                await MarketplaceService.removeMarketplace(name);
                console.log(`Successfully removed marketplace: ${name}`);
            } catch (err: any) {
                console.error(`Failed to remove marketplace: ${err.message}`);
                process.exit(1);
            }
            process.exit(0);
        });

    marketplace.command("update [name]")
        .description("Update marketplace(s) from their source - updates all if no name specified")
        .action(async (name) => {
            const { MarketplaceService } = await import("../services/marketplace/MarketplaceService.js");
            try {
                if (name) {
                    await MarketplaceService.refreshMarketplace(name, (msg) => console.log(msg));
                } else {
                    await MarketplaceService.refreshAllMarketplaces((msg) => console.log(msg));
                }
            } catch (err: any) {
                console.error(`Update failed: ${err.message}`);
                process.exit(1);
            }
            process.exit(0);
        });

    plugin.command("list")
        .description("List installed plugins")
        .action(async () => {
            const { getInstalledPlugins } = await import("../services/mcp/McpClientManager.js");
            const { enabled, disabled } = await getInstalledPlugins();

            console.log("\nEnabled Plugins:");
            if (enabled.length === 0) console.log("  (none)");
            else enabled.forEach(p => console.log(`  - ${p.id} (${p.version})`));

            console.log("\nDisabled Plugins:");
            if (disabled.length === 0) console.log("  (none)");
            else disabled.forEach(p => console.log(`  - ${p.id} (${p.version})`));

            process.exit(0);
        });

    plugin.command("install <id>")
        .description("Install a plugin from a marketplace")
        .option("-s, --scope <scope>", "Scope (user/project/local)", "user")
        .action(async (id, cmdOptions) => {
            const { installPlugin } = await import("../services/mcp/PluginManager.js");
            console.log(`Installing plugin: ${id}...`);
            const result = await installPlugin(id, cmdOptions.scope);
            if (result.success) {
                console.log(`✅ ${result.message}`);
            } else {
                console.error(`❌ ${result.message}`);
                process.exit(1);
            }
            process.exit(0);
        });

    plugin.command("uninstall <id>")
        .description("Uninstall a plugin")
        .option("-s, --scope <scope>", "Scope (user/project/local)", "user")
        .action(async (id, cmdOptions) => {
            const { uninstallPlugin } = await import("../services/mcp/PluginManager.js");
            const result = await uninstallPlugin(id, cmdOptions.scope);
            if (result.success) {
                console.log(`✅ ${result.message}`);
            } else {
                console.error(`❌ ${result.message}`);
                process.exit(1);
            }
            process.exit(0);
        });

    plugin.command("enable <id>")
        .description("Enable a plugin")
        .option("-s, --scope <scope>", "Scope (user/project/local)", "user")
        .action(async (id, cmdOptions) => {
            const { enablePlugin } = await import("../services/mcp/PluginManager.js");
            const result = await enablePlugin(id, cmdOptions.scope);
            if (result.success) {
                console.log(`✅ ${result.message}`);
            } else {
                console.error(`❌ ${result.message}`);
                process.exit(1);
            }
            process.exit(0);
        });

    plugin.command("disable <id>")
        .description("Disable a plugin")
        .option("-s, --scope <scope>", "Scope (user/project/local)", "user")
        .action(async (id, cmdOptions) => {
            const { disablePlugin } = await import("../services/mcp/PluginManager.js");
            const result = await disablePlugin(id, cmdOptions.scope);
            if (result.success) {
                console.log(`✅ ${result.message}`);
            } else {
                console.error(`❌ ${result.message}`);
                process.exit(1);
            }
            process.exit(0);
        });

    plugin.command("validate <path>")
        .description("Validate a plugin or marketplace manifest")
        .action(async (manifestPath) => {
            const { validateManifest } = await import("../utils/validation/PluginValidator.js");
            const result = validateManifest(manifestPath);
            if (result.success) {
                console.log("✅ Manifest is valid.");
                if (result.warnings && result.warnings.length > 0) {
                    console.log("\nWarnings:");
                    result.warnings.forEach(w => console.warn(`- ${w}`));
                }
            } else {
                console.error("❌ Manifest validation failed:");
                result.errors.forEach(e => console.error(`- ${e}`));
                process.exit(1);
            }
            process.exit(0);
        });

    program.command("setup-token <token>")
        .description("Set up a long-lived authentication token (requires Claude subscription)")
        .action(async (token) => {
            const { AuthService } = await import("../services/auth/AuthService.js");
            await AuthService.setupToken(token);
            console.log("Successfully set up authentication token.");
            process.exit(0);
        });

    program.command("update")
        .description("Check for updates and install if available")
        .option("-f, --force", "Force update even if already up to date")
        .action(async (cmdOptions) => {
            const React = await import("react");
            const { render } = await import("ink");
            const { InstallationView } = await import("../components/terminal/InstallationView.js");
            const { AppStateProvider } = await import("../contexts/AppStateContext.js");

            await new Promise<void>((resolve) => {
                const { unmount } = render(
                    React.default.createElement(AppStateProvider, null,
                        React.default.createElement(InstallationView, {
                            force: cmdOptions.force,
                            target: "latest",
                            onDone: (msg) => {
                                console.log(msg);
                                unmount();
                                resolve();
                            }
                        })
                    )
                );
            });
            process.exit(0);
        });

    const config = program.command("config").description("Manage Claude Code configuration");

    config.command("set <key> <value>")
        .description("Set a configuration value")
        .option("-s, --scope <scope>", "Scope (user/project/local)", "user")
        .action(async (key, value, cmdOptions) => {
            const { updateSettings } = await import("../services/terminal/settings.js");
            let parsedValue: any = value;
            if (value === "true") parsedValue = true;
            else if (value === "false") parsedValue = false;
            else if (!isNaN(Number(value))) parsedValue = Number(value);

            const scope = (cmdOptions.scope + "Settings") as any;
            const { error } = updateSettings(scope, { [key]: parsedValue });
            if (error) {
                console.error(`❌ Failed to set config: ${error.message}`);
                process.exit(1);
            }
            console.log(`✅ Set ${key} to ${value} in ${cmdOptions.scope} scope.`);
            process.exit(0);
        });

    config.command("get <key>")
        .description("Get a configuration value")
        .action(async (key) => {
            const { mergeSettings } = await import("../services/terminal/settings.js");
            const settings = mergeSettings();
            if (settings[key] !== undefined) {
                console.log(settings[key]);
            } else {
                console.error(`❌ Configuration key "${key}" not found.`);
                process.exit(1);
            }
            process.exit(0);
        });

    config.command("remove <key>")
        .alias("rm")
        .description("Remove a configuration value")
        .option("-s, --scope <scope>", "Scope (user/project/local)", "user")
        .action(async (key, cmdOptions) => {
            const { updateSettings } = await import("../services/terminal/settings.js");
            const scope = (cmdOptions.scope + "Settings") as any;
            updateSettings(scope, { [key]: undefined });
            console.log(`✅ Removed ${key} from ${cmdOptions.scope} scope.`);
            process.exit(0);
        });

    config.command("list")
        .description("List all configuration values")
        .action(async () => {
            const { mergeSettings } = await import("../services/terminal/settings.js");
            const settings = mergeSettings();
            console.log("\nCurrent Configuration:");
            for (const [key, value] of Object.entries(settings)) {
                if (typeof value === "object" && value !== null) continue; // Skip complex objects like mcpServers
                console.log(`  ${key.padEnd(25)}: ${value}`);
            }
            process.exit(0);
        });

    program.command("reset-project-choices")
        .description("Reset project-specific choices (e.g. allowed/denied tools and directories)")
        .action(async () => {
            const { updateSettings } = await import("../services/terminal/settings.js");
            updateSettings("projectSettings", {
                permissions: {
                    allow: [],
                    deny: [],
                    ask: []
                },
                enabledPlugins: {}
            });
            console.log("✅ Project choices have been reset.");
            process.exit(0);
        });

    // Default action for the main command (starts chat or runs prompt)
    program.action(() => {
        // No-op here, the logic continues after parseAsync
    });

    // Suppress default help command
    program.helpCommand(false);

    // Process global options manually first for early flags or let commander handle it without exiting
    // We use parseOptions to get options without triggering actions
    const { operands, unknown } = program.parseOptions(process.argv.slice(2));
    const options = program.opts();

    // Security check for dangerouslySkipPermissions
    if (options.dangerouslySkipPermissions) {
        if (process.platform !== "win32" && typeof process.getuid === "function" && process.getuid() === 0 && !process.env.IS_SANDBOX) {
            console.error("--dangerously-skip-permissions cannot be used with root/sudo privileges for security reasons");
            process.exit(1);
        }
    }

    if (options.teleport) {
        // Implement minimal teleport logic: just set session ID if provided
        if (typeof options.teleport === "string") {
            const { setSessionId } = await import("../services/session/sessionStore.js");
            setSessionId(options.teleport);
            console.log(`Resuming teleport session: ${options.teleport}`);
        } else {
            console.log("Teleport interactive picker not yet implemented.");
        }
    }

    if (options.remote) {
        console.log(`Connecting to remote session: "${options.remote}"...`);
        // In real app, this would start the remote client
        process.exit(0);
    }

    // Process early flags
    if (options.settings) {
        const { updateSettings } = await import("../services/terminal/settings.js");
        try {
            const content = options.settings.startsWith("{")
                ? JSON.parse(options.settings)
                : JSON.parse(fs.readFileSync(options.settings, "utf-8"));
            updateSettings("flagSettings", content);
        } catch (err) {
            console.error(`Failed to load settings: ${err}`);
            process.exit(1);
        }
    }

    if (options.systemPromptFile) {
        try {
            options.systemPrompt = fs.readFileSync(options.systemPromptFile, "utf-8");
        } catch (err) {
            console.error(`Failed to read system prompt file: ${err}`);
            process.exit(1);
        }
    }

    if (options.appendSystemPromptFile) {
        try {
            options.appendSystemPrompt = fs.readFileSync(options.appendSystemPromptFile, "utf-8");
        } catch (err) {
            console.error(`Failed to read append system prompt file: ${err}`);
            process.exit(1);
        }
    }

    if (options.ripgrep) {
        const { execa } = await import("execa");
        const rgPath = "rg";
        try {
            await execa(rgPath, program.args, { stdio: "inherit" });
            process.exit(0);
        } catch (err: any) {
            process.exit(err.exitCode || 1);
        }
    }

    if (options.mcpConfig) {
        const { McpServerManager } = await import("../services/mcp/McpServerManager.js");
        const configs = Array.isArray(options.mcpConfig) ? options.mcpConfig : [options.mcpConfig];
        for (const configStr of configs) {
            try {
                let config;
                if (configStr.startsWith("{")) {
                    config = JSON.parse(configStr);
                } else if (fs.existsSync(configStr)) {
                    config = JSON.parse(fs.readFileSync(configStr, "utf-8"));
                } else {
                    console.error(`Invalid MCP config: ${configStr}`);
                    continue;
                }

                if (config.mcpServers) {
                    for (const [name, serverConfig] of Object.entries(config.mcpServers)) {
                        await McpServerManager.addMcpServer(name, serverConfig, "dynamic");
                    }
                }
            } catch (err) {
                console.error(`Failed to load MCP config "${configStr}": ${err}`);
            }
        }
    }

    // Now parse fully to handle subcommands
    await program.parseAsync(process.argv);
    const prompt = program.args[0];

    // Handle early exit flags (internal)
    if (options.chromeNativeHost) {
        await startChromeNativeHost();
        process.exit(0);
    }

    if (options.mcpCli) {
        const mcpCliProgram = new Command();
        mcpCliProgram.name("claude --mcp-cli");

        mcpCliProgram.command("list")
            .description("List MCP servers from current session")
            .action(async () => {
                const { loadMcpState } = await import("../services/mcp/McpClientManager.js");
                const { getSessionId } = await import("../services/session/sessionStore.js");
                const state = loadMcpState(getSessionId());
                console.log(JSON.stringify(state.clients, null, 2));
                process.exit(0);
            });

        mcpCliProgram.command("get <name>")
            .description("Get details about an MCP server from current session")
            .action(async (name) => {
                const { loadMcpState } = await import("../services/mcp/McpClientManager.js");
                const { getSessionId } = await import("../services/session/sessionStore.js");
                const state = loadMcpState(getSessionId());
                const server = state.configs[name];
                if (!server) {
                    console.error(`Server "${name}" not found in current session.`);
                    process.exit(1);
                }
                console.log(JSON.stringify(server, null, 2));
                process.exit(0);
            });

        mcpCliProgram.command("read <resource>")
            .description("Read an MCP resource")
            .option("--json", "Output in JSON format")
            .action(async (resource) => {
                const { loadMcpState } = await import("../services/mcp/McpClientManager.js");
                const { getSessionId } = await import("../services/session/sessionStore.js");
                const state = loadMcpState(getSessionId());

                // Find resource in state
                const found = state.resources && Object.values(state.resources).flat().find((r: any) => r.uri === resource);
                if (found) {
                    console.log(`Resource metadata found for "${resource}". Live reading requires an active session.`);
                    console.log(JSON.stringify(found, null, 2));
                } else {
                    console.error(`Resource "${resource}" not found in current session.`);
                }
                process.exit(0);
            });

        await mcpCliProgram.parseAsync(process.argv.slice(process.argv.indexOf("--mcp-cli") + 1));
        process.exit(0);
    }

    if (options.settingSources) {
        // This would restrict which sources are loaded in settings.ts
    }

    // Determine if we should continue to TUI or run in print mode
    if (options.print) {
        // Full non-interactive mode requires wiring up the StreamingHandler and MainLoop
        // with a ConsoleStream to output results without the Ink TUI.
        logger("cli").info("Print mode requested. Non-interactive execution is currently being finalized.");
    }

    // Store parsed options in global state for services to use
    const { updateAppState } = await import("../contexts/AppStateContext.js");
    updateAppState(s => {
        const next = {
            ...s,
            verbose: options.verbose || s.verbose,
            mainLoopModel: options.model || s.mainLoopModel,
            thinkingEnabled: options.thinking !== false,
            // Map permission related options
            toolPermissionContext: {
                ...s.toolPermissionContext,
                mode: options.dangerouslySkipPermissions ? "allow" : (options.permissionMode || s.toolPermissionContext?.mode || "ask"),
            },
            maxThinkingTokens: options.maxThinkingTokens || s.maxThinkingTokens,
            maxTurns: options.maxTurns || s.maxTurns,
            maxBudgetUsd: options.maxBudgetUsd || s.maxBudgetUsd,
            outputFormat: options.outputFormat || s.outputFormat || "text",
            inputFormat: options.inputFormat || s.inputFormat || "text",
            jsonSchema: options.jsonSchema || s.jsonSchema,
            includePartialMessages: options.includePartialMessages || s.includePartialMessages,
            systemPrompt: options.systemPrompt || s.systemPrompt,
            appendSystemPrompt: options.appendSystemPrompt || s.appendSystemPrompt,
            betaHeaders: options.betas || s.betaHeaders || [],
            agent: options.agent || s.agent,
            fallbackModel: options.fallbackModel || s.fallbackModel,
            enableAuthStatus: options.enableAuthStatus || s.enableAuthStatus,
            mcpClientCount: options.mcpConfig ? options.mcpConfig.length : 0,
            sessionPersistence: options.sessionPersistence !== false
        };

        if (options.addDir) {
            // Add directories to allowed list
            const rules = next.toolPermissionContext.alwaysAllowRules.cli || [];
            next.toolPermissionContext.alwaysAllowRules.cli = Array.from(new Set([...rules, ...options.addDir]));
        }

        if (options.allowedTools) {
            const rules = next.toolPermissionContext.alwaysAllowRules.cli || [];
            next.toolPermissionContext.alwaysAllowRules.cli = Array.from(new Set([...rules, ...options.allowedTools]));
        }

        if (options.disallowedTools) {
            const rules = next.toolPermissionContext.alwaysDenyRules.cli || [];
            next.toolPermissionContext.alwaysDenyRules.cli = Array.from(new Set([...rules, ...options.disallowedTools]));
        }

        return next;
    });

    // Handle session resumption (continue/resume/fork)
    let initialMessages: any[] = [];
    const { setSessionId } = await import("../services/session/sessionStore.js");

    if (options.continue || options.resume) {
        const { loadSession } = await import("../services/session/SessionLoader.js");
        let sessionIdToLoad = typeof options.resume === "string" ? options.resume : undefined;

        if (options.continue && !sessionIdToLoad) {
            // Find most recent session
            // sessionIdToLoad = ...
        }

        const sessionData = loadSession(sessionIdToLoad);
        if (sessionData) {
            initialMessages = sessionData.messages;
            if (options.forkSession) {
                const { randomUUID } = await import("node:crypto");
                setSessionId(randomUUID());
            } else {
                setSessionId(sessionData.sessionId);
            }
        } else if (options.resume && typeof options.resume === "string") {
            console.error(`Session ${options.resume} not found.`);
            process.exit(1);
        }
    } else if (options.sessionId) {
        setSessionId(options.sessionId);
    } else {
        // New session
        const { randomUUID } = await import("node:crypto");
        setSessionId(randomUUID());
    }

    return { prompt, options, initialMessages };
}
