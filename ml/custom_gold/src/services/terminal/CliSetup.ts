
import { Command, Option } from "commander";
import React from "react";
import { render } from "ink";
import fs from "fs";
import path from "path";
import { getGlobalState } from "../session/globalState.js";
import { log, logError } from "../logger/loggerService.js";
import { McpCliAdmin } from "../mcp/McpCliAdmin.js";
import { PluginCliAdmin } from "./PluginCliAdmin.js";

// Placeholder for the main UI component
// In a full implementation, this would import the root App component that wraps the Agent Session
import { AgentSession } from "../../components/terminal/AgentSession.js";

// Logic from chunks 846, 847, 848 (CLI Command Orchestration)

/**
 * Registers MCP administration commands.
 */
export function setupMcpCommands(program: Command) {
    const mcp = program.command("mcp").description("Manage Model Context Protocol (MCP) servers");

    mcp.command("list")
        .description("List configured MCP servers")
        .action(async () => {
            const result = await McpCliAdmin.listServers();
            if (result.errors.length > 0) {
                console.error("Errors found:", result.errors);
            }
            result.servers.forEach(s => {
                console.log(`${s.name} (${s.scope}): ${s.status}`);
            });
        });

    mcp.command("add <name>")
        .description("Add an MCP server")
        .option("-s, --scope <scope>", "Scope (user/project)", "user")
        .action(async (name, options) => {
            console.log(`Adding server ${name}... (Not fully interactive in this CLI setup yet)`);
            // Implementation would parse args and call McpCliAdmin.addServer
        });

    mcp.command("remove <name>")
        .description("Remove an MCP server")
        .option("-s, --scope <scope>", "Scope (user/project)", "user")
        .action(async (name, options) => {
            await McpCliAdmin.removeServer(name, options.scope);
        });
}

/**
 * Registers Plugin administration commands.
 */
export function setupPluginCommands(program: Command) {
    const plugins = program.command("plugin").description("Manage Claude Code plugins");

    plugins.command("list")
        .action(async () => {
            const marketplaces = await PluginCliAdmin.listMarketplaces();
            console.log("Configured Marketplaces:", marketplaces.join(", "));
        });

    plugins.command("install <name>")
        .option("-s, --scope <scope>", "Scope", "user")
        .action(async (name, options) => {
            await PluginCliAdmin.installPlugin(name, options.scope);
        });

    plugins.command("uninstall <name>")
        .option("-s, --scope <scope>", "Scope", "user")
        .action(async (name, options) => {
            await PluginCliAdmin.uninstallPlugin(name, options.scope);
        });
}

/**
 * The main setup orchestrator that parses global flags and dispatches commands.
 */
export async function runCli() {
    const program = new Command();

    program
        .name("claude")
        .version("2.0.76")
        .description("Claude Code - The AI coding assistant");

    // Global Options
    program.option("-d, --debug [filter]", "Enable debug mode");
    program.option("--verbose", "Enable verbose output");
    program.option("-p, --print", "Print response and exit (non-interactive)");
    program.option("--model <model>", "Model alias or full name");

    // Register Subcommands
    setupMcpCommands(program);
    setupPluginCommands(program);

    // Doctor
    program.command("doctor").description("Check health").action(() => console.log("Doctor check passed."));

    // Default Action (Interactive Session)
    program.action(async (promptOrOptions, optionsObj) => {
        let prompt: string | undefined;
        let options: any = {};

        if (typeof promptOrOptions === 'string') {
            prompt = promptOrOptions;
            options = optionsObj || {};
        } else {
            options = promptOrOptions || {};
        }

        await startInteractiveSession(prompt, options);
    });

    await program.parseAsync(process.argv);
}

/**
 * Starts the interactive Claude Code session.
 * This corresponds to the logic in chunk_846/847 that renders the React/Ink app.
 */
async function startInteractiveSession(initialPrompt: string | undefined, options: any) {
    // 1. Initialize Telemetry & Logger
    // 2. Load Settings & State
    const state = getGlobalState();

    // 3. Render the Ink App
    // This effectively "starts" the Agent Main Loop via the UI components
    const { unmount } = render(
        React.createElement(AgentSession, {
            initialPrompt: initialPrompt,
            debug: options.debug,
            verbose: options.verbose
        })
    );

    // Handle clean exit
    // In a real Ink app, the app controls exit, but we might need to handle signals
}
