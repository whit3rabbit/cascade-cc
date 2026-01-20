
import * as fs from "fs";
import * as path from "path";
import { evaluatePromptBashCommands } from "../bash/PromptEvaluator.js";
import { parseFrontmatter } from "../../utils/markdown/MarkdownUtils.js";
import { logEvent } from "../telemetry/TelemetryService.js";

interface PluginCommandConfig {
    isSkillMode?: boolean;
}

// Helper to recursively scan for .md files
function scanDirectory(dir: string): string[] {
    let results: string[] = [];
    if (!fs.existsSync(dir)) return results;

    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            results = results.concat(scanDirectory(fullPath));
        } else if (entry.isFile() && entry.name.endsWith(".md")) {
            results.push(fullPath);
        }
    }
    return results;
}

export async function createCommandFromMarkdown(
    name: string,
    fileData: { filePath: string, baseDir: string, frontmatter: any, content: string },
    repository: string,
    manifest: any,
    pluginRootInHost: string,
    isSkill: boolean,
    config: PluginCommandConfig
): Promise<any> {
    try {
        const { frontmatter, content } = fileData;

        // Construct prompt object matching interface expected by PromptManager
        return {
            type: "prompt",
            name,
            description: frontmatter.description || "Plugin command",
            // ... map other fields
            contentLength: content.length,
            source: "plugin",
            pluginInfo: {
                pluginManifest: manifest,
                repository,
                path: fileData.filePath
            },
            isEnabled: () => true,
            isHidden: isSkill,
            async getPromptForCommand(args: string, context: any) {
                let promptText = config.isSkillMode
                    ? `Base directory for this skill: ${path.dirname(fileData.filePath)}\n\n${content}`
                    : content;

                if (args) {
                    if (promptText.includes("$ARGUMENTS")) {
                        promptText = promptText.replace(/\$ARGUMENTS/g, args);
                    } else {
                        promptText += `\n\nARGUMENTS: ${args}`;
                    }
                }

                if (pluginRootInHost) {
                    promptText = promptText.replace(/\$\{CLAUDE_PLUGIN_ROOT\}/g, pluginRootInHost);
                }

                promptText = await evaluatePromptBashCommands(promptText, context, `/${name}`);
                // Simplified evaluation for now

                return [{ type: "text", text: promptText }];
            }
        };
    } catch (error) {
        console.error(`Failed to create command from ${fileData.filePath}`, error);
        return null;
    }
}

export async function loadPluginCommands(
    plugin: any,
    config: PluginCommandConfig = { isSkillMode: false },
    ignoredPaths: Set<string> = new Set()
): Promise<any[]> {
    const commands: any[] = [];
    const pluginRoot = plugin.path;

    const processPath = async (p: string, isDefault: boolean) => {
        if (!fs.existsSync(p)) return;

        let files: string[] = [];
        if (fs.statSync(p).isDirectory()) {
            files = scanDirectory(p);
        } else if (p.endsWith(".md")) {
            files = [p];
        }

        for (const file of files) {
            if (ignoredPaths.has(file)) continue;

            try {
                const contentStr = fs.readFileSync(file, 'utf-8');
                const { frontmatter, content } = parseFrontmatter(contentStr);

                let commandName = "";
                if (frontmatter.name) commandName = frontmatter.name;
                else {
                    const relative = path.relative(isDefault ? (plugin.commandsPath || p) : path.dirname(p), file);
                    commandName = relative.replace(/\.md$/, "").replace(/\//g, ":");
                    if (isDefault) commandName = `${plugin.name}:${commandName}`;
                }

                // If overriding command name logic required from chunk_362, insert here
                // Simplified: usage plugin:command syntax

                const cmd = await createCommandFromMarkdown(
                    commandName,
                    { filePath: file, baseDir: pluginRoot, frontmatter, content },
                    plugin.repository || plugin.source,
                    plugin.manifest,
                    plugin.path,
                    config.isSkillMode || false,
                    config
                );
                if (cmd) commands.push(cmd);
            } catch (e) {
                console.error(`Failed to load command from ${file}:`, e);
            }
        }
    };

    if (plugin.commandsPath) {
        await processPath(plugin.commandsPath, true);
    }
    if (plugin.commandsPaths) {
        for (const p of plugin.commandsPaths) await processPath(p, false);
    }

    return commands;
}

export async function loadPluginSkills(
    plugin: any,
    ignoredPaths: Set<string> = new Set()
): Promise<any[]> {
    return loadPluginCommands(plugin, { isSkillMode: true }, ignoredPaths);
}

export async function loadPluginAgents(
    plugin: any,
    ignoredPaths: Set<string> = new Set()
): Promise<any[]> {
    // Agents logic is slightly different (l62 in chunk_362), creates agent definition objects
    const agents: any[] = [];
    const pluginRoot = plugin.path;

    const processPath = async (p: string, isDefault: boolean) => {
        if (!fs.existsSync(p)) return;

        let files: string[] = [];
        if (fs.statSync(p).isDirectory()) {
            files = scanDirectory(p);
        } else if (p.endsWith(".md")) {
            files = [p];
        }

        for (const file of files) {
            if (ignoredPaths.has(file)) continue;

            try {
                const contentStr = fs.readFileSync(file, 'utf-8');
                const { frontmatter, content } = parseFrontmatter(contentStr);

                const agentName = frontmatter.name || path.basename(file).replace(/\.md$/, "");
                const namespacedName = `${plugin.name}:${agentName}`;

                const agent = {
                    agentType: namespacedName,
                    whenToUse: frontmatter.description || frontmatter["when-to-use"] || `Agent from ${plugin.name} plugin`,
                    tools: frontmatter.tools, // parse if needed
                    getSystemPrompt: () => content.trim(),
                    source: "plugin",
                    plugin: plugin.name
                };
                agents.push(agent);
            } catch (e) {
                console.error(`Failed to load agent from ${file}:`, e);
            }
        }
    };

    if (plugin.agentsPath) {
        await processPath(plugin.agentsPath, true);
    }
    if (plugin.agentsPaths) {
        for (const p of plugin.agentsPaths) await processPath(p, false);
    }

    return agents;
}
