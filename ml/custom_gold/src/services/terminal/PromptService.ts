
// Logic from chunk_579.ts (Prompt Generation Services)

import os from "node:os";
import { execa } from "execa"; // Assuming execa is used for shell commands

// --- OS Version Helper (gY7) ---
export async function getOsVersion(): Promise<string> {
    try {
        if (process.platform === "win32") {
            return os.release();
        }
        const { stdout } = await execa("uname", ["-sr"]);
        return stdout.trim();
    } catch {
        return "unknown";
    }
}

// --- Environment Info Generator (TW9) ---
export async function getEnvironmentInfo(modelId: string, additionalDirs: string[] = []): Promise<string> {
    const isGit = true; // Placeholder for git check logic
    const platform = process.platform;
    const osVersion = await getOsVersion();
    const today = new Date().toDateString();

    const modelInfo = modelId.includes("claude-3")
        ? `You are powered by the model named Claude 3. The exact model ID is ${modelId}.`
        : `You are powered by the model ${modelId}.`;

    const cutoffInfo = (modelId.includes("claude-3-5") || modelId.includes("claude-3-opus"))
        ? "\n\nAssistant knowledge cutoff is January 2025."
        : "";

    const additionalDirsInfo = additionalDirs.length > 0
        ? `Additional working directories: ${additionalDirs.join(", ")}\n`
        : "";

    return `Here is useful information about the environment you are running in:
<env>
Working directory: ${process.cwd()}
Is directory a git repo: ${isGit ? "Yes" : "No"}
${additionalDirsInfo}Platform: ${platform}
OS Version: ${osVersion}
Today's date: ${today}
</env>
${modelInfo}${cutoffInfo}
<claude_background_info>
The most recent frontier Claude model is Claude 3.5 Sonnet (model ID: 'claude-3-5-sonnet-20241022').
</claude_background_info>
`;
}

// --- MCP Instructions Formatter (hY7) ---
export function formatMcpInstructions(mcpServers: any[]): string {
    const serversWithInstructions = mcpServers.filter(s => s.status === "connected" && s.instructions);
    if (serversWithInstructions.length === 0) return "";

    return `
# MCP Server Instructions

The following MCP servers have provided instructions for how to use their tools and resources:

${serversWithInstructions.map(s => `## ${s.name}\n${s.instructions}`).join("\n\n")}
`;
}

// --- Scratchpad Instructions (mY7) ---
export function getScratchpadPrompt(scratchpadDir: string | undefined): string {
    if (!scratchpadDir) return "";

    return `
# Scratchpad Directory

IMPORTANT: Always use this scratchpad directory for temporary files instead of \`/tmp\` or other system temp directories:
\`${scratchpadDir}\`

Use this directory for ALL temporary file needs:
- Storing intermediate results or data during multi-step tasks
- Writing temporary scripts or configuration files
- Saving outputs that don't belong in the user's project
- Creating working files during analysis or processing
- Any file that would otherwise go to \`/tmp\`

Only use \`/tmp\` if the user explicitly requests it.

The scratchpad directory is session-specific, isolated from the user's project, and can be used freely without permission prompts.
`;
}

// --- MCP CLI Instructions (jW9) ---
export function getMcpCliPrompt(mcpTools: any[]): string {
    if (!mcpTools || mcpTools.length === 0) return "";

    return `
# MCP CLI Command

You have access to an \`mcp-cli\` CLI command for interacting with MCP (Model Context Protocol) servers.

**MANDATORY PREREQUISITE - THIS IS A HARD REQUIREMENT**

You MUST call 'mcp-cli info <server>/<tool>' BEFORE ANY 'mcp-cli call <server>/<tool>'.

**NEVER** make an mcp-cli call without checking the schema first.
**ALWAYS** run mcp-cli info first, THEN make the call.

Commands:
\`\`\`bash
mcp-cli info <server>/<tool>           # REQUIRED before ANY call - View JSON schema
mcp-cli call <server>/<tool> '<json>'  # Only run AFTER mcp-cli info
mcp-cli servers                        # List all connected MCP servers
mcp-cli tools [server]                 # List available tools
\`\`\`

Available MCP tools:
${mcpTools.map(t => `- ${t.name}`).join("\n")}
`;
}
