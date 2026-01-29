/**
 * File: src/tools/ToolGroups.ts
 * Role: Categorization and grouping of available tools for UI display.
 */

export interface ToolGroup {
    id: string;
    name: string;
    toolNames: string[];
    isMcp?: boolean;
}

/**
 * Standard tool categories for the Claude CLI.
 */
export const TOOL_GROUPS: Record<string, ToolGroup> = {
    READ: {
        id: "read",
        name: "Read-only tools",
        toolNames: ["read_file", "list_dir", "grep_search", "find_by_name", "read_url_content", "ls", "cat"]
    },
    EDIT: {
        id: "edit",
        name: "Edit tools",
        toolNames: ["replace_file_content", "multi_replace_file_content", "write_to_file", "sed"]
    },
    EXECUTION: {
        id: "execution",
        name: "Execution tools",
        toolNames: ["run_command", "send_command_input"]
    },
    MCP: {
        id: "mcp",
        name: "MCP tools",
        isMcp: true,
        toolNames: [] // Populated dynamically
    },
    OTHER: {
        id: "other",
        name: "Other tools",
        toolNames: ["Teammate"]
    }
};

/**
 * Helper to check which group a tool belongs to.
 */
export function getToolGroup(toolName: string): ToolGroup {
    for (const group of Object.values(TOOL_GROUPS)) {
        if (group.toolNames.includes(toolName)) return group;
    }
    return TOOL_GROUPS.OTHER;
}
