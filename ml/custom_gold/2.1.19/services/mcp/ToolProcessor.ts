/**
 * File: src/services/mcp/ToolProcessor.ts
 * Role: Prepares and transforms tool definitions and inputs for Claude conversations.
 */

// Basic interface for tool configuration
interface ToolConfig {
    name: string;
    prompt: (context: any) => Promise<string>;
    inputJSONSchema?: any;
    inputSchema?: any;
    strict?: boolean;
    input_examples?: any[];
    [key: string]: any;
}

// Basic interface for context passed to tool prompt
interface ToolContext {
    getToolPermissionContext?: any;
    tools?: any;
    agents?: any;
    [key: string]: any;
}

export interface ToolDetails {
    name: string;
    description: string;
    input_schema: any;
    strict?: boolean;
    input_examples?: any[];
}

/**
 * Prepares tool description and input schema for a tool.
 */
export async function prepareToolForConversation(toolConfig: ToolConfig, context: ToolContext): Promise<ToolDetails> {
    // Basic tool preparation logic
    const toolDetails: ToolDetails = {
        name: toolConfig.name,
        description: await toolConfig.prompt({
            getToolPermissionContext: context.getToolPermissionContext,
            tools: context.tools,
            agents: context.agents,
        }),
        input_schema: toolConfig.inputJSONSchema || toolConfig.inputSchema,
    };

    if (toolConfig.strict) toolDetails.strict = true;
    if (toolConfig.input_examples) toolDetails.input_examples = toolConfig.input_examples;

    return toolDetails;
}

/**
 * Transforms tool input data based on tool name before sending to the tool implementation.
 * This is where high-level 'Claude-friendly' inputs are converted to low-level system inputs.
 */
export function transformToolInput(toolDefinition: { name: string }, inputData: any, _context?: any): any {
    switch (toolDefinition.name) {
        case "bash":
            {
                const { command, timeout, description, run_in_background } = inputData;
                // Remove common prefix if present (artifact of some prompts)
                let sanitizedCommand = command ? command.replace(/^cd .*? && /, "") : command;

                return {
                    command: sanitizedCommand,
                    description,
                    ...(timeout ? { timeout } : {}),
                    ...(run_in_background ? { run_in_background } : {}),
                };
            }
        case "file_write":
            {
                // Transformation for multi-edit or legacy fields
                return {
                    file_path: inputData.file_path,
                    old_string: inputData.old_string,
                    new_string: inputData.new_string,
                    replace_all: inputData.replace_all || false
                };
            }
        default:
            return inputData;
    }
}
