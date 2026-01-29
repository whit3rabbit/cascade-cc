/**
 * File: src/services/mcp/McpContent.ts
 * Role: Processing and parsing Claude's structured output for MCP interactions.
 */

import { parseBalancedJSON } from '../../utils/json/BalancedParser.js';

export interface McpContentItem {
    type: string;
    text?: string;
    parsed?: any;
    [key: string]: any;
}

export interface McpMessage {
    content: McpContentItem[];
    [key: string]: any;
}

export interface OutputFormatOptions {
    type?: string;
    parse?: boolean | ((text: string) => any);
}

export interface McpOptions {
    output_format?: OutputFormatOptions;
}

/**
 * Processes output from Claude, optionally parsing structured data (JSON).
 * 
 * @param {McpMessage} message - The message object containing content blocks.
 * @param {McpOptions} [options] - Configuration for parsing output.
 * @returns {McpMessage & { parsed_output: any }}
 */
export function processOutput(message: McpMessage, options?: McpOptions): any {
    if (!options || !options.output_format || !options.output_format.parse) {
        return {
            ...message,
            content: message.content.map(item => {
                if (item.type === "text") {
                    return { ...item, parsed: null };
                }
                return item;
            }),
            parsed_output: null
        };
    }

    return parseStructuredOutput(message, options);
}

/**
 * Parses structured output from message content.
 * 
 * @param {McpMessage} message - The message object.
 * @param {McpOptions} options - The options containing the output format.
 * @returns {any} The updated message with parsed content.
 */
export function parseStructuredOutput(message: McpMessage, options: McpOptions): any {
    let parsedOutput: any = null;
    const updatedContent = message.content.map(item => {
        if (item.type === "text" && item.text) {
            const parsed = parseTextOutput(options, item.text);
            if (parsedOutput === null) parsedOutput = parsed;
            return { ...item, parsed };
        }
        return item;
    });

    return {
        ...message,
        content: updatedContent,
        parsed_output: parsedOutput
    };
}

/**
 * Parses a single text block based on the expected output format.
 * 
 * @param {McpOptions} options - The options containing the output format.
 * @param {string} text - The raw text to parse.
 * @returns {any | null}
 */
export function parseTextOutput(options: McpOptions, text: string): any | null {
    if (options.output_format?.type !== "json_schema") return null;

    try {
        if (typeof options.output_format.parse === 'function') {
            return options.output_format.parse(text);
        }
        // Fallback to our balanced JSON parser for robust handling of potentially 
        // incomplete or slightly malformed tool outputs.
        return parseBalancedJSON(text);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        console.error(`Failed to parse structured output: ${message}`);
        return null;
    }
}
