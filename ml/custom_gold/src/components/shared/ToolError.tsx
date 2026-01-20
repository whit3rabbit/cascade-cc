import React from "react";
import { Box } from "ink";
import { Text } from "../../vendor/inkText.js";
import { stripUnderline } from "../../services/terminal/terminalUtils.js";
import { Indent } from "./Indent.js";

const MAX_ERROR_LINES = 10;

function extractTagContent(text: string, tagName: string): string | null {
    if (!text.trim() || !tagName.trim()) return null;

    const safeTag = tagName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const tagRegex = new RegExp(`<${safeTag}(?:\\s+[^>]*)?>([\\s\\S]*?)<\\/${safeTag}>`, "gi");
    const openTagRegex = new RegExp(`<${safeTag}(?:\\s+[^>]*?)?>`, "gi");
    const closeTagRegex = new RegExp(`<\\/${safeTag}>`, "gi");

    let match: RegExpExecArray | null;
    let scanStart = 0;

    while ((match = tagRegex.exec(text)) !== null) {
        const content = match[1];
        const prefix = text.slice(scanStart, match.index);

        let depth = 0;
        openTagRegex.lastIndex = 0;
        while (openTagRegex.exec(prefix) !== null) depth++;
        closeTagRegex.lastIndex = 0;
        while (closeTagRegex.exec(prefix) !== null) depth--;

        if (depth === 0 && content) return content;
        scanStart = match.index + match[0].length;
    }

    return null;
}

function stripSandboxViolations(text: string): string {
    return text.replace(/<sandbox_violations>[\s\S]*?<\/sandbox_violations>/g, "");
}

/**
 * Specialized error component for tool execution failures.
 */
export const ToolError: React.FC<{
    result: any;
    verbose?: boolean;
}> = ({ result, verbose }) => {
    let errorMessage: string;

    if (typeof result !== "string") {
        errorMessage = "Tool execution failed";
    } else {
        const rawError = extractTagContent(result, "tool_use_error") ?? result;
        const cleanError = stripSandboxViolations(rawError).trim();

        if (!verbose && cleanError.includes("InputValidationError: ")) {
            errorMessage = "Invalid tool parameters";
        } else if (cleanError.startsWith("Error: ")) {
            errorMessage = cleanError;
        } else {
            errorMessage = `Error: ${cleanError}`;
        }
    }

    const lines = errorMessage.split("\n");
    const displayLines = verbose ? lines : lines.slice(0, MAX_ERROR_LINES);
    const hiddenCount = lines.length - displayLines.length;

    return (
        <Indent>
            <Box flexDirection="column">
                <Text color="error">
                    {stripUnderline(displayLines.join("\n"))}
                </Text>
                {!verbose && hiddenCount > 0 && (
                    <Box>
                        <Text dimColor>
                            … +{hiddenCount} {hiddenCount === 1 ? "line" : "lines"} (
                        </Text>
                        <Text dimColor bold>ctrl+o</Text>
                        <Text dimColor> to see all)</Text>
                    </Box>
                )}
            </Box>
        </Indent>
    );
};
