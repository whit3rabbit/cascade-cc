import * as React from "react";
import * as fs from "node:fs";
import { Box } from "../../vendor/inkBox.js";
import { Text } from "../../vendor/inkText.js";
import { StackCleaner } from "../../utils/shared/stackCleaner.js";

const stackCleaner = new StackCleaner();

/**
 * Component for rendering error stacks with code snippets.
 * Deobfuscated from Rl1 in chunk_202.ts.
 */
export const ErrorDisplay: React.FC<{ error: Error }> = ({ error }) => {
    const stack = error.stack || "";
    const lines = stack.split("\n").slice(1);
    const firstFrame = lines.length > 0 ? parseStackLine(lines[0]) : null;

    let codeSnippet: { line: number; value: string }[] = [];
    if (firstFrame?.file && fs.existsSync(firstFrame.file)) {
        try {
            const content = fs.readFileSync(firstFrame.file, "utf8");
            // Simplified snippet extraction
            const allLines = content.split("\n");
            const start = Math.max(0, firstFrame.line - 3);
            const end = Math.min(allLines.length, firstFrame.line + 2);
            codeSnippet = allLines.slice(start, end).map((val, idx) => ({
                line: start + idx + 1,
                value: val
            }));
        } catch (e) {
            // Ignore
        }
    }

    return (
        <Box flexDirection="column" padding={1}>
            <Box>
                <Text backgroundColor="red" color="white" bold> ERROR </Text>
                <Text> {error.message}</Text>
            </Box>

            {firstFrame && (
                <Box marginTop={1}>
                    <Text dimColor>{firstFrame.file}:{firstFrame.line}:{firstFrame.column}</Text>
                </Box>
            )}

            {codeSnippet.length > 0 && (
                <Box marginTop={1} flexDirection="column">
                    {codeSnippet.map(({ line, value }) => (
                        <Box key={line}>
                            <Box width={5}>
                                <Text
                                    dimColor={line !== firstFrame?.line}
                                    backgroundColor={line === firstFrame?.line ? "red" : undefined}
                                    color={line === firstFrame?.line ? "white" : undefined}
                                >
                                    {String(line).padStart(4)}:
                                </Text>
                            </Box>
                            <Text
                                backgroundColor={line === firstFrame?.line ? "red" : undefined}
                                color={line === firstFrame?.line ? "white" : undefined}
                            >
                                {" " + value}
                            </Text>
                        </Box>
                    ))}
                </Box>
            )}

            {lines.length > 0 && (
                <Box marginTop={1} flexDirection="column">
                    {lines.map((line, i) => (
                        <Text key={i} dimColor>- {line.trim()}</Text>
                    ))}
                </Box>
            )}
        </Box>
    );
};

function parseStackLine(line: string) {
    const match = line.match(/at (?:(.+)\s+\()?(?:(.+?):(\d+):(\d+))\)?/);
    if (!match) return null;
    return {
        function: match[1] || "<anonymous>",
        file: match[2],
        line: parseInt(match[3], 10),
        column: parseInt(match[4], 10)
    };
}
