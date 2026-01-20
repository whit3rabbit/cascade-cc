
// Logic from chunk_506.ts (Message Content Rendering)

import React from 'react';
import { Box, Text } from 'ink';
import { relative, sep } from 'path';
import { figures } from '../../vendor/terminalFigures.js'; // Assuming mapped

// --- Constants & Helpers ---
const CWD_RESET_REGEX = /(?:^|\n)(Shell cwd was reset to .+)$/;

function cleanStderr(stderr: string) {
    if (!stderr.match(/<sandbox_violations>([\s\S]*?)<\/sandbox_violations>/)) {
        // simple clean if no tags?
        return { cleanedStderr: stderr };
    }
    // Remove violations tags (X91 logic placeholder)
    return { cleanedStderr: stderr.replace(/<sandbox_violations>[\s\S]*?<\/sandbox_violations>/g, "").trim() };
}

function processStderr(stderr: string) {
    const { cleanedStderr } = cleanStderr(stderr);
    const match = cleanedStderr.match(CWD_RESET_REGEX);
    if (!match) {
        return { cleanedStderr, cwdResetWarning: null };
    }
    return {
        cleanedStderr: cleanedStderr.replace(CWD_RESET_REGEX, "").trim(),
        cwdResetWarning: match[1] ?? null
    };
}

// --- Thinking Display (lo2) ---
export function ThinkingDisplay({ thinking, addMargin, isVerbose, isTranscriptMode }: any) {
    if (!thinking) return null;

    if (!(isTranscriptMode || isVerbose)) {
        return (
            <Box marginTop={addMargin ? 1 : 0}>
                <Text dimColor italic>∴ Thinking (ctrl+o to expand)</Text>
            </Box>
        );
    }

    return (
        <Box flexDirection="column" marginTop={addMargin ? 1 : 0} width="100%" gap={1}>
            <Text dimColor italic>∴ Thinking…</Text>
            <Box paddingLeft={2}>
                <Text dimColor italic>{thinking}</Text>
            </Box>
        </Box>
    );
}

// --- Bash Output (w4A) ---
export function ShellOutput({ stdout, stderr, summary, isImage, backgroundTaskId, verbose, type }: any) {
    // Logic from w4A
    // const [theme] = useTheme(); // Stub theme

    let originalStdout = stdout || "";
    const { cleanedStderr, cwdResetWarning } = processStderr(stderr || "");

    // Image handling
    if (isImage) {
        return (
            <Box height={1}>
                <Text dimColor>[Image data detected and sent to Claude]</Text>
            </Box>
        );
    }

    // Background Task
    if (backgroundTaskId && !originalStdout && !cleanedStderr && !summary && !cwdResetWarning) {
        return (
            <Box height={1}>
                <Text dimColor>Running in the background (↓ manage)</Text>
            </Box>
        );
    }

    // Summary mode
    if (summary) {
        if (!verbose) {
            return (
                <Box flexDirection="column">
                    <Text>{summary}</Text>
                </Box>
            );
        }
        return (
            <Box flexDirection="column">
                <Text>{summary}</Text>
                {(originalStdout !== "" || cleanedStderr !== "" || cwdResetWarning) && (
                    <Box flexDirection="column" marginTop={1}>
                        <Text bold>=== Original Output ===</Text>
                        {originalStdout !== "" && <Text>{originalStdout}</Text>}
                        {cleanedStderr !== "" && <Text color="red">{cleanedStderr}</Text>}
                        {cwdResetWarning && <Text color="yellow">{cwdResetWarning}</Text>}
                    </Box>
                )}
            </Box>
        );
    }

    // Default verbose output
    return (
        <Box flexDirection="column">
            {originalStdout !== "" && <Text>{originalStdout}</Text>}
            {cleanedStderr !== "" && <Text color="red">{cleanedStderr}</Text>}
            {cwdResetWarning && <Text color="yellow">{cwdResetWarning}</Text>}
        </Box>
    );
}

// --- Local Command Output (So2) ---
export function LocalCommandOutput({ content }: any) {
    // const stdout = extractTag(content, "local-command-stdout");
    // const stderr = extractTag(content, "local-command-stderr");
    // Simplified parsing for now:
    const stdout = content.includes("<local-command-stdout>") ? content.split("<local-command-stdout>")[1].split("</local-command-stdout>")[0] : "";
    const stderr = content.includes("<local-command-stderr>") ? content.split("<local-command-stderr>")[1].split("</local-command-stderr>")[0] : "";

    if (!stdout && !stderr) {
        return <Box><Text dimColor>(No output)</Text></Box>;
    }

    return (
        <Box flexDirection="column">
            {stdout && stdout.trim().split('\n').map((line: string, i: number) => (
                <Box key={`out-${i}`}>
                    <Text color="white">{i === 0 ? "  ⎿  " : "     "}</Text>
                    <Text>{line}</Text>
                </Box>
            ))}
            {stderr && stderr.trim().split('\n').map((line: string, i: number) => (
                <Box key={`err-${i}`}>
                    <Text color="red">{i === 0 ? "  ⎿  " : "     "}</Text>
                    <Text color="red">{line}</Text>
                </Box>
            ))}
        </Box>
    );
}

// --- Diagnostic Renderer (oo2) ---
export function DiagnosticRenderer({ attachment, verbose }: any) {
    const { files } = attachment;
    if (files.length === 0) return null;

    const count = files.reduce((acc: number, f: any) => acc + f.diagnostics.length, 0);
    const fileCount = files.length;

    if (verbose) {
        return (
            <Box flexDirection="column">
                {files.map((file: any, i: number) => (
                    <React.Fragment key={i}>
                        <Box>
                            <Text dimColor wrap="wrap">
                                <Text bold>{relative(process.cwd(), file.uri.replace("file://", ""))}</Text>
                                <Text dimColor> ({file.uri.split(':')[0]}):</Text>
                            </Text>
                        </Box>
                        {file.diagnostics.map((diag: any, j: number) => (
                            <Box key={j} paddingLeft={2}>
                                <Text dimColor wrap="wrap">
                                    {diag.severity === 1 ? figures.cross : figures.warning} [Line {diag.range.start.line + 1}:{diag.range.start.character + 1}] {diag.message}
                                </Text>
                            </Box>
                        ))}
                    </React.Fragment>
                ))}
            </Box>
        );
    }

    return (
        <Box>
            <Text dimColor wrap="wrap">
                Found <Text bold>{count}</Text> new diagnostic {count === 1 ? "issue" : "issues"} in {fileCount} {fileCount === 1 ? "file" : "files"} (ctrl+o to expand)
            </Text>
        </Box>
    );
}

// --- Image Renderer Stub (po2) ---
export function ImageRenderer({ imageId, addMargin }: any) {
    const label = imageId ? `[Image #${imageId}]` : "[Image]";
    // Actual image rendering omitted for now
    if (addMargin) {
        return <Box marginTop={1}><Text>{label}</Text></Box>;
    }
    return <Box><Text>{label}</Text></Box>;
}


// --- Main Attachment Switch (so2) ---
export function AttachmentRenderer({ attachment, addMargin, verbose }: any) {
    switch (attachment.type) {
        case "directory":
            return (
                <Box marginTop={addMargin ? 1 : 0}>
                    <Text>Listed directory <Text bold>{relative(process.cwd(), attachment.path)}{sep}</Text></Text>
                </Box>
            );
        case "file":
        case "already_read_file":
            const isText = attachment.content.type === "text";
            const info = isText ? `${attachment.content.file.numLines} lines` : "Binary";
            return (
                <Box marginTop={addMargin ? 1 : 0}>
                    <Text>Read <Text bold>{relative(process.cwd(), attachment.filename)}</Text> ({info})</Text>
                </Box>
            );
        case "compact_file_reference":
            return (
                <Box marginTop={addMargin ? 1 : 0}>
                    <Text>Referenced file <Text bold>{relative(process.cwd(), attachment.filename)}</Text></Text>
                </Box>
            );
        case "selected_lines_in_ide":
            return (
                <Box marginTop={addMargin ? 1 : 0}>
                    <Text>⧉ Selected <Text bold>{attachment.lineEnd - attachment.lineStart + 1}</Text> lines from <Text bold>{relative(process.cwd(), attachment.filename)}</Text> in {attachment.ideName}</Text>
                </Box>
            );
        case "todo":
            // Only show if post-compact
            if (attachment.context === "post-compact") {
                return (
                    <Box marginTop={addMargin ? 1 : 0}>
                        <Text>Todo list read ({attachment.itemCount} {attachment.itemCount === 1 ? "item" : "items"})</Text>
                    </Box>
                );
            }
            return null;
        case "invoked_skills":
            if (attachment.skills.length === 0) return null;
            return (
                <Box marginTop={addMargin ? 1 : 0}>
                    <Text>Slash commands restored ({attachment.skills.map((s: any) => s.name).join(", ")})</Text>
                </Box>
            );
        case "diagnostics":
            return <DiagnosticRenderer attachment={attachment} verbose={verbose} />;
        case "task_status":
            return (
                <Box flexDirection="row" width="100%" marginTop={1} paddingLeft={2}>
                    <Text dimColor>Task "<Text bold>{attachment.description}</Text>" {attachment.status === "completed" ? "completed in background" : attachment.status} {attachment.deltaSummary ? `: ${attachment.deltaSummary}` : ""}</Text>
                </Box>
            );
        case "mcp_resource":
            return (
                <Box marginTop={addMargin ? 1 : 0}>
                    <Text>Read MCP resource <Text bold>{attachment.name}</Text> from {attachment.server}</Text>
                </Box>
            );
        case "hook_system_message":
            return (
                <Box marginTop={addMargin ? 1 : 0}>
                    <Text>{attachment.hookName} says: {attachment.content}</Text>
                </Box>
            );
        case "hook_error_during_execution":
            if (!verbose) return null;
            return <Box><Text color="yellow">{attachment.hookName} hook warning: {attachment.content}</Text></Box>;
        default:
            // Fallback for handled but silent types or unknown types
            return null;
    }
}

// --- Main Message Content Switch (L4A) ---
export function MessageContentRenderer({ param, addMargin, isVerbose, thinkingMetadata }: any) {
    // If text field contains tags, it might need special handling (e.g. bash-stdout)
    // In canonical implementation, this parsing logic would be more robust (To2, So2, etc)
    const text = param.text || "";

    if (text === "(No content)" || text.trim() === "(No content)") return null;

    if (text.includes("<bash-stdout") || text.includes("<bash-stderr")) {
        // Extract content
        const stdout = text.includes("<bash-stdout>") ? text.split("<bash-stdout>")[1].split("</bash-stdout>")[0] : "";
        const stderr = text.includes("<bash-stderr>") ? text.split("<bash-stderr>")[1].split("</bash-stderr>")[0] : "";
        return <ShellOutput stdout={stdout} stderr={stderr} verbose={isVerbose} />;
    }

    if (text.includes("<local-command-stdout") || text.includes("<local-command-stderr")) {
        return <LocalCommandOutput content={text} />;
    }

    if (text.includes("<background-task-output>")) {
        const out = text.split("<background-task-output>")[1].split("</background-task-output>")[0];
        return <Box><Text dimColor>{out}</Text></Box>;
    }

    if (param.thinking) {
        return <ThinkingDisplay thinking={param.thinking} addMargin={addMargin} isVerbose={isVerbose} />;
    }

    if (param.type === "redacted_thinking") {
        if (!isVerbose) return null;
        return <Box marginTop={addMargin ? 1 : 0}><Text dimColor italic>✻ Thinking…</Text></Box>;
    }

    // Default Text
    return (
        <Box marginTop={addMargin ? 1 : 0}>
            <Text>{text}</Text>
        </Box>
    );
}
