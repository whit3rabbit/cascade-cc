import chalk from "chalk";
import stringWidth from "string-width";

/**
 * Terminal utilities for title management and content folding.
 */

export function setTerminalTitle(title: string): void {
    if (process.env.CLAUDE_CODE_DISABLE_TERMINAL_TITLE === "true") return;

    const formattedTitle = title ? `✳ ${title}` : "";
    if (process.platform === "win32") {
        process.title = formattedTitle;
    } else {
        process.stdout.write(`\x1B]0;${formattedTitle}\x07`);
    }
}

/**
 * Uses the LLM to analyze the message and update the terminal title if a new topic is detected.
 */
export async function updateTitleFromTopic(message: string): Promise<void> {
    if (message.startsWith("<local-command-stdout>")) return;

    // This assumes a call to a chat function that returns a completion
    // logic simplified here as the actual implementation depends on chat utilities
}

export function clearScreen(): Promise<void> {
    return new Promise((resolve) => {
        process.stdout.write("\x1B[2J\x1B[3J\x1B[H", () => resolve());
    });
}

const ABOVE_THE_FOLD_LINES = 3;
const FOLD_THRESHOLD = 9;

function sliceByColumns(text: string, startColumn: number, endColumn: number): string {
    if (startColumn <= 0 && endColumn >= stringWidth(text)) return text;

    let output = "";
    let currentWidth = 0;
    // @ts-ignore - Intl.Segmenter exists in Node 18+
    const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });

    for (const { segment } of segmenter.segment(text)) {
        const segmentWidth = stringWidth(segment);
        const nextWidth = currentWidth + segmentWidth;
        if (nextWidth <= startColumn) {
            currentWidth = nextWidth;
            continue;
        }
        if (currentWidth >= endColumn) break;

        output += segment;
        currentWidth = nextWidth;

        if (currentWidth >= endColumn) break;
    }

    return output;
}

export function foldLines(text: string, maxColumns: number): { aboveTheFold: string; remainingLines: number } {
    const lines = text.split("\n");
    const foldedLines: string[] = [];

    for (const line of lines) {
        const visualLength = stringWidth(line);
        if (visualLength <= maxColumns) {
            foldedLines.push(line.trimEnd());
        } else {
            let offset = 0;
            while (offset < visualLength) {
                const chunk = sliceByColumns(line, offset, offset + maxColumns);
                foldedLines.push(chunk.trimEnd());
                offset += maxColumns;
            }
        }
    }

    const overflow = foldedLines.length - ABOVE_THE_FOLD_LINES;
    if (overflow === 1) {
        return {
            aboveTheFold: foldedLines.slice(0, ABOVE_THE_FOLD_LINES + 1).join("\n").trimEnd(),
            remainingLines: 0
        };
    }

    return {
        aboveTheFold: foldedLines.slice(0, ABOVE_THE_FOLD_LINES).join("\n").trimEnd(),
        remainingLines: Math.max(0, overflow)
    };
}

function getExpandHintLabel(): string {
    return "(ctrl+o to expand)";
}

export function foldContent(text: string, columns: number): string {
    const trimmed = text.trimEnd();
    if (!trimmed) return "";

    const { aboveTheFold, remainingLines } = foldLines(trimmed, Math.max(columns - FOLD_THRESHOLD, 10));

    if (remainingLines > 0) {
        return `${aboveTheFold}\n${chalk.dim(`… +${remainingLines} lines ${getExpandHintLabel()}`)}`;
    }
    return aboveTheFold;
}

function tryFormatJsonLine(text: string): string {
    try {
        const parsed = JSON.parse(text);
        const minified = JSON.stringify(parsed);
        const normalizedInput = text.replace(/\s+/g, "");
        const normalizedMinified = minified.replace(/\s+/g, "");
        if (normalizedInput !== normalizedMinified) return text;
        return JSON.stringify(parsed, null, 2);
    } catch {
        return text;
    }
}

export function formatJsonInBlocks(text: string): string {
    return text.split("\n").map(tryFormatJsonLine).join("\n");
}


export function stripUnderline(text: string): string {
    return text.replace(/\u001b\[([0-9]+;)*4(;[0-9]+)*m|\u001b\[4(;[0-9]+)*m|\u001b\[([0-9]+;)*4m/g, "");
}

export function getCwd(): string {
    return process.cwd();
}
