
import React, { useMemo } from "react";
import { Box } from "ink";
import { diffWordsWithSpace } from "diff";
import wrapAnsi from "wrap-ansi";
import { ThemedBox, ThemedText as Text } from "../terminal/ThemedComponents.js";
import { useTheme } from "../../services/terminal/themeManager.js";

export interface DiffLine {
    code: string;
    type: "add" | "remove" | "nochange";
    i: number;
    originalCode?: string;
    wordDiff?: boolean;
    matchedLine?: DiffLine;
}

export interface DiffHunk {
    lines: string[];
    oldStart: number;
    newStart: number;
    oldLines: number;
    newLines: number;
}

export interface DiffViewProps {
    patch: DiffHunk;
    dim?: boolean;
    width: number;
}

const WORD_DIFF_THRESHOLD = 0.4;

function wrapByWidth(text: string, width: number): string[] {
    return wrapAnsi(text, width, { hard: true, trim: false }).split("\n");
}

function parseHunkLines(lines: string[]): DiffLine[] {
    return lines.map((line) => {
        if (line.startsWith("+")) {
            return {
                code: line.slice(1),
                i: 0,
                type: "add",
                originalCode: line.slice(1)
            };
        }
        if (line.startsWith("-")) {
            return {
                code: line.slice(1),
                i: 0,
                type: "remove",
                originalCode: line.slice(1)
            };
        }
        return {
            code: line.slice(1),
            i: 0,
            type: "nochange",
            originalCode: line.slice(1)
        };
    });
}

function correlateDiffLines(lines: DiffLine[]): DiffLine[] {
    const result: DiffLine[] = [];
    let index = 0;

    while (index < lines.length) {
        const line = lines[index];
        if (!line) {
            index += 1;
            continue;
        }

        if (line.type === "remove") {
            const removed: DiffLine[] = [line];
            let cursor = index + 1;

            while (cursor < lines.length && lines[cursor]?.type === "remove") {
                const next = lines[cursor];
                if (next) removed.push(next);
                cursor += 1;
            }

            const added: DiffLine[] = [];
            while (cursor < lines.length && lines[cursor]?.type === "add") {
                const next = lines[cursor];
                if (next) added.push(next);
                cursor += 1;
            }

            if (removed.length > 0 && added.length > 0) {
                const pairs = Math.min(removed.length, added.length);
                for (let i = 0; i < pairs; i += 1) {
                    const removedLine = removed[i];
                    const addedLine = added[i];
                    if (!removedLine || !addedLine) continue;
                    removedLine.wordDiff = true;
                    addedLine.wordDiff = true;
                    removedLine.matchedLine = addedLine;
                    addedLine.matchedLine = removedLine;
                }
                result.push(...removed.filter(Boolean), ...added.filter(Boolean));
                index = cursor;
            } else {
                result.push(line);
                index += 1;
            }
            continue;
        }

        result.push(line);
        index += 1;
    }

    return result;
}

function computeWordDiff(left: string, right: string) {
    return diffWordsWithSpace(left, right, { ignoreCase: false });
}

function renderWordDiffLines(
    line: DiffLine,
    width: number,
    lineNumberWidth: number,
    dim: boolean | undefined,
    useThemeColors: boolean
): React.ReactNode[] | null {
    const { type, i, wordDiff, matchedLine, originalCode } = line;
    if (!wordDiff || !matchedLine) return null;

    const left = type === "remove" ? originalCode ?? "" : matchedLine.originalCode ?? "";
    const right = type === "remove" ? matchedLine.originalCode ?? "" : originalCode ?? "";
    const diff = computeWordDiff(left, right);
    const totalLength = left.length + right.length;

    const changedLength = diff
        .filter((part) => part.added || part.removed)
        .reduce((sum, part) => sum + part.value.length, 0);

    if (changedLength / totalLength > WORD_DIFF_THRESHOLD || dim) return null;

    const sign = type === "add" ? "+" : "-";
    const signWidth = sign.length;
    const contentWidth = Math.max(1, width - lineNumberWidth - 1 - signWidth);

    const wrappedLines: Array<{ content: React.ReactNode[]; contentWidth: number }> = [];
    let currentParts: React.ReactNode[] = [];
    let currentWidth = 0;

    diff.forEach((part, partIndex) => {
        let include = false;
        let backgroundColor: string | undefined;

        if (type === "add") {
            if (part.added) {
                include = true;
                backgroundColor = "diffAddedWord";
            } else if (!part.removed) {
                include = true;
            }
        } else if (type === "remove") {
            if (part.removed) {
                include = true;
                backgroundColor = "diffRemovedWord";
            } else if (!part.added) {
                include = true;
            }
        }

        if (!include) return;

        wrapByWidth(part.value, contentWidth).forEach((segment, segmentIndex) => {
            if (!segment) return;
            if (segmentIndex > 0 || currentWidth + segment.length > contentWidth) {
                if (currentParts.length > 0) {
                    wrappedLines.push({ content: [...currentParts], contentWidth: currentWidth });
                }
                currentParts = [];
                currentWidth = 0;
            }

            currentParts.push(
                <Text key={`part-${partIndex}-${segmentIndex}`} backgroundColor={backgroundColor}>
                    {segment}
                </Text>
            );
            currentWidth += segment.length;
        });
    });

    if (currentParts.length > 0) {
        wrappedLines.push({ content: currentParts, contentWidth: currentWidth });
    }

    return wrappedLines.map(({ content, contentWidth }, segmentIndex) => {
        const key = `${type}-${i}-${segmentIndex}`;
        const backgroundColor =
            type === "add"
                ? dim
                    ? "diffAddedDimmed"
                    : "diffAdded"
                : dim
                    ? "diffRemovedDimmed"
                    : "diffRemoved";
        const lineNumber = segmentIndex === 0 ? i : undefined;
        const prefix =
            (lineNumber !== undefined ? lineNumber.toString().padStart(lineNumberWidth) : " ".repeat(lineNumberWidth)) +
            " ";
        const usedWidth = prefix.length + signWidth + contentWidth;
        const padding = Math.max(0, width - usedWidth);

        return (
            <Text
                key={key}
                color={useThemeColors ? "text" : undefined}
                backgroundColor={backgroundColor}
                dimColor={dim}
            >
                {prefix}
                {sign}
                {content}
                {" ".repeat(padding)}
            </Text>
        );
    });
}

function applyLineNumbers(lines: DiffLine[], startLine: number): DiffLine[] {
    let lineNumber = startLine;
    const result: DiffLine[] = [];
    const queue = [...lines];

    while (queue.length > 0) {
        const line = queue.shift();
        if (!line) continue;

        const entry: DiffLine = { ...line, i: lineNumber };

        switch (line.type) {
            case "nochange":
                lineNumber += 1;
                result.push(entry);
                break;
            case "add":
                lineNumber += 1;
                result.push(entry);
                break;
            case "remove": {
                result.push(entry);
                let removedCount = 0;
                while (queue[0]?.type === "remove") {
                    lineNumber += 1;
                    const next = queue.shift();
                    if (!next) continue;
                    result.push({ ...next, i: lineNumber });
                    removedCount += 1;
                }
                lineNumber -= removedCount;
                break;
            }
        }
    }

    return result;
}

function buildDiffRows(
    lines: string[],
    startLine: number,
    width: number,
    dim: boolean | undefined,
    useThemeColors: boolean
): React.ReactNode[] {
    const clampedWidth = Math.max(1, Math.floor(width));
    const parsed = parseHunkLines(lines);
    const correlated = correlateDiffLines(parsed);
    const numbered = applyLineNumbers(correlated, startLine);
    const maxLineNumber = Math.max(...numbered.map(({ i }) => i), 0);
    const lineNumberWidth = Math.max(maxLineNumber.toString().length + 1, 0);

    return numbered.flatMap((line) => {
        const { type, code, i, wordDiff, matchedLine } = line;

        if (wordDiff && matchedLine) {
            const wordDiffRows = renderWordDiffLines(line, clampedWidth, lineNumberWidth, dim, useThemeColors);
            if (wordDiffRows !== null) return wordDiffRows;
        }

        const sign = type === "add" ? "+" : type === "remove" ? "-" : " ";
        const contentWidth = Math.max(1, clampedWidth - lineNumberWidth - 1 - 2);

        return wrapByWidth(code, contentWidth).map((segment, segmentIndex) => {
            const key = `${type}-${i}-${segmentIndex}`;
            const lineNumber = segmentIndex === 0 ? i : undefined;
            const prefix =
                (lineNumber !== undefined ? lineNumber.toString().padStart(lineNumberWidth) : " ".repeat(lineNumberWidth)) +
                " ";
            const usedWidth = prefix.length + 1 + segment.length;
            const padding = Math.max(0, clampedWidth - usedWidth);

            switch (type) {
                case "add":
                    return (
                        <Text
                            key={key}
                            color={useThemeColors ? "text" : undefined}
                            backgroundColor={dim ? "diffAddedDimmed" : "diffAdded"}
                            dimColor={dim}
                        >
                            {prefix}
                            {sign}
                            {segment}
                            {" ".repeat(padding)}
                        </Text>
                    );
                case "remove":
                    return (
                        <Text
                            key={key}
                            color={useThemeColors ? "text" : undefined}
                            backgroundColor={dim ? "diffRemovedDimmed" : "diffRemoved"}
                            dimColor={dim}
                        >
                            {prefix}
                            {sign}
                            {segment}
                            {" ".repeat(padding)}
                        </Text>
                    );
                case "nochange":
                    return (
                        <Text key={key} color={useThemeColors ? "text" : undefined} dimColor={dim}>
                            <Text dimColor>{prefix}</Text>
                            {sign}
                            {segment}
                            {" ".repeat(padding)}
                        </Text>
                    );
                default:
                    return null;
            }
        });
    }) as any[];
}

export function DiffView({ patch, dim, width }: DiffViewProps) {
    const [theme] = useTheme();
    const useThemeColors = Boolean(theme);
    const rows = useMemo(
        () => buildDiffRows(patch.lines, patch.oldStart, width, dim, useThemeColors),
        [patch.lines, patch.oldStart, width, dim, useThemeColors]
    );

    return (
        <ThemedBox flexDirection="column" flexGrow={1}>
            {rows.map((row, index) => (
                <ThemedBox key={index}>{row}</ThemedBox>
            ))}
        </ThemedBox>
    );
}
