
import { diffLines } from "diff";
import fs from "node:fs";

/**
 * Normalizes quotes and other special characters.
 * Corresponds to cY2 in chunk_387.ts.
 */
export function normalizeQuotes(str: string): string {
    return str
        .replaceAll("‘", "'")
        .replaceAll("’", "'")
        .replaceAll("“", '"')
        .replaceAll("”", '"');
}

/**
 * Normalizes trailing whitespace on each line.
 * Corresponds to VJ0 in chunk_387.ts.
 */
export function normalizeWhitespace(str: string): string {
    const lines = str.split(/(\r\n|\n|\r)/);
    let result = "";
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line !== undefined) {
            // Even lines are content, odd are delimiters
            if (i % 2 === 0) {
                result += line.replace(/\s+$/, "");
            } else {
                result += line;
            }
        }
    }
    return result;
}

const TOKEN_REPLACEMENTS: Record<string, string> = {
    "<fnr>": "<function_results>",
    "<n>": "<name>",
    "</n>": "</name>",
    "<o>": "<output>",
    "</o>": "</output>",
    "<e>": "<error>",
    "</e>": "</error>",
    "<s>": "<system>",
    "</s>": "</system>",
    "<r>": "<result>",
    "</r>": "</result>",
    "< META_START >": "<META_START>",
    "< META_END >": "<META_END>",
    "< EOT >": "<EOT>",
    "< META >": "<META>",
    "< SOS >": "<SOS>",
    "\n\nH:": "\n\nHuman:",
    "\n\nA:": "\n\nAssistant:"
};

/**
 * Normalizes special tokens used by the model.
 * Corresponds to SI5 in chunk_387.ts.
 */
export function normalizeTokens(str: string): { result: string, appliedReplacements: { from: string, to: string }[] } {
    let result = str;
    const appliedReplacements: { from: string, to: string }[] = [];

    for (const [from, to] of Object.entries(TOKEN_REPLACEMENTS)) {
        const original = result;
        result = result.split(from).join(to);
        if (result !== original) {
            appliedReplacements.push({ from, to });
        }
    }

    return { result, appliedReplacements };
}

/**
 * Core replacement logic.
 * Corresponds to lY2 in chunk_387.ts.
 */
export function applyReplace(content: string, oldString: string, newString: string, replaceAll: boolean = false): string {
    if (newString !== "") {
        if (replaceAll) {
            return content.split(oldString).join(newString);
        }
        return content.replace(oldString, newString);
    }

    // Special case for deletion: if oldString doesn't end with newline but is followed by one, include it.
    if (!oldString.endsWith("\n") && content.includes(oldString + "\n")) {
        const search = oldString + "\n";
        if (replaceAll) {
            return content.split(search).join(newString);
        }
        return content.replace(search, newString);
    }

    if (replaceAll) {
        return content.split(oldString).join(newString);
    }
    return content.replace(oldString, newString);
}

/**
 * Processes edits to handle normalization and quote fuzzy matching.
 * Corresponds to aY2 in chunk_387.ts.
 */
export function processEdits(
    fileContent: string,
    edits: { old_string: string, new_string: string, replace_all?: boolean }[]
): { old_string: string, new_string: string, replace_all: boolean }[] {
    return edits.map(edit => {
        const { old_string, new_string, replace_all = false } = edit;
        const normalizedNew = normalizeWhitespace(new_string);

        // 1. Direct match
        if (fileContent.includes(old_string)) {
            return { old_string, new_string: normalizedNew, replace_all };
        }

        // 2. Token normalization match
        const { result: normOld, appliedReplacements } = normalizeTokens(old_string);
        if (fileContent.includes(normOld)) {
            let normNew = normalizedNew;
            for (const { from, to } of appliedReplacements) {
                normNew = normNew.split(from).join(to);
            }
            return { old_string: normOld, new_string: normNew, replace_all };
        }

        // 3. Quote normalization fuzzy match
        const contentNorm = normalizeQuotes(fileContent);
        const oldNorm = normalizeQuotes(old_string);
        const index = contentNorm.indexOf(oldNorm);
        if (index !== -1) {
            const actualOld = fileContent.substring(index, index + old_string.length);
            return { old_string: actualOld, new_string: normalizedNew, replace_all };
        }

        return { old_string, new_string: normalizedNew, replace_all };
    });
}

/**
 * Main entry point for applying a set of edits to a file.
 * Corresponds to VSA in chunk_387.ts.
 */
export function applyEdits(
    filePath: string,
    fileContents: string | undefined,
    edits: { old_string: string, new_string: string, replace_all?: boolean }[]
): { patch: any, updatedFile: string } {
    let currentContent = fileContents || "";
    const appliedNewStrings: string[] = [];

    // Special case: creation of a file
    if (!fileContents && edits.length === 1 && edits[0].old_string === "" && edits[0].new_string === "") {
        return {
            patch: generateHunks({
                filePath,
                oldContent: "",
                newContent: ""
            }),
            updatedFile: ""
        };
    }

    const processedEdits = processEdits(currentContent, edits);

    for (const edit of processedEdits) {
        const trimmedOld = edit.old_string.replace(/\n+$/, "");
        for (const applied of appliedNewStrings) {
            if (trimmedOld !== "" && applied.includes(trimmedOld)) {
                throw new Error("Cannot edit file: old_string is a substring of a new_string from a previous edit.");
            }
        }

        const original = currentContent;
        if (edit.old_string === "") {
            currentContent = edit.new_string;
        } else {
            currentContent = applyReplace(currentContent, edit.old_string, edit.new_string, edit.replace_all);
        }

        if (currentContent === original) {
            throw new Error(`String not found in file: ${edit.old_string}`);
        }

        appliedNewStrings.push(edit.new_string);
    }

    if (currentContent === fileContents) {
        throw new Error("Original and edited file match exactly. Failed to apply edit.");
    }

    return {
        patch: generateHunks({
            filePath,
            oldContent: fileContents || "",
            newContent: currentContent
        }),
        updatedFile: currentContent
    };
}

const AMPERSAND_TOKEN = "<<:AMPERSAND_TOKEN:>>";
const DOLLAR_TOKEN = "<<:DOLLAR_TOKEN:>>";

function escapeTokens(str: string): string {
    return str.replaceAll("&", AMPERSAND_TOKEN).replaceAll("$", DOLLAR_TOKEN);
}

function unescapeTokens(str: string): string {
    return str.replaceAll(AMPERSAND_TOKEN, "&").replaceAll(DOLLAR_TOKEN, "$");
}

export interface Hunk {
    oldStart: number;
    oldLines: number;
    newStart: number;
    newLines: number;
    lines: string[];
}

/**
 * Generates unified diff hunks.
 * Deeply deobfuscated from D_A in chunk_291.ts.
 */
export function generateHunks({
    filePath,
    oldContent,
    newContent,
    ignoreWhitespace = false,
    singleHunk = false,
    context = 3
}: {
    filePath: string,
    oldContent: string,
    newContent: string,
    ignoreWhitespace?: boolean,
    singleHunk?: boolean,
    context?: number
}) {
    const encodedOld = escapeTokens(oldContent);
    const encodedNew = escapeTokens(newContent);

    const diff = diffLines(encodedOld, encodedNew, { ignoreWhitespace });
    const hunks: Hunk[] = [];

    let oldStart = 1;
    let newStart = 1;
    let currentHunk: Hunk | null = null;
    const contextLines = singleHunk ? 1000000 : context;

    for (let i = 0; i < diff.length; i++) {
        const part = diff[i];
        const lines = part.value.split(/\r?\n/);
        if (lines[lines.length - 1] === "") lines.pop();

        const count = lines.length;

        if (part.added || part.removed) {
            if (!currentHunk) {
                // Determine context from previous part
                const prevPart = diff[i - 1];
                let contextPreLines: string[] = [];
                if (prevPart && !prevPart.added && !prevPart.removed) {
                    const prevLines = prevPart.value.split(/\r?\n/);
                    if (prevLines[prevLines.length - 1] === "") prevLines.pop();
                    contextPreLines = prevLines.slice(-contextLines).map(l => " " + unescapeTokens(l));
                }

                currentHunk = {
                    oldStart: oldStart - contextPreLines.length,
                    oldLines: contextPreLines.length,
                    newStart: newStart - contextPreLines.length,
                    newLines: contextPreLines.length,
                    lines: contextPreLines
                };
            }

            if (part.added) {
                currentHunk.newLines += count;
                currentHunk.lines.push(...lines.map(l => "+" + unescapeTokens(l)));
                newStart += count;
            } else {
                currentHunk.oldLines += count;
                currentHunk.lines.push(...lines.map(l => "-" + unescapeTokens(l)));
                oldStart += count;
            }
        } else {
            if (currentHunk) {
                const postLines = lines.slice(0, contextLines).map(l => " " + unescapeTokens(l));
                currentHunk.oldLines += postLines.length;
                currentHunk.newLines += postLines.length;
                currentHunk.lines.push(...postLines);

                // If next part is close enough, merger (but we keep it simple for now)
                hunks.push(currentHunk);
                currentHunk = null;
            }
            oldStart += count;
            newStart += count;
        }
    }

    if (currentHunk) hunks.push(currentHunk);

    return {
        filePath,
        oldContent,
        newContent,
        hunks
    };
}

/**
 * Converts hunks back to granular edits.
 * Corresponds to nY2 in chunk_387.ts.
 */
export function hunksToEdits(hunks: Hunk[]): { old_string: string, new_string: string, replace_all: boolean }[] {
    return hunks.map(hunk => {
        const oldLines: string[] = [];
        const newLines: string[] = [];

        for (const line of hunk.lines) {
            if (line.startsWith(" ")) {
                oldLines.push(line.slice(1));
                newLines.push(line.slice(1));
            } else if (line.startsWith("-")) {
                oldLines.push(line.slice(1));
            } else if (line.startsWith("+")) {
                newLines.push(line.slice(1));
            }
        }

        return {
            old_string: oldLines.join("\n"),
            new_string: newLines.join("\n"),
            replace_all: false
        };
    });
}
