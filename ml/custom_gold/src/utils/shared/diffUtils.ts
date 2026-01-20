import * as diff from 'diff';

const AMPERSAND_TOKEN = "<<:AMPERSAND_TOKEN:>>";
const DOLLAR_TOKEN = "<<:DOLLAR_TOKEN:>>";
const CONTEXT_LINES = 3;

export function expandTabs(text: string): string {
    return text.replace(/^\t+/gm, (match) => "  ".repeat(match.length));
}

export function escapeDiffTokens(text: string): string {
    return text.replaceAll("&", AMPERSAND_TOKEN).replaceAll("$", DOLLAR_TOKEN);
}

export function restoreDiffTokens(text: string): string {
    return text.replaceAll(AMPERSAND_TOKEN, "&").replaceAll(DOLLAR_TOKEN, "$");
}

export function diffStats(hunks: any[], oldContent?: string): { added: number; removed: number } {
    let added = 0;
    let removed = 0;

    if (hunks.length === 0 && oldContent) {
        // Fallback or specific logic from original code
        added = oldContent.split(/\r?\n/).length;
    } else {
        // This is a simplified version of the logic in z_A
        // Ideally we would inspect the hunks structure from the diff library
        // The original code passed 'A' which seemed to be an array of hunks
        for (const hunk of hunks) {
            added += hunk.lines.filter((l: string) => l.startsWith("+")).length;
            removed += hunk.lines.filter((l: string) => l.startsWith("-")).length;
        }
    }

    // Telemetry placeholders
    // mF1(added, removed); 
    // n("tengu_file_changed", { lines_added: added, lines_removed: removed });

    return { added, removed };
}

export function createPatch(
    filePath: string,
    oldContent: string,
    newContent: string,
    ignoreWhitespace: boolean = false,
    singleHunk: boolean = false
): any[] { // Returns array of hunks with restored tokens
    const escapedOld = escapeDiffTokens(oldContent);
    const escapedNew = escapeDiffTokens(newContent);

    // createUnifiedDiff returns a string usually, but the original code (D_A) returned an object with 'hunks'
    // We need to adapt this to use the 'diff' library's struturedPatch if we want hunks, 
    // or parse the unified diff string.
    // The original code used a custom diff implementation that returned structured data.
    // Let's use diff.structuredPatch from the 'diff' package.

    const patch = diff.structuredPatch(
        filePath,
        filePath,
        escapedOld,
        escapedNew,
        undefined,
        undefined,
        {
            context: singleHunk ? 100000 : CONTEXT_LINES,
            ignoreWhitespace: ignoreWhitespace
        }
    );

    return patch.hunks.map(hunk => ({
        ...hunk,
        lines: hunk.lines.map(restoreDiffTokens)
    }));
}

export function applyEdits(
    filePath: string,
    fileContents: string,
    edits: Array<{ old_string: string; new_string: string; replace_all?: boolean }>,
    ignoreWhitespace: boolean = false
): any[] {
    const escapedContent = escapeDiffTokens(expandTabs(fileContents));

    const newContent = edits.reduce((currentContent, edit) => {
        const { old_string, new_string, replace_all } = edit;
        const escapedOld = escapeDiffTokens(expandTabs(old_string));
        const escapedNew = escapeDiffTokens(expandTabs(new_string));

        if (replace_all) {
            return currentContent.replaceAll(escapedOld, () => escapedNew);
        } else {
            return currentContent.replace(escapedOld, () => escapedNew);
        }
    }, escapedContent);

    // Verify changes were made? Original code just generates diff.

    const patch = diff.structuredPatch(
        filePath,
        filePath,
        escapedContent,
        newContent,
        undefined,
        undefined,
        {
            context: CONTEXT_LINES,
            ignoreWhitespace: ignoreWhitespace
        }
    );

    return patch.hunks.map(hunk => ({
        ...hunk,
        lines: hunk.lines.map(restoreDiffTokens)
    }));
}
export function createUnifiedDiff(
    oldFileName: string,
    newFileName: string,
    oldStr: string,
    newStr: string,
    oldHeader: string,
    newHeader: string,
    options: any
): any {
    // Checks for circular dependencies or specific handling for patch generation
    if (typeof options === 'function') {
        options = { callback: options };
    }

    // Using the 'diff' package's structured patch or unified diff creation
    return diff.createTwoFilesPatch(
        oldFileName,
        newFileName,
        oldStr,
        newStr,
        oldHeader || '',
        newHeader || '',
        options
    );
}

// Re-export other diff functions if needed, or implement helpers
// chunk_290.ts seems to construct a patch manually or wraps it
