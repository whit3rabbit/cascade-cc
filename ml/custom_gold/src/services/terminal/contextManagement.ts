/**
 * Context management logic.
 * Deobfuscated from hzB in chunk_226.ts.
 */

const MAX_INPUT_TOKENS = 180000;
const TARGET_INPUT_TOKENS = 40000;

export interface ContextManagementOptions {
    hasThinking?: boolean;
}

export function getContextManagementEdits(options: ContextManagementOptions = {}): any {
    const preserveThinking = process.env.PRESERVE_THINKING === "enabled";
    const useClearResults = process.env.USE_API_CLEAR_TOOL_RESULTS === "true";
    const useClearUses = process.env.USE_API_CLEAR_TOOL_USES === "true";

    if (!useClearResults && !useClearUses && (!preserveThinking || !options.hasThinking)) {
        return undefined;
    }

    const edits: any[] = [];

    if (useClearResults) {
        const max = process.env.API_MAX_INPUT_TOKENS ? parseInt(process.env.API_MAX_INPUT_TOKENS) : MAX_INPUT_TOKENS;
        const target = process.env.API_TARGET_INPUT_TOKENS ? parseInt(process.env.API_TARGET_INPUT_TOKENS) : TARGET_INPUT_TOKENS;

        edits.push({
            type: "clear_tool_uses_20250919",
            trigger: {
                type: "input_tokens",
                value: max
            },
            clear_at_least: {
                type: "input_tokens",
                value: max - target
            },
            clear_tool_inputs: ["Shell", "Glob", "Grep", "Read", "WebFetch", "WebSearch"]
        });
    }

    if (useClearUses) {
        const max = process.env.API_MAX_INPUT_TOKENS ? parseInt(process.env.API_MAX_INPUT_TOKENS) : MAX_INPUT_TOKENS;
        const target = process.env.API_TARGET_INPUT_TOKENS ? parseInt(process.env.API_TARGET_INPUT_TOKENS) : TARGET_INPUT_TOKENS;

        edits.push({
            type: "clear_tool_uses_20250919",
            trigger: {
                type: "input_tokens",
                value: max
            },
            clear_at_least: {
                type: "input_tokens",
                value: max - target
            },
            exclude_tools: ["Edit", "Write", "NotebookEdit"]
        });
    }

    if (preserveThinking && options.hasThinking) {
        edits.push({
            type: "clear_thinking_20251015",
            keep: "all"
        });
    }

    return edits.length > 0 ? { edits } : undefined;
}
