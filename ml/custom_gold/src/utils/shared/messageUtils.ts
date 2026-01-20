/**
 * Message utilities.
 * Deobfuscated from q63 in chunk_226.ts.
 */

/**
 * Extracts the main text content from a list of messages (usually for the last user message).
 */
export function extractMessageText(messages: any[]): string {
    const userMsg = messages.find((m) => m.role === "user");
    if (!userMsg) return "";

    const content = userMsg.content;
    if (typeof content === "string") return content;

    if (Array.isArray(content)) {
        const textBlock = content.find((c) => c.type === "text");
        if (textBlock && textBlock.type === "text") {
            return textBlock.text;
        }
    }

    return "";
}

/**
 * Normalizes tool results for context management.
 */
export function normalizeToolResult(result: any): string {
    if (typeof result === "string") return result;
    try {
        return JSON.stringify(result);
    } catch {
        return String(result);
    }
}
