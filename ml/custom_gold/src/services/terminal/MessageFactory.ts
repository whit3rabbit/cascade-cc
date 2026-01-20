
import { randomUUID } from "node:crypto";

export function createUserMessage(content: any, options: any = {}) {
    return {
        type: "user",
        message: {
            role: "user",
            content: content || " "
        },
        isMeta: options.isMeta || false,
        isVisibleInTranscriptOnly: options.isVisibleInTranscriptOnly || false,
        isCompactSummary: options.isCompactSummary || false,
        uuid: options.uuid || randomUUID(),
        timestamp: options.timestamp || new Date().toISOString(),
        toolUseResult: options.toolUseResult,
        thinkingMetadata: options.thinkingMetadata,
        todos: options.todos,
        imagePasteIds: options.imagePasteIds
    };
}

export function createAssistantMessage(content: any, options: any = {}) {
    return {
        type: "assistant",
        uuid: options.uuid || randomUUID(),
        timestamp: options.timestamp || new Date().toISOString(),
        message: {
            id: randomUUID(),
            container: null,
            model: options.model || "claude-3-5-sonnet",
            role: "assistant",
            stop_reason: "stop_sequence",
            stop_sequence: "",
            type: "message",
            usage: options.usage || {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0
            },
            content: typeof content === "string" ? [{ type: "text", text: content }] : content,
            context_management: null
        },
        requestId: options.requestId,
        error: options.error,
        isApiErrorMessage: options.isApiErrorMessage || false
    };
}

export function createMetadataMessage(options: any) {
    if (options.content && typeof options.content === "object" && !Array.isArray(options.content)) {
        // Handle direct content block if passed
    }
    return createUserMessage(options.content, { ...options, isMeta: true });
}

export function createBannerMessage(text: string, type: "info" | "warning" | "error" | "suggestion" = "info") {
    return createUserMessage(text, { isMeta: true, bannerType: type });
}

export function createAttachment(attachment: any) {
    return {
        attachment,
        type: "attachment" as const,
        uuid: randomUUID(),
        timestamp: new Date().toISOString()
    };
}

export const createAttachmentMessage = createAttachment;

// GX
export function normalizeMessages(messages: any[]): any[] {
    let hasComplexMessages = false;
    return messages.flatMap((msg) => {
        switch (msg.type) {
            case "assistant":
                hasComplexMessages = hasComplexMessages || msg.message.content.length > 1;
                return msg.message.content.map((block: any) => {
                    return {
                        ...msg,
                        uuid: hasComplexMessages ? randomUUID() : msg.uuid,
                        message: {
                            ...msg.message,
                            content: [block],
                            context_management: msg.message.context_management ?? null
                        }
                    };
                });
            case "user":
                if (typeof msg.message.content === "string") {
                    return [{
                        ...msg,
                        uuid: hasComplexMessages ? randomUUID() : msg.uuid,
                        message: {
                            ...msg.message,
                            content: [{ type: "text", text: msg.message.content }]
                        }
                    }];
                }
                hasComplexMessages = hasComplexMessages || msg.message.content.length > 1;
                let imageCounter = 0;
                return msg.message.content.map((block: any) => {
                    const isImage = block.type === "image";
                    const imageId = isImage && msg.imagePasteIds ? msg.imagePasteIds[imageCounter++] : undefined;
                    return {
                        ...msg,
                        uuid: hasComplexMessages ? randomUUID() : msg.uuid,
                        message: {
                            ...msg.message,
                            content: [block]
                        },
                        imagePasteIds: imageId !== undefined ? [imageId] : undefined
                    };
                });
            default:
                return [msg];
        }
    });
}

// Q9 Logic
export function extractTagContent(text: string, tagName: string): string | null {
    if (!text.trim() || !tagName.trim()) return null;
    const escapedTag = tagName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const regex = new RegExp(`<${escapedTag}(?:\\s+[^>]*)?>([\\s\\S]*?)<\\/${escapedTag}>`, "gi");
    let match;
    let lastIndex = 0;
    while ((match = regex.exec(text)) !== null) {
        const content = match[1];
        const prefix = text.slice(lastIndex, match.index);

        // Count unclosed tags to handle nesting
        const openMatches = prefix.match(new RegExp(`<${escapedTag}(?:\\s+[^>]*?)?>`, "gi"))?.length || 0;
        const closeMatches = prefix.match(new RegExp(`<\\/${escapedTag}>`, "gi"))?.length || 0;

        if (openMatches === closeMatches && content) return content.trim();
        lastIndex = match.index + match[0].length;
    }
    return null;
}

export function createToolResultMessage(tool: any, output: any, toolUseID: string) {
    let content: any[] = [];
    if (typeof output === "string") {
        content.push({ type: "text", text: output });
    } else if (Array.isArray(output)) {
        content = output;
    } else {
        content.push({ type: "text", text: JSON.stringify(output) });
    }

    return {
        type: "tool_result",
        tool_use_id: toolUseID,
        content: content
    };
}
