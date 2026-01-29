/**
 * File: src/utils/shared/messageUtils.ts
 * Role: Utilities for extracting, transforming, and cleaning up conversation messages and tool outputs.
 */

// --- Types ---

export interface MessageContent {
    type: string;
    text?: string;
    [key: string]: any;
}

export interface Message {
    role: 'user' | 'assistant' | 'system';
    content: string | MessageContent[];
    [key: string]: any;
}

export interface AttachmentBlock {
    type: 'attachment';
    attachment: {
        type: string;
        [key: string]: any;
    };
}

/**
 * Extracts the primary text content from a list of conversation messages.
 * Typically used to find the latest user prompt.
 * 
 * @param messages - The list of messages.
 * @returns {string} The extracted text.
 */
export function extractMessageText(messages: Message[]): string {
    const userMsg = messages.find((m) => m.role === "user");
    if (!userMsg) return "";

    const content = userMsg.content;
    if (typeof content === "string") {
        return content;
    }

    if (Array.isArray(content)) {
        const textBlock = content.find((c) => c.type === "text");
        if (textBlock) {
            return textBlock.text || "";
        }
    }

    return "";
}

/**
 * Normalizes a tool result (string or object) into a string representation.
 */
export function normalizeToolResult(result: any): string {
    if (typeof result === "string") {
        return result;
    }
    try {
        return JSON.stringify(result);
    } catch {
        return String(result);
    }
}

/**
 * Simplifies and normalizes attachment types for storage or display.
 */
export function transformAttachment(block: any): any {
    if (block?.type !== "attachment") {
        return block;
    }
    const attachment = block.attachment;
    if (!attachment) return block;

    if (attachment.type === "new_file") {
        return {
            ...block,
            attachment: {
                ...attachment,
                type: "file"
            }
        };
    }
    if (attachment.type === "new_directory") {
        return {
            ...block,
            attachment: {
                ...attachment,
                type: "directory"
            }
        };
    }
    return block;
}

/**
 * Transforms a list of messages for cleanup or history management.
 */
export function transformMessagesForCleanup(messages: any[]): any[] {
    try {
        return messages.map(transformAttachment);
    } catch (error) {
        console.error("[MessageUtils] Error during message cleanup:", error);
        return messages;
    }
}

/**
 * Identifies and handles invoked skills found in message attachments.
 */
export function registerInvokedSkills(messages: any[]): void {
    for (const msg of messages) {
        if (msg.type !== "attachment" || msg.attachment?.type !== "invoked_skills") {
            continue;
        }
        const skills = msg.attachment.skills || [];
        for (const skill of skills) {
            if (skill.name && skill.path) {
                console.log(`[MessageUtils] Registered invoked skill: ${skill.name} (${skill.path})`);
            }
        }
    }
}
