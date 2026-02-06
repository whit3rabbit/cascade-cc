/**
 * File: src/tools/SendMessageTool.ts
 * Role: Allows agents to communicate with each other via mailbox using the deobfuscated schema.
 */

import { z } from 'zod';
import { writeToMailbox } from '../services/teams/TeammateMailbox.js';
import { getAgentId } from '../utils/shared/runtimeAndEnv.js';
import { SendMessagePayload } from '../types/AgentTypes.js';

// Input schema aligned with deobfuscated chunk 2424, 2425
const SendMessageInputSchema = z.object({
    contentType: z.enum(["message", "broadcast", "request", "response"]).describe('Message type: "message" for DMs, "broadcast" to all teammates, "request" for protocol requests (shutdown, plan approval), "response" for protocol responses'),
    recipient: z.string().optional().describe("Agent name of the recipient (required for message and request types)"),
    messageContent: z.string().optional().describe("Message text, reason, or feedback"),
    subtype: z.enum(["shutdown", "plan_approval"]).optional().describe("Protocol subtype (required for request and response types)"),
    requestId: z.string().optional().describe("Request ID to respond to (required for response type)"),
    approve: z.boolean().optional().describe("Whether to approve the request (required for response type)")
});

export const SendMessageTool = {
    name: "SendMessage",
    maxResultSizeChars: 100000,
    userFacingName() {
        return "SendMessage";
    },
    input_schema: {
        type: "object",
        properties: {
            contentType: {
                type: "string",
                enum: ["message", "broadcast", "request", "response"],
                description: 'Message type: "message" for DMs, "broadcast" to all teammates, "request" for protocol requests (shutdown, plan approval), "response" for protocol responses'
            },
            recipient: {
                type: "string",
                description: "Agent name of the recipient (required for message and request types)"
            },
            messageContent: {
                type: "string",
                description: "Message text, reason, or feedback"
            },
            subtype: {
                type: "string",
                enum: ["shutdown", "plan_approval"],
                description: "Protocol subtype (required for request and response types)"
            },
            requestId: {
                type: "string",
                description: "Request ID to respond to (required for response type)"
            },
            approve: {
                type: "boolean",
                description: "Whether to approve the request (required for response type)"
            }
        },
        required: ["contentType"]
    },

    async validateInput(input: any) {
        const result = SendMessageInputSchema.safeParse(input);
        if (!result.success) {
            return {
                result: false,
                message: result.error.issues.map(e => `${e.path.join('.')}: ${e.message}`).join(', ')
            };
        }

        const T = result.data;
        if (T.contentType === "message") {
            if (!T.recipient || T.recipient.trim().length === 0) {
                return { result: false, message: "recipient is required for message type", errorCode: 1 };
            }
            if (!T.messageContent || T.messageContent.trim().length === 0) {
                return { result: false, message: "content is required for message type", errorCode: 2 };
            }
        }
        if (T.contentType === "broadcast") {
            if (!T.messageContent || T.messageContent.trim().length === 0) {
                return { result: false, message: "content is required for broadcast type", errorCode: 3 };
            }
        }
        if (T.contentType === "request") {
            if (!T.subtype) {
                return { result: false, message: "subtype is required for request type", errorCode: 4 };
            }
            if (!T.recipient || T.recipient.trim().length === 0) {
                return { result: false, message: "recipient is required for request type", errorCode: 5 };
            }
        }
        if (T.contentType === "response") {
            if (!T.subtype) {
                return { result: false, message: "subtype is required for response type", errorCode: 6 };
            }
            if (!T.requestId || T.requestId.trim().length === 0) {
                return { result: false, message: "request_id is required for response type", errorCode: 7 };
            }
            if (T.approve === undefined) {
                return { result: false, message: "approve is required for response type", errorCode: 8 };
            }
            if (T.subtype === "shutdown" && !T.approve && (!T.messageContent || T.messageContent.trim().length === 0)) {
                return { result: false, message: "content (reason) is required when rejecting a shutdown request", errorCode: 9 };
            }
            if (T.subtype === "plan_approval" && (!T.recipient || T.recipient.trim().length === 0)) {
                return { result: false, message: "recipient is required for plan approval/rejection responses", errorCode: 10 };
            }
        }
        return { result: true };
    },

    async call(T: SendMessagePayload) {
        const validation = await this.validateInput(T);
        if (!validation.result) {
            return { is_error: true, content: validation.message };
        }

        const sender = getAgentId() || "unknown";

        // Recipient Validation
        // 2.1.19 Alignment: TeammateTool.write does not validate recipient existence (fire and forget/dynamic), so we skip findAgent check.
        if (T.recipient && T.recipient !== "User") {
            // No-op validation for dynamic teammates
        }

        const msgObject = {
            from: sender,
            text: T.messageContent || "",
            timestamp: new Date().toISOString(),
            subtype: T.subtype,
            contentType: T.contentType,
            requestId: T.requestId,
            approve: T.approve
        };

        try {
            const target = T.contentType === "broadcast" ? "ALL" : T.recipient || "User";
            writeToMailbox(target, msgObject as any);
            return {
                content: `Message (${T.contentType}) sent to ${target}.`
            };
        } catch (error: any) {
            return {
                is_error: true,
                content: `Failed to send message: ${error.message}`
            };
        }
    }
};
