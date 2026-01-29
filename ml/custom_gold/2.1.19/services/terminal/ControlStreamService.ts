/**
 * File: src/services/terminal/ControlStreamService.ts
 * Role: UTILITY_HELPER
 * Aggregated from 1 chunks
 */

import { randomUUID } from "node:crypto";
import { z } from 'zod';
import {
    checkToolPermissions,
    formatDecisionReason,
    handlePermissionResponse,
    $_ as checkToolPermissionsAlias,
    Rn2 as formatDecisionReasonAlias,
    kdA as handlePermissionResponseAlias
} from "./PermissionService.js";
import {
    ToolPermissionResponseSchema as uv1,
    HookCallbackSchema as PV1
} from "./schemas.js";

interface PendingRequest {
    request: any;
    resolve: (value: any) => void;
    reject: (reason?: any) => void;
    schema?: z.ZodTypeAny;
}

export interface ControlStreamServiceOptions {
    replayUserMessages?: boolean;
}

/**
 * Service for managing control stream communication, including tool permissions
 * and MCP (Model Context Protocol) interactions.
 */
export class ControlStreamService {
    private input: AsyncIterable<string>;
    private output: (line: string) => void;
    private replayUserMessages: boolean;
    private pendingRequests: Map<string, PendingRequest>;
    public structuredInput: AsyncGenerator<any, void, unknown>;
    private inputClosed: boolean;
    private unexpectedResponseCallback?: (response: any) => Promise<void> | void;

    constructor(input: AsyncIterable<string>, output: (line: string) => void, replayUserMessages = false) {
        this.input = input;
        this.output = output;
        this.replayUserMessages = replayUserMessages;
        this.pendingRequests = new Map();
        this.structuredInput = this.read();
        this.inputClosed = false;
    }

    /**
     * Reads and processes lines from the input stream.
     * Yields parsed messages.
     */
    async *read() {
        let buffer = "";

        for await (const chunk of this.input) {
            buffer += chunk;
            let newLineIndex;

            // Process complete lines from the buffer
            while ((newLineIndex = buffer.indexOf("\n")) !== -1) {
                const line = buffer.slice(0, newLineIndex);
                buffer = buffer.slice(newLineIndex + 1); // Remove processed line from buffer

                const parsedMessage = await this.processLine(line);
                if (parsedMessage) {
                    yield parsedMessage;
                }
            }
        }

        // Process any remaining data in the buffer
        if (buffer) {
            const parsedMessage = await this.processLine(buffer);
            if (parsedMessage) {
                yield parsedMessage;
            }
        }

        this.inputClosed = true;

        // Reject any pending requests when the input stream closes
        for (const request of this.pendingRequests.values()) {
            request.reject(new Error("Control stream closed before response received"));
        }
    }

    /**
     * Gets pending permission requests.
     */
    getPendingPermissionRequests(): any[] {
        return Array.from(this.pendingRequests.values())
            .map(requestData => requestData.request)
            .filter(request => request.request.subtype === "can_use_tool");
    }

    /**
     * Sets a callback for handling unexpected responses.
     */
    setUnexpectedResponseCallback(callback: (response: any) => Promise<void> | void): void {
        this.unexpectedResponseCallback = callback;
    }

    /**
     * Parses and processes a single line from the control stream.
     */
    async processLine(line: string): Promise<any> {
        try {
            const message = this.safeParseJSON(line);

            if (!message) {
                return;
            }

            switch (message.type) {
                case "keep_alive":
                    return;
                case "update_environment_variables":
                    for (const [key, value] of Object.entries(message.variables as Record<string, string>)) {
                        process.env[key] = value;
                    }
                    return;
                case "control_response":
                    const pendingRequest = this.pendingRequests.get(message.response.request_id);

                    if (!pendingRequest) {
                        if (this.unexpectedResponseCallback) {
                            await this.unexpectedResponseCallback(message);
                        }
                        return;
                    }

                    this.pendingRequests.delete(message.response.request_id);

                    if (message.response.subtype === "error") {
                        pendingRequest.reject(new Error(message.response.error));
                        return;
                    }

                    if (pendingRequest.schema) {
                        try {
                            pendingRequest.resolve(pendingRequest.schema.parse(message.response.response));
                        } catch (error) {
                            pendingRequest.reject(error);
                        }
                    }
                    else {
                        pendingRequest.resolve({});
                    }

                    if (!this.replayUserMessages) {
                        return;
                    }

                    return message;

                case "control_request":
                    if (!message.request) {
                        console.error("Error: Missing request on control_request");
                    }
                    return message; // Handle control requests
                case "user":
                    if (message.message.role !== "user") {
                        console.error(`Error: Expected message role 'user', got '${message.message.role}'`);
                    }
                    return message;
                default:
                    console.error(`Error: Expected message type 'user' or 'control', got '${message.type}'`);
            }
        } catch (error) {
            console.error(`Error parsing streaming input line: ${line}:`, error);
            // process.exit(1); 
        }
    }

    /**
     * Safely parses a JSON string, returning null on failure.
     */
    safeParseJSON(jsonString: string): any | null {
        try {
            return JSON.parse(jsonString);
        } catch (error) {
            return null;
        }
    }

    /**
     * Writes a message to the output stream.
     */
    async write(message: any): Promise<void> {
        this.output(JSON.stringify(message) + "\n");
    }

    /**
     * Sends a control request and waits for a response.
     */
    async sendRequest(request: any, schema?: z.ZodTypeAny, abortSignal?: AbortSignal): Promise<any> {
        const requestId = randomUUID();
        const message = {
            type: "control_request",
            request_id: requestId,
            request: request,
        };

        if (this.inputClosed) {
            throw new Error("Stream closed");
        }

        if (abortSignal?.aborted) {
            throw new Error("Request aborted");
        }

        await this.write(message);

        const cleanup = () => {
            this.write({
                type: "control_cancel_request",
                request_id: requestId,
            });
            const pendingRequest = this.pendingRequests.get(requestId);
            if (pendingRequest) {
                pendingRequest.reject(new Error("Request cancelled"));
            }
        };

        if (abortSignal) {
            abortSignal.addEventListener("abort", cleanup, { once: true });
        }

        try {
            return await new Promise((resolve, reject) => {
                this.pendingRequests.set(requestId, {
                    request: message,
                    resolve,
                    reject,
                    schema,
                });
            });
        } finally {
            if (abortSignal) {
                abortSignal.removeEventListener("abort", cleanup);
            }
            this.pendingRequests.delete(requestId);
        }
    }

    /**
     * Creates a function to handle 'can_use_tool' requests.
     */
    createCanUseTool(preCheck?: () => void) {
        return async (toolName: string, input: any, context: any, ...rest: any[]) => {
            const { agentId, abortController } = context || {};

            const permissionResult = await checkToolPermissionsAlias(toolName, input, context, ...rest);
            if (permissionResult.behavior === "allow" || permissionResult.behavior === "deny") {
                return permissionResult;
            }

            try {
                preCheck?.();
                const response = await this.sendRequest(
                    {
                        subtype: "can_use_tool",
                        tool_name: toolName,
                        input: input,
                        permission_suggestions: permissionResult.suggestions,
                        blocked_path: permissionResult.blockedPath,
                        decision_reason: formatDecisionReasonAlias((permissionResult as any).decisionReason) || "Tool use request",
                        toolUseId: rest.find(item => typeof item === 'string'),
                        agent_id: agentId,
                    },
                    uv1,
                    abortController?.signal
                );
                return handlePermissionResponseAlias(response, toolName, input, context);
            } catch (error) {
                return handlePermissionResponseAlias(
                    {
                        behavior: "deny",
                        message: `Tool permission request failed: ${error}`,
                        toolUseID: rest.find(item => typeof item === 'string'),
                    },
                    toolName,
                    input,
                    context
                );
            }
        };
    }

    /**
     * Creates a hook callback function for the control stream.
     */
    createHookCallback(callbackId: string, timeout: number) {
        return {
            type: "callback",
            timeout: timeout,
            callback: async (input: any, toolUseId: any, abortSignal?: AbortSignal) => {
                try {
                    return await this.sendRequest(
                        {
                            subtype: "hook_callback",
                            callback_id: callbackId,
                            input: input,
                            toolUseId: toolUseId,
                        },
                        PV1,
                        abortSignal,
                    );
                } catch (error) {
                    console.error(`Error in hook callback ${callbackId}:`, error);
                    return {}; // or appropriate error handling
                }
            },
        };
    }

    /**
     * Sends an MCP (Model Context Protocol) message.
     */
    async sendMcpMessage(serverName: string, message: any): Promise<any> {
        const response = await this.sendRequest(
            {
                subtype: "mcp_message",
                server_name: serverName,
                message: message,
            },
            z.object({ mcp_response: z.any() }),
        );
        return response.mcp_response;
    }
}
