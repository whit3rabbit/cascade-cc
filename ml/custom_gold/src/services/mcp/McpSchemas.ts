import { z } from "zod";

export const LATEST_PROTOCOL_VERSION = "2025-11-25";
export const JSONRPC_VERSION = "2.0";

const RelatedTaskSchema = z.object({
    taskId: z.string()
});

const MetaSchema = z.object({
    progressToken: z.union([z.string(), z.number().int()]).optional(),
    "io.modelcontextprotocol/related-task": RelatedTaskSchema.optional()
}).passthrough().optional();

function McpRequestSchema<T extends string, P extends z.ZodTypeAny>(method: T, params: P) {
    return z.object({
        method: z.literal(method),
        params: params.optional()
    });
}

// --- Base Types ---

export const JsonRpcRequestSchema = z.object({
    jsonrpc: z.literal(JSONRPC_VERSION),
    id: z.union([z.string(), z.number().int()]),
    method: z.string(),
    params: z.object({
        _meta: MetaSchema,
        task: z.object({
            ttl: z.number().optional(),
            pollInterval: z.number().optional()
        }).optional()
    }).passthrough().optional()
}).strict();

export const JsonRpcNotificationSchema = z.object({
    jsonrpc: z.literal(JSONRPC_VERSION),
    method: z.string(),
    params: z.object({
        _meta: MetaSchema
    }).passthrough().optional()
}).strict();

export const JsonRpcResponseSchema = z.object({
    jsonrpc: z.literal(JSONRPC_VERSION),
    id: z.union([z.string(), z.number().int()]),
    result: z.object({
        _meta: MetaSchema
    }).passthrough()
}).strict();

export const JsonRpcErrorSchema = z.object({
    jsonrpc: z.literal(JSONRPC_VERSION),
    id: z.union([z.string(), z.number().int()]),
    error: z.object({
        code: z.number().int(),
        message: z.string(),
        data: z.unknown().optional()
    })
}).strict();

export const JsonRpcMessageSchema = z.union([
    JsonRpcRequestSchema,
    JsonRpcNotificationSchema,
    JsonRpcResponseSchema,
    JsonRpcErrorSchema
]);

export enum ErrorCode {
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,
    ConnectionClosed = -32000,
    RequestTimeout = -32001,
    UrlElicitationRequired = -32042
}

// --- Content Types ---

export const TextContentSchema = z.object({
    type: z.literal("text"),
    text: z.string(),
    annotations: z.object({
        audience: z.array(z.enum(["user", "assistant"])).optional(),
        priority: z.number().min(0).max(1).optional()
    }).optional(),
    _meta: z.record(z.string(), z.unknown()).optional()
});

export const ImageContentSchema = z.object({
    type: z.literal("image"),
    data: z.string().refine((val) => {
        try {
            return atob(val), true;
        } catch {
            return false;
        }
    }, "Invalid Base64 string"),
    mimeType: z.string(),
    annotations: z.object({}).passthrough().optional(),
    _meta: z.record(z.string(), z.unknown()).optional()
});

export const EmbeddedResourceSchema = z.object({
    type: z.literal("resource"),
    resource: z.object({
        uri: z.string(),
        mimeType: z.string().optional(),
        text: z.string().optional(),
        blob: z.string().optional(), // Base64
        _meta: z.object({}).passthrough().optional()
    }),
    annotations: z.object({}).passthrough().optional(),
    _meta: z.record(z.string(), z.unknown()).optional()
});

export const ContentBlockSchema = z.union([
    TextContentSchema,
    ImageContentSchema,
    EmbeddedResourceSchema
]);

export const RoleSchema = z.enum(["user", "assistant"]);

export const MessageSchema = z.object({
    role: RoleSchema,
    content: z.union([ContentBlockSchema, z.array(ContentBlockSchema)]),
    _meta: z.object({}).passthrough().optional()
});

// --- Initialize ---

export const ClientInfoSchema = z.object({
    name: z.string(),
    version: z.string(),
    websiteUrl: z.string().optional()
});

export const ServerInfoSchema = ClientInfoSchema;

export const ClientCapabilitiesSchema = z.object({
    experimental: z.record(z.string(), z.unknown()).optional(),
    sampling: z.object({}).optional(),
    roots: z.object({ listChanged: z.boolean().optional() }).optional()
});

export const ServerCapabilitiesSchema = z.object({
    experimental: z.record(z.string(), z.unknown()).optional(),
    logging: z.object({}).optional(),
    prompts: z.object({ listChanged: z.boolean().optional() }).optional(),
    resources: z.object({ listChanged: z.boolean().optional(), subscribe: z.boolean().optional() }).optional(),
    tools: z.object({ listChanged: z.boolean().optional() }).optional()
});

export const InitializeRequestSchema = z.object({
    method: z.literal("initialize"),
    params: z.object({
        protocolVersion: z.string(),
        capabilities: ClientCapabilitiesSchema,
        clientInfo: ClientInfoSchema,
        _meta: z.object({}).passthrough().optional()
    })
});

export const InitializeResultSchema = z.object({
    protocolVersion: z.string(),
    capabilities: ServerCapabilitiesSchema,
    serverInfo: ServerInfoSchema,
    instructions: z.string().optional(),
    _meta: z.object({}).passthrough().optional()
});

// --- Tools ---

export const ToolSchema = z.object({
    name: z.string(),
    description: z.string().optional(),
    inputSchema: z.object({
        type: z.literal("object"),
        properties: z.record(z.string(), z.unknown()).optional(),
        required: z.array(z.string()).optional()
    }).passthrough(),
    _meta: z.object({}).passthrough().optional()
});

export const ListToolsRequestSchema = z.object({
    params: z.object({
        cursor: z.string().optional(),
        _meta: z.object({}).passthrough().optional()
    }).optional()
});

export const ListToolsResultSchema = z.object({
    tools: z.array(ToolSchema),
    nextCursor: z.string().optional(),
    _meta: z.object({}).passthrough().optional()
});

export const CallToolRequestSchema = z.object({
    params: z.object({
        name: z.string(),
        arguments: z.record(z.string(), z.unknown()).optional(),
        _meta: z.object({}).passthrough().optional()
    })
});

export const ToolResultSchema = z.object({
    content: z.array(ContentBlockSchema).default([]),
    isError: z.boolean().optional(),
    _meta: z.object({}).passthrough().optional()
});

// --- Tasks ---

export const TaskStatusSchema = z.enum(["working", "input_required", "completed", "failed", "cancelled"]);

export const TaskSchema = z.object({
    taskId: z.string(),
    status: TaskStatusSchema,
    ttl: z.number(),
    createdAt: z.string(),
    lastUpdatedAt: z.string(),
    pollInterval: z.number().optional(),
    statusMessage: z.string().optional()
});

export const ListResourcesRequestSchema = McpRequestSchema("resources/list", z.object({
    cursor: z.string().optional()
}).optional());

export const ListResourcesResultSchema = z.object({
    resources: z.array(z.object({
        uri: z.string(),
        name: z.string(),
        description: z.string().optional(),
        mimeType: z.string().optional()
    })),
    nextCursor: z.string().optional()
});

export const ListPromptsRequestSchema = McpRequestSchema("prompts/list", z.object({
    cursor: z.string().optional()
}).optional());

export const ListPromptsResultSchema = z.object({
    prompts: z.array(z.object({
        name: z.string(),
        description: z.string().optional(),
        arguments: z.array(z.object({
            name: z.string(),
            description: z.string().optional(),
            required: z.boolean().optional()
        })).optional()
    })),
    nextCursor: z.string().optional()
});

// --- Types ---

export type JsonRpcRequest = z.infer<typeof JsonRpcRequestSchema>;
export type JsonRpcNotification = z.infer<typeof JsonRpcNotificationSchema>;
export type JsonRpcResponse = z.infer<typeof JsonRpcResponseSchema>;
export type JsonRpcError = z.infer<typeof JsonRpcErrorSchema>;
export type JsonRpcMessage = z.infer<typeof JsonRpcMessageSchema>;
export type TextContent = z.infer<typeof TextContentSchema>;
export type ImageContent = z.infer<typeof ImageContentSchema>;
export type ContentBlock = z.infer<typeof ContentBlockSchema>;
export type Message = z.infer<typeof MessageSchema>;
export type Tool = z.infer<typeof ToolSchema>;
export type ClientCapabilities = z.infer<typeof ClientCapabilitiesSchema>;
export type ServerCapabilities = z.infer<typeof ServerCapabilitiesSchema>;
export type Task = z.infer<typeof TaskSchema>;

export const McpSchemas = {
    InitializeRequestSchema,
    InitializeResultSchema,
    ListToolsRequestSchema,
    ListToolsResultSchema,
    CallToolRequestSchema,
    ToolResultSchema,
    TaskSchema
};
