/**
 * File: src/utils/json/rpc.ts
 * Role: Utility functions for JSON-RPC message validation.
 */

export interface JsonRpcResponse {
    jsonrpc?: string;
    result?: any;
    error?: any;
    id?: string | number | null;
}

export interface JsonRpcRequest {
    jsonrpc?: string;
    method: string;
    params?: any;
    id?: string | number | null;
}

/**
 * Checks if a given object is a JSON-RPC response (contains 'result' or 'error').
 */
export function isJsonRpcResponse(obj: any): obj is JsonRpcResponse {
    if (!obj || typeof obj !== 'object') return false;
    return "result" in obj || "error" in obj;
}

/**
 * Checks if a given object is a JSON-RPC request (contains a string 'method').
 */
export function isJsonRpcRequest(obj: any): obj is JsonRpcRequest {
    if (!obj || typeof obj !== 'object') return false;
    return "method" in obj && typeof obj.method === "string";
}
