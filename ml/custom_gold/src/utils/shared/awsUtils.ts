
/**
 * Utility functions for handling AWS SDK errors and configuration.
 * Based on deobfuscated logic from chunk_729.ts and chunk_730.ts.
 */

export class ServiceException extends Error {
    $fault?: "client" | "server";
    $metadata?: any;
    constructor(info: { name: string; message: string; $fault?: "client" | "server"; $metadata?: any }) {
        super(info.message);
        this.name = info.name;
        this.$fault = info.$fault;
        this.$metadata = info.$metadata;
        Object.setPrototypeOf(this, ServiceException.prototype);
    }
}

export function decorateServiceException(exception: any, overrides: any = {}) {
    Object.entries(overrides).forEach(([key, value]) => {
        if (value !== undefined && (exception[key] == null || exception[key] === "")) {
            exception[key] = value;
        }
    });
    exception.message = exception.message || exception.Message || "UnknownError";
    delete exception.Message;
    return exception;
}

export function deserializeMetadata(response: any) {
    return {
        httpStatusCode: response.statusCode,
        requestId: response.headers["x-amzn-requestid"] ?? response.headers["x-amzn-request-id"] ?? response.headers["x-amz-request-id"],
        extendedRequestId: response.headers["x-amz-id-2"],
        cfId: response.headers["x-amz-cf-id"]
    };
}

export function throwDefaultError({ output, parsedBody, exceptionCtor, errorCode }: any) {
    const metadata = deserializeMetadata(output);
    const statusCode = metadata.httpStatusCode ? String(metadata.httpStatusCode) : undefined;
    const name = parsedBody?.code || parsedBody?.Code || errorCode || statusCode || "UnknownError";
    const error = new exceptionCtor({
        name,
        message: parsedBody?.message || parsedBody?.Message || "UnknownError",
        $fault: "client",
        $metadata: metadata
    });
    throw decorateServiceException(error, parsedBody);
}

export function parseDate(value: any): Date | undefined {
    if (value == null) return undefined;
    if (typeof value === "number") return new Date(value * 1000);
    if (typeof value === "string") return new Date(value);
    return undefined;
}

export function parseBoolean(value: any): boolean | undefined {
    if (value == null) return undefined;
    if (typeof value === "boolean") return value;
    if (value === "true") return true;
    if (value === "false") return false;
    return undefined;
}
