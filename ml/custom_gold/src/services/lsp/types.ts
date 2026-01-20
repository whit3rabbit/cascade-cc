
// Logic from CZ0

export namespace ErrorCodes {
    export const ParseError = -32700;
    export const InvalidRequest = -32600;
    export const MethodNotFound = -32601;
    export const InvalidParams = -32602;
    export const InternalError = -32603;
    export const jsonrpcReservedErrorRangeStart = -32099;
    export const serverErrorStart = -32099;
    export const MessageWriteError = -32099;
    export const MessageReadError = -32098;
    export const PendingResponseRejected = -32097;
    export const ConnectionInactive = -32096;
    export const ServerNotInitialized = -32002;
    export const UnknownErrorCode = -32001;
    export const jsonrpcReservedErrorRangeEnd = -32000;
    export const serverErrorEnd = -32000;
}

export class ResponseError extends Error {
    public readonly code: number;
    public readonly data: any;

    constructor(code: number, message: string, data?: any) {
        super(message);
        this.code = Number.isInteger(code) ? code : ErrorCodes.UnknownErrorCode;
        this.data = data;
        Object.setPrototypeOf(this, ResponseError.prototype);
    }

    public toJson() {
        return {
            code: this.code,
            message: this.message,
            data: this.data
        };
    }
}

export abstract class AbstractMessageSignature {
    public readonly method: string;
    public readonly numberOfParams: number;

    constructor(method: string, numberOfParams: number) {
        this.method = method;
        this.numberOfParams = numberOfParams;
    }
}

export class RequestType<P, R, E> extends AbstractMessageSignature {
    constructor(method: string) {
        super(method, 1);
    }
}

export class NotificationType<P> extends AbstractMessageSignature {
    constructor(method: string) {
        super(method, 1);
    }
}

export interface Message {
    jsonrpc: string;
}

export interface RequestMessage extends Message {
    id: number | string;
    method: string;
    params?: any;
}

export interface ResponseMessage extends Message {
    id: number | string | null;
    result?: any;
    error?: ResponseError;
}

export interface NotificationMessage extends Message {
    method: string;
    params?: any;
}
