/**
 * File: src/utils/shared/errors.ts
 * Role: Common error classes used across the application.
 */

export class RequestAbortedError extends Error {
    constructor(message: string = "Request aborted") {
        super(message);
        this.name = "RequestAbortedError";
    }
}

export class NotSupportedError extends Error {
    constructor(message: string = "Not supported") {
        super(message);
        this.name = "NotSupportedError";
    }
}

export class InvalidArgumentError extends Error {
    constructor(message: string = "Invalid argument") {
        super(message);
        this.name = "InvalidArgumentError";
    }
}
