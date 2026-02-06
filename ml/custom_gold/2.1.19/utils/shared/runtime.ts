/**
 * File: src/utils/shared/runtime.ts
 * Role: Core runtime type system for validation and schema definition (Zod-like).
 */

export type ParseStatus = "valid" | "dirty" | "aborted";

export interface ParseResult<T> {
    success: boolean;
    data?: T;
    error?: any;
}

/**
 * Base class for all runtime types.
 */
export abstract class ZodType<T = any> {
    readonly _def: any;
    "~standard" = { version: 1, vendor: "zod", validate: (v: any) => this.safeParse(v) };

    constructor(def: any) {
        this._def = def;
    }

    abstract _parse(input: any): any;

    parse(data: any): T {
        const result = this.safeParse(data);
        if (result.success) return result.data!;
        throw result.error;
    }

    safeParse(data: any): ParseResult<T> {
        try {
            const parsed = this._parse(data);
            return { success: true, data: parsed };
        } catch (e) {
            return { success: false, error: e };
        }
    }

    optional(): ZodOptional<this> {
        return new ZodOptional(this);
    }

    nullable(): ZodNullable<this> {
        return new ZodNullable(this);
    }

    describe(description: string): this {
        this._def.description = description;
        return this;
    }
}

/**
 * String type.
 */
export class ZodString extends ZodType<string> {
    _parse(input: any) {
        if (typeof input !== "string") throw new Error("Expected string");
        return input;
    }
}

/**
 * Number type.
 */
export class ZodNumber extends ZodType<number> {
    _parse(input: any) {
        if (typeof input !== "number") throw new Error("Expected number");
        return input;
    }
}

/**
 * Boolean type.
 */
export class ZodBoolean extends ZodType<boolean> {
    _parse(input: any) {
        if (typeof input !== "boolean") throw new Error("Expected boolean");
        return input;
    }
}

/**
 * Object type.
 */
export class ZodObject<T extends Record<string, ZodType>> extends ZodType<{ [K in keyof T]: any }> {
    _parse(input: any) {
        if (typeof input !== "object" || input === null) throw new Error("Expected object");
        const result: any = {};
        for (const [key, schema] of Object.entries(this._def.properties)) {
            result[key] = (schema as ZodType)._parse(input[key]);
        }
        return result;
    }
}

/**
 * Optional wrapper.
 */
export class ZodOptional<T extends ZodType> extends ZodType<any> {
    constructor(private inner: T) {
        super({ ...inner._def, isOptional: true });
    }
    _parse(input: any) {
        if (input === undefined) return undefined;
        return this.inner._parse(input);
    }
}

/**
 * Nullable wrapper.
 */
export class ZodNullable<T extends ZodType> extends ZodType<any> {
    constructor(private inner: T) {
        super({ ...inner._def, isNullable: true });
    }
    _parse(input: any) {
        if (input === null) return null;
        return this.inner._parse(input);
    }
}

// --- Factory Functions ---

export const z = {
    string: () => new ZodString({ type: "string" }),
    number: () => new ZodNumber({ type: "number" }),
    boolean: () => new ZodBoolean({ type: "boolean" }),
    object: <T extends Record<string, ZodType>>(props: T) => new ZodObject<T>({ type: "object", properties: props }),
};

// Aliases for compatibility with the deobfuscated code's exports
export { z as TypeFactory };
export const createStringSchema = z.string;
export const createNumberSchema = z.number;
export const createBooleanSchema = z.boolean;
export const createObjectSchema = z.object;

import { EnvService } from '../../services/config/EnvService.js';

/**
 * Checks if the current environment is running in demo mode.
 * @returns {boolean}
 */
export function isDemo(): boolean {
    return EnvService.isTruthy('CLAUDE_CODE_DEMO');
}

/**
 * Diagnostics stub.
 */
export const diag = {
    debug: (..._args: any[]) => { /* console.debug('[Diag]', ...args); */ },
    info: (..._args: any[]) => { /* console.info('[Diag]', ...args); */ },
    warn: (...args: any[]) => { console.warn('[Diag]', ...args); },
    error: (...args: any[]) => { console.error('[Diag]', ...args); }
};

/**
 * Creates a unique context key.
 * @param {string} description - The description of the key.
 * @returns {symbol} A unique symbol for the context key.
 */
export function createContextKey(description: string): symbol {
    return Symbol.for(description);
}

/**
 * Logs a message to the terminal.
 */
export function terminalLog(message: string, level: string = "info"): void {
    if (level === "error") {
        console.error(message);
    } else if (level === "warn") {
        console.warn(message);
    } else {
        console.log(message);
    }
}

export const errorLog = (err: Error | string) => console.error(err);
export const infoLog = (msg: string, ...args: any[]) => console.info(msg, ...args);
export const m1 = () => process.cwd();
