
/**
 * Internal Zod-like validation library implementing the Standard Schema interface.
 * Based on deobfuscated logic from chunk_753.ts and chunk_754.ts.
 */

export interface StandardSchemaV1<I = any, O = I> {
    readonly "~standard": {
        readonly version: 1;
        readonly vendor: "zod";
        readonly validate: (value: unknown) => StandardResult<O> | Promise<StandardResult<O>>;
        readonly types?: {
            readonly input: I;
            readonly output: O;
        };
    };
}

export type StandardResult<T> = StandardSuccessResult<T> | StandardFailureResult;

export interface StandardSuccessResult<T> {
    readonly value: T;
    readonly issues?: undefined;
}

export interface StandardFailureResult {
    readonly issues: ReadonlyArray<StandardIssue>;
    readonly value?: undefined;
}

export interface StandardIssue {
    readonly message?: string;
    readonly path?: ReadonlyArray<string | number>;
    readonly code?: string;
    readonly expected?: string;
    readonly input?: any;
    readonly inst?: any;
    readonly format?: string;
    readonly note?: string;
    readonly pattern?: string;
    readonly continue?: boolean;
}

export abstract class BaseSchema<T = any> implements StandardSchemaV1<any, T> {
    _zod: {
        def: any;
        bag: any;
        version: string;
        run: (payload: any, ctx: any) => any;
        parse: (payload: any, ctx: any) => any;
        check?: (payload: any) => any;
        onattach: Array<(inst: any) => void>;
        traits: Set<string>;
        pattern?: RegExp;
        values?: Set<any>;
        optin?: "optional";
        optout?: "optional";
        propValues?: Record<string, Set<any>>;
        deferred?: Array<() => void>;
    };

    constructor(def: any = {}) {
        this._zod = {
            def,
            bag: def._zod?.bag || {},
            version: "1.0.0",
            run: (p, c) => this._zod.parse(p, c),
            parse: (p, c) => p,
            onattach: [],
            traits: new Set(["$ZodType"])
        };

        // Standard Schema implementation logic from chunk_753
        this["~standard"] = {
            version: 1,
            vendor: "zod",
            validate: (value: any) => {
                try {
                    const payload = { value, issues: [] };
                    const result = this._zod.run(payload, { async: false });
                    if (result instanceof Promise) {
                        return result.then(r => r.issues.length ? { issues: r.issues } : { value: r.value });
                    }
                    return result.issues.length ? { issues: result.issues } : { value: result.value };
                } catch (e) {
                    // Fallback to async if sync fails (matching chunk_753 line 62)
                    const payload = { value, issues: [] };
                    return Promise.resolve(this._zod.run(payload, { async: true })).then(r =>
                        r.issues.length ? { issues: r.issues } : { value: r.value }
                    );
                }
            }
        };
    }

    readonly "~standard": StandardSchemaV1<any, T>["~standard"];

    static attachTraits(inst: any, traits: string[]) {
        for (const trait of traits) inst._zod.traits.add(trait);
    }
}

export class StringSchema extends BaseSchema<string> {
    constructor(def: any = {}) {
        super(def);
        this._zod.traits.add("$ZodString");
        this._zod.parse = (payload: any, ctx: any) => {
            if (def.coerce) payload.value = String(payload.value);
            if (typeof payload.value === "string") return payload;
            payload.issues.push({
                expected: "string",
                code: "invalid_type",
                input: payload.value,
                inst: this
            });
            return payload;
        };
    }
}

export class URLSchema extends StringSchema {
    constructor(def: any = {}) {
        super(def);
        this._zod.traits.add("$ZodURL");
        this._zod.check = (payload: any) => {
            try {
                const url = new URL(payload.value);
                if (def.hostname && !def.hostname.test(url.hostname)) {
                    payload.issues.push({ code: "invalid_format", format: "url", note: "Invalid hostname", input: payload.value, inst: this });
                }
                if (def.protocol && !def.protocol.test(url.protocol.replace(/:$/, ""))) {
                    payload.issues.push({ code: "invalid_format", format: "url", note: "Invalid protocol", input: payload.value, inst: this });
                }
                payload.value = url.href;
            } catch (e) {
                payload.issues.push({ code: "invalid_format", format: "url", input: payload.value, inst: this });
            }
        };
    }
}

export class ArraySchema<T = any> extends BaseSchema<T[]> {
    constructor(element: BaseSchema<T>, def: any = {}) {
        super({ ...def, element });
        this._zod.traits.add("$ZodArray");
        this._zod.parse = (payload: any, ctx: any) => {
            const input = payload.value;
            if (!Array.isArray(input)) {
                payload.issues.push({ expected: "array", code: "invalid_type", input, inst: this });
                return payload;
            }
            payload.value = new Array(input.length);
            const promises: Promise<any>[] = [];
            for (let i = 0; i < input.length; i++) {
                const result = element._zod.run({ value: input[i], issues: [] }, ctx);
                if (result instanceof Promise) {
                    promises.push(result.then(r => {
                        if (r.issues.length) payload.issues.push(...r.issues.map((iss: any) => ({ ...iss, path: [i, ...(iss.path || [])] })));
                        payload.value[i] = r.value;
                    }));
                } else {
                    if (result.issues.length) payload.issues.push(...result.issues.map((iss: any) => ({ ...iss, path: [i, ...(iss.path || [])] })));
                    payload.value[i] = result.value;
                }
            }
            return promises.length ? Promise.all(promises).then(() => payload) : payload;
        };
    }
}

export class ObjectSchema<T extends Record<string, any>> extends BaseSchema<T> {
    constructor(shape: Record<string, BaseSchema>, def: any = {}) {
        super({ ...def, shape });
        this._zod.traits.add("$ZodObject");
        const keys = Object.keys(shape);

        this._zod.parse = (payload: any, ctx: any) => {
            const input = payload.value;
            if (typeof input !== "object" || input === null) {
                payload.issues.push({ expected: "object", code: "invalid_type", input, inst: this });
                return payload;
            }

            const newResult: Record<string, any> = {};
            const promises: Promise<any>[] = [];

            for (const key of keys) {
                const schema = shape[key];
                const result = schema._zod.run({ value: input[key], issues: [] }, ctx);

                const processResult = (r: any) => {
                    if (r.issues.length) {
                        payload.issues.push(...r.issues.map((iss: any) => ({
                            ...iss,
                            path: [key, ...(iss.path || [])]
                        })));
                    }
                    newResult[key] = r.value;
                };

                if (result instanceof Promise) {
                    promises.push(result.then(processResult));
                } else {
                    processResult(result);
                }
            }

            if (!def.catchall) {
                const unrecognized = Object.keys(input).filter(k => !keys.includes(k));
                if (unrecognized.length) {
                    payload.issues.push({ code: "unrecognized_keys", keys: unrecognized, input, inst: this });
                }
            }

            if (promises.length) {
                return Promise.all(promises).then(() => {
                    payload.value = newResult;
                    return payload;
                });
            }
            payload.value = newResult;
            return payload;
        };
    }
}

// Factory functions matching the library's exports
export class EnumSchema<T extends string> extends BaseSchema<T> {
    constructor(values: T[], def: any = {}) {
        super({ ...def, values });
        this._zod.traits.add("$ZodEnum");
        this._zod.values = new Set(values);
        this._zod.parse = (payload: any, ctx: any) => {
            if (this._zod.values?.has(payload.value)) return payload;
            payload.issues.push({ code: "invalid_value", values, input: payload.value, inst: this });
            return payload;
        };
    }
}

export class LiteralSchema<T> extends BaseSchema<T> {
    constructor(value: T, def: any = {}) {
        super({ ...def, value });
        this._zod.traits.add("$ZodLiteral");
        this._zod.values = new Set([value]);
        this._zod.parse = (payload: any, ctx: any) => {
            if (payload.value === value) return payload;
            payload.issues.push({ code: "invalid_value", values: [value], input: payload.value, inst: this });
            return payload;
        };
    }
}

export class FileSchema extends BaseSchema<File> {
    constructor(def: any = {}) {
        super(def);
        this._zod.traits.add("$ZodFile");
        this._zod.parse = (payload: any, ctx: any) => {
            if (payload.value instanceof File) return payload;
            payload.issues.push({ expected: "file", code: "invalid_type", input: payload.value, inst: this });
            return payload;
        };
    }
}

export class RecordSchema<K extends string | number | symbol, V> extends BaseSchema<Record<K, V>> {
    constructor(keyType: BaseSchema<K>, valueType: BaseSchema<V>, def: any = {}) {
        super({ ...def, keyType, valueType });
        this._zod.traits.add("$ZodRecord");
        this._zod.parse = (payload: any, ctx: any) => {
            const input = payload.value;
            if (typeof input !== "object" || input === null) {
                payload.issues.push({ expected: "record", code: "invalid_type", input, inst: this });
                return payload;
            }
            const newResult: any = {};
            const promises: Promise<any>[] = [];
            for (const key of Object.keys(input)) {
                const keyResult = keyType._zod.run({ value: key, issues: [] }, ctx);
                if (keyResult.issues.length) {
                    payload.issues.push({ code: "invalid_key", input: key, path: [key], inst: this });
                    continue;
                }
                const valResult = valueType._zod.run({ value: input[key], issues: [] }, ctx);
                const processVal = (r: any) => {
                    if (r.issues.length) {
                        payload.issues.push(...r.issues.map((iss: any) => ({ ...iss, path: [key, ...(iss.path || [])] })));
                    }
                    newResult[keyResult.value] = r.value;
                };
                if (valResult instanceof Promise) {
                    promises.push(valResult.then(processVal));
                } else {
                    processVal(valResult);
                }
            }
            if (promises.length) return Promise.all(promises).then(() => { payload.value = newResult; return payload; });
            payload.value = newResult;
            return payload;
        };
    }
}

export class OptionalSchema<T> extends BaseSchema<T | undefined> {
    constructor(innerType: BaseSchema<T>, def: any = {}) {
        super({ ...def, innerType });
        this._zod.traits.add("$ZodOptional");
        this._zod.optin = "optional";
        this._zod.optout = "optional";
        this._zod.parse = (payload: any, ctx: any) => {
            if (payload.value === undefined) return payload;
            return innerType._zod.run(payload, ctx);
        };
    }
}

export class NullableSchema<T> extends BaseSchema<T | null> {
    constructor(innerType: BaseSchema<T>, def: any = {}) {
        super({ ...def, innerType });
        this._zod.traits.add("$ZodNullable");
        this._zod.parse = (payload: any, ctx: any) => {
            if (payload.value === null) return payload;
            return innerType._zod.run(payload, ctx);
        };
    }
}

export class DefaultSchema<T> extends BaseSchema<T> {
    constructor(innerType: BaseSchema<T>, defaultValue: T, def: any = {}) {
        super({ ...def, innerType, defaultValue });
        this._zod.traits.add("$ZodDefault");
        this._zod.optin = "optional";
        this._zod.parse = (payload: any, ctx: any) => {
            if (payload.value === undefined) {
                payload.value = defaultValue;
                return payload;
            }
            return innerType._zod.run(payload, ctx);
        };
    }
}

// Factory functions matching the library's exports
export const zStandard = {
    string: (def?: any) => new StringSchema(def),
    url: (def?: any) => new URLSchema(def),
    array: <T>(element: BaseSchema<T>, def?: any) => new ArraySchema(element, def),
    object: <T extends Record<string, any>>(shape: Record<string, BaseSchema>, def?: any) => new ObjectSchema<T>(shape, def),
    enum: <T extends string>(values: T[], def?: any) => new EnumSchema<T>(values, def),
    literal: <T>(value: T, def?: any) => new LiteralSchema<T>(value, def),
    file: (def?: any) => new FileSchema(def),
    record: <K extends string | number | symbol, V>(keyType: BaseSchema<K>, valueType: BaseSchema<V>, def?: any) => new RecordSchema<K, V>(keyType, valueType, def),
    optional: <T>(inner: BaseSchema<T>) => new OptionalSchema<T>(inner),
    nullable: <T>(inner: BaseSchema<T>) => new NullableSchema<T>(inner),
    default: <T>(inner: BaseSchema<T>, val: T) => new DefaultSchema<T>(inner, val),
};
