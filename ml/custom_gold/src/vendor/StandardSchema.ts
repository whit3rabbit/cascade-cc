
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
    readonly message: string;
    readonly path?: ReadonlyArray<string | number>;
}

export abstract class BaseSchema<T = any> implements StandardSchemaV1<any, T> {
    _zod: {
        def: any;
        bag: any;
        version: string;
        run: (payload: any, ctx: any) => any;
        parse: (payload: any, ctx: any) => any;
        onattach: Array<(inst: any) => void>;
        traits: Set<string>;
        pattern?: RegExp;
        values?: Set<any>;
        optin?: "optional";
        optout?: "optional";
        propValues?: Record<string, Set<any>>;
    };

    constructor(def: any = {}) {
        this._zod = {
            def,
            bag: {},
            version: "1.0.0", // Dummy version
            run: (p, c) => this._zod.parse(p, c),
            parse: (p, c) => p,
            onattach: [],
            traits: new Set(["$ZodType"])
        };

        this["~standard"] = {
            version: 1,
            vendor: "zod",
            validate: (value: any) => {
                const payload = { value, issues: [] };
                const result = this._zod.run(payload, { async: true });
                if (result instanceof Promise) {
                    return result.then((res) => {
                        if (res.issues.length) return { issues: res.issues };
                        return { value: res.value };
                    });
                }
                if (result.issues.length) return { issues: result.issues };
                return { value: result.value };
            }
        };
    }

    readonly "~standard": StandardSchemaV1<any, T>["~standard"];

    // Helper for attaching traits/checks
    static init(inst: any, def: any) {
        inst._zod.def = def;
        // ... logic from lines 11-52 of chunk_753
    }
}

export class StringSchema extends BaseSchema<string> {
    constructor(def: any = {}) {
        super(def);
        this._zod.traits.add("$ZodString");
        this._zod.parse = (payload: any, ctx: any) => {
            if (def.coerce) {
                payload.value = String(payload.value);
            }
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

export class NumberSchema extends BaseSchema<number> {
    constructor(def: any = {}) {
        super(def);
        this._zod.traits.add("$ZodNumber");
        this._zod.parse = (payload: any, ctx: any) => {
            if (def.coerce) {
                payload.value = Number(payload.value);
            }
            const val = payload.value;
            if (typeof val === "number" && !Number.isNaN(val) && Number.isFinite(val)) return payload;

            const received = typeof val === "number" ?
                (Number.isNaN(val) ? "NaN" : (!Number.isFinite(val) ? "Infinity" : undefined)) : undefined;

            payload.issues.push({
                expected: "number",
                code: "invalid_type",
                input: val,
                inst: this,
                ...(received ? { received } : {})
            });
            return payload;
        };
    }
}

// ... and so on for other types
