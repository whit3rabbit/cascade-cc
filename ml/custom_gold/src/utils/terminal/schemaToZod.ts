
import { z } from 'zod';

// fa5
export function jsonSchemaToZod(schema: any): z.ZodTypeAny {
    if (schema.type === "string" && (schema.enum || schema.oneOf)) {
        const options = schema.enum || schema.oneOf.map((o: any) => o.const);
        if (!options || options.length === 0) return z.never();
        return z.enum(options as [string, ...string[]]);
    }

    if (schema.type === "string") {
        let zod = z.string();
        if (schema.minLength !== undefined) zod = zod.min(schema.minLength);
        if (schema.maxLength !== undefined) zod = zod.max(schema.maxLength);

        switch (schema.format) {
            case "email": return zod.email();
            case "uri": return zod.url();
            case "date": return zod.regex(/^\d{4}-\d{2}-\d{2}$/, "Invalid date format (YYYY-MM-DD)");
            case "date-time": return zod.datetime({ offset: true });
        }
        return zod;
    }

    if (schema.type === "number" || schema.type === "integer") {
        let zod = z.coerce.number();
        if (schema.type === "integer") zod = zod.int();
        if (schema.minimum !== undefined) zod = zod.min(schema.minimum);
        if (schema.maximum !== undefined) zod = zod.max(schema.maximum);
        return zod;
    }

    if (schema.type === "boolean") {
        return z.coerce.boolean();
    }

    throw new Error(`Unsupported schema: ${JSON.stringify(schema)}`);
}

// dC0
export function validateAgainstSchema(value: any, schema: any) {
    try {
        const zod = jsonSchemaToZod(schema);
        const result = zod.safeParse(value);
        if (result.success) {
            return { value: result.data, isValid: true };
        }
        return {
            isValid: false,
            error: result.error.errors.map((e: any) => e.message).join("; ")
        };
    } catch (err: any) {
        return { isValid: false, error: err.message };
    }
}
