/**
 * File: src/utils/shared/jsonSchema.ts
 * Role: Converts Zod-like schemas into JSON Schema format for tool definitions and communication.
 */

export interface JSONSchemaOptions {
    cycles?: "ref" | "throw";
    reused?: "inline" | "ref";
    external?: any;
}

/**
 * Generator class to transform internal schemas to JSON Schema (Draft 7 / 2020-12).
 */
export class JSONSchemaGenerator {
    private seen = new Map<any, any>();
    private counter = 0;
    private target: string;

    constructor(target: string = "draft-2020-12") {
        this.target = target;
    }

    /**
     * Main entry point to emit a JSON Schema from a schema definition.
     */
    emit(schema: any, options?: JSONSchemaOptions): any {
        // This is a simplified reconstruction of the massive emitter in runtime.js
        const rootSchema = this.process(schema);
        return rootSchema;
    }

    private process(schema: any): any {
        if (this.seen.has(schema)) return { $ref: `#/${this.target === "draft-2020-12" ? "$defs" : "definitions"}/${this.seen.get(schema).id}` };

        const def = schema._def || {};
        const result: any = {};

        switch (def.type) {
            case "string":
                result.type = "string";
                if (def.format) result.format = def.format;
                break;
            case "number":
                result.type = "number";
                break;
            case "boolean":
                result.type = "boolean";
                break;
            case "object":
                result.type = "object";
                result.properties = {};
                const required: string[] = [];
                for (const [key, prop] of Object.entries(def.properties || {})) {
                    result.properties[key] = this.process(prop);
                    if (!(prop as any)._def.isOptional) required.push(key);
                }
                if (required.length > 0) result.required = required;
                break;
            case "array":
                result.type = "array";
                result.items = this.process(def.itemType);
                break;
            default:
                result.type = "any";
        }

        return result;
    }
}

/**
 * Convenience helper to generate JSON schema.
 */
export function generateJSONSchema(schema: any): any {
    const gen = new JSONSchemaGenerator();
    return gen.emit(schema);
}
