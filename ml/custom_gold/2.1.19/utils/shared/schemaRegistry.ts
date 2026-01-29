/**
 * File: src/utils/shared/schemaRegistry.ts
 * Role: Registry for managing and traversing schema definitions, typically for Zod-like structures.
 */

/**
 * A registry that maps objects (schemas) to metadata, with ID-based lookup and parent inheritance.
 */
export class Registry {
    private _map = new WeakMap<object, any>();
    private _idmap = new Map<string, object>();

    /**
     * Adds an object to the registry with associated metadata.
     */
    add(object: object, value: any): this {
        this._map.set(object, value);
        if (value && typeof value === "object" && "id" in value) {
            if (this._idmap.has(value.id)) {
                throw new Error(`ID ${value.id} already exists in the registry`);
            }
            this._idmap.set(value.id, object);
        }
        return this;
    }

    /**
     * Removes an object from the registry.
     */
    remove(object: object): this {
        this._map.delete(object);
        return this;
    }

    /**
     * Retrieves metadata for an object, traversing the parent chain if it exists.
     */
    get(object: any): any {
        const parent = object._zod?.parent;
        if (parent) {
            const parentValue = this.get(parent) ?? {};
            // eslint-disable-next-line @typescript-eslint/no-unused-vars
            const { id, ...parentProps } = parentValue;
            return {
                ...parentProps,
                ...(this._map.get(object) || {})
            };
        }
        return this._map.get(object);
    }

    /**
     * Checks if an object is present in the registry.
     */
    has(object: object): boolean {
        return this._map.has(object);
    }

    /**
     * Resolves an object by its registered ID.
     */
    getById(id: string): object | undefined {
        return this._idmap.get(id);
    }
}

/**
 * Helper to create a new Registry instance.
 */
export function createRegistry(): Registry {
    return new Registry();
}

// Global registry instance
export const registry = createRegistry();

// Schema factory helpers (mimicking the original's Zod-like factory pattern)

export type SchemaOptions = Record<string, any>;

export function createStringSchema(Type: any, options?: SchemaOptions) {
    return new Type({ type: "string", ...(options ?? {}) });
}

export function createNumberSchema(Type: any, options?: SchemaOptions) {
    return new Type({ type: "number", checks: [], ...(options ?? {}) });
}

export function createFormattedStringSchema(Type: any, format: string, options?: SchemaOptions) {
    return new Type({
        type: "string",
        format,
        check: "string_format",
        abort: false,
        ...(options ?? {})
    });
}
