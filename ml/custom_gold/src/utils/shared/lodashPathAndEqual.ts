/**
 * Deep equality and path resolution utilities.
 * Partial deobfuscation of lodash-like functions in chunk_2.ts.
 */

/**
 * Deep equality check.
 * Deobfuscated from gj0 in chunk_2.ts.
 */
export function isEqual(value: any, other: any): boolean {
    if (value === other) return true;
    if (value == null || other == null || (typeof value !== 'object' && typeof other !== 'object')) {
        return value !== value && other !== other;
    }
    return baseIsEqual(value, other);
}

function baseIsEqual(value: any, other: any): boolean {
    // Simplified version of the deep equality logic in chunk_2.ts
    if (typeof value !== typeof other) return false;
    if (Array.isArray(value)) {
        if (!Array.isArray(other) || value.length !== other.length) return false;
        for (let i = 0; i < value.length; i++) {
            if (!isEqual(value[i], other[i])) return false;
        }
        return true;
    }
    if (typeof value === 'object' && value !== null && other !== null) {
        const keysA = Object.keys(value);
        const keysB = Object.keys(other);
        if (keysA.length !== keysB.length) return false;
        for (const key of keysA) {
            if (!Object.prototype.hasOwnProperty.call(other, key) || !isEqual(value[key], other[key])) {
                return false;
            }
        }
        return true;
    }
    return value === other;
}

/**
 * Retrieves value at path from object.
 * Deobfuscated from Vw9 in chunk_2.ts.
 */
export function get(object: any, path: string | string[], defaultValue?: any): any {
    const result = object == null ? undefined : baseGet(object, castPath(path));
    return result === undefined ? defaultValue : result;
}

function baseGet(object: any, path: string[]): any {
    let index = 0;
    const length = path.length;
    while (object != null && index < length) {
        object = object[path[index++]];
    }
    return (index && index == length) ? object : undefined;
}

/**
 * Checks if property path exists on object.
 * Deobfuscated from Fw9 in chunk_2.ts.
 */
export function has(object: any, path: string | string[]): boolean {
    const castedPath = castPath(path);
    let index = -1;
    let length = castedPath.length;
    let result = false;

    while (++index < length) {
        const key = castedPath[index];
        if (!(result = object != null && Object.prototype.hasOwnProperty.call(object, key))) {
            break;
        }
        object = object[key];
    }
    if (result || ++index != length) {
        return result;
    }
    length = object == null ? 0 : object.length;
    return !!length && Number.isInteger(length) && length >= 0 && (Array.isArray(object) || isArguments(object));
}

function castPath(value: any): string[] {
    if (Array.isArray(value)) return value;
    if (typeof value === 'string') {
        return value.split(/[.[\]]+/).filter(Boolean);
    }
    return [String(value)];
}

function isArguments(value: any): boolean {
    return value != null && typeof value === 'object' && Object.prototype.toString.call(value) === '[object Arguments]';
}

/**
 * Returns object keys.
 */
export function keys(object: any): string[] {
    return object == null ? [] : Object.keys(object);
}
