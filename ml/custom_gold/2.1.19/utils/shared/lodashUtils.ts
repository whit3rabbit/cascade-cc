/**
 * File: src/utils/shared/lodashUtils.ts
 * Role: Lightweight utility functions mimicking popular lodash patterns for type checks and object manipulation.
 */

const objectToString = Object.prototype.toString;

/**
 * Checks if a value is an object or function.
 */
export function isObject(value: any): value is object {
    const type = typeof value;
    return value != null && (type === 'object' || type === 'function');
}

/**
 * Returns the internal [[Class]] tag of an object.
 */
export function getTag(value: any): string {
    if (value == null) {
        return value === undefined ? '[object Undefined]' : '[object Null]';
    }
    return objectToString.call(value);
}

/**
 * Checks if a value is a function, including async and generator functions.
 */
export function isFunction(value: any): value is Function {
    if (!isObject(value)) return false;
    const tag = getTag(value);
    return tag === '[object Function]' ||
        tag === '[object GeneratorFunction]' ||
        tag === '[object AsyncFunction]' ||
        tag === '[object Proxy]';
}

/**
 * Checks if a value is a native function.
 */
export function isNative(value: any): boolean {
    if (!isObject(value)) return false;
    // Patterns for native code identification
    return (
        /^\[object .+?Constructor\]$/.test(getTag(value)) ||
        (typeof value.toString === 'function' && /\[native code\]/.test(value.toString()))
    );
}

/**
 * Safely retrieves a value from an object given a key.
 */
export function getValue(object: any, key: string | number | symbol): any {
    return object == null ? undefined : object[key];
}

/**
 * Maps values of an object using an iteratee function.
 */
export function mapValues<T, U>(
    object: Record<string, T>,
    iteratee: (value: T, key: string, obj: Record<string, T>) => U
): Record<string, U> {
    const result: Record<string, U> = {};
    if (object) {
        Object.keys(object).forEach((key) => {
            result[key] = iteratee(object[key], key, object);
        });
    }
    return result;
}

/**
 * Global root object detection.
 */
export const root = (typeof global === 'object' && global && global.Object === Object && global) ||
    (typeof self === 'object' && self && self.Object === Object && self) ||
    Function('return this')();

/**
 * Global Symbol reference.
 */
export const Symbol = root.Symbol;
