export function isNotNullish<T>(value: T | null | undefined): value is T {
    return value !== undefined && value !== null;
}

export function hashAttributes(attributes: Record<string, any>): string {
    let keys = Object.keys(attributes);
    if (keys.length === 0) return "";
    keys = keys.sort();
    return JSON.stringify(keys.map((key) => [key, attributes[key]]));
}

export function instrumentationScopeId(scope: { name: string; version?: string; schemaUrl?: string }): string {
    return `${scope.name}:${scope.version ?? ""}:${scope.schemaUrl ?? ""}`;
}

export class TimeoutError extends Error {
    constructor(message: string) {
        super(message);
        Object.setPrototypeOf(this, TimeoutError.prototype);
    }
}

export function callWithTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
    let timeoutHandle: NodeJS.Timeout;
    const timeoutPromise = new Promise<never>((_, reject) => {
        timeoutHandle = setTimeout(function () {
            reject(new TimeoutError("Operation timed out."));
        }, timeoutMs);
    });

    return Promise.race([promise, timeoutPromise]).then(
        (result) => {
            clearTimeout(timeoutHandle);
            return result;
        },
        (error) => {
            clearTimeout(timeoutHandle);
            throw error;
        }
    );
}

export async function PromiseAllSettled<T>(promises: Promise<T>[]): Promise<PromiseSettledResult<T>[]> {
    return Promise.all(
        promises.map(async (p) => {
            try {
                return {
                    status: "fulfilled",
                    value: await p,
                } as PromiseFulfilledResult<T>;
            } catch (error) {
                return {
                    status: "rejected",
                    reason: error,
                } as PromiseRejectedResult;
            }
        })
    );
}

export function isPromiseAllSettledRejectionResult(result: PromiseSettledResult<any>): result is PromiseRejectedResult {
    return result.status === "rejected";
}

export function FlatMap<T, U>(arr: T[], fn: (item: T) => U[]): U[] {
    const result: U[] = [];
    arr.forEach((item) => {
        result.push(...fn(item));
    });
    return result;
}

export function setEquals<T>(setA: Set<T>, setB: Set<T>): boolean {
    if (setA.size !== setB.size) return false;
    for (const item of setA) {
        if (!setB.has(item)) return false;
    }
    return true;
}

export function binarySearchUB(arr: number[], value: number): number {
    let low = 0;
    let high = arr.length - 1;
    let result = arr.length;

    while (high >= low) {
        const mid = low + Math.trunc((high - low) / 2);
        if (arr[mid] < value) {
            low = mid + 1;
        } else {
            result = mid;
            high = mid - 1;
        }
    }
    return result;
}

export function equalsCaseInsensitive(a: string, b: string): boolean {
    return a.toLowerCase() === b.toLowerCase();
}

// Bitwise / Math Utils from chunk_390
const SIGNIFICAND_WIDTH = 52;
const MAX_UINT32 = 2146435072; // Something related to bit manipulation
const MASK_1 = 1048575;
const EXPONENT_BIAS = 1023;
export const MIN_NORMAL_EXPONENT = -EXPONENT_BIAS + 1;
export const MAX_NORMAL_EXPONENT = EXPONENT_BIAS;
export const MIN_VALUE = Math.pow(2, -1022);

export function getNormalBase2(value: number): number {
    const buffer = new ArrayBuffer(8);
    const view = new DataView(buffer);
    view.setFloat64(0, value);
    // Extract exponent
    return ((view.getUint32(0) & 2146435072) >> 20) - 1023;
}

export function getSignificand(value: number): number {
    const buffer = new ArrayBuffer(8);
    const view = new DataView(buffer);
    view.setFloat64(0, value);
    const hi = view.getUint32(0);
    const lo = view.getUint32(4);
    // Combine to get significand
    return (hi & 1048575) * Math.pow(2, 32) + lo;
}

export function ldexp(value: number, exp: number): number {
    if (value === 0 || value === Number.POSITIVE_INFINITY || value === Number.NEGATIVE_INFINITY || Number.isNaN(value)) return value;
    return value * Math.pow(2, exp);
}

export function nextGreaterSquare(value: number): number {
    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value++;
    return value;
}

export function millisToHrTime(millis: number): [number, number] {
    const seconds = Math.floor(millis / 1000);
    const nanos = (millis % 1000) * 1000000;
    return [seconds, nanos];
}

export function hrTimeToMicroseconds(hrTime: [number, number]): number {
    return hrTime[0] * 1000000 + Math.floor(hrTime[1] / 1000);
}
