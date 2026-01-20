import { createRequire } from "node:module";

const nativeCreate: any = Object.create;
const {
    getPrototypeOf: nativeGetPrototypeOf,
    defineProperty: nativeDefineProperty,
    getOwnPropertyNames: nativeGetOwnPropertyNames
} = Object;
const nativeHasOwnProperty = Object.prototype.hasOwnProperty;

/**
 * Helper to load modules (effectively a require wrapper for ESM)
 */
export const loadModule = (module: any, exports?: any, result?: any) => {
    result = module != null ? nativeCreate(nativeGetPrototypeOf(module)) : {};
    let defaultExport = exports || !module || !module.__esModule ? nativeDefineProperty(result, "default", {
        value: module,
        enumerable: true
    }) : result;
    for (let key of nativeGetOwnPropertyNames(module))
        if (!nativeHasOwnProperty.call(defaultExport, key)) nativeDefineProperty(defaultExport, key, {
            get: () => module[key],
            enumerable: true
        });
    return defaultExport;
};

/**
 * Module definition helper
 */
export const defineModule = (fn: (exports: any, module: any) => void) => {
    let module: any;
    return () => (module || fn((module = { exports: {} }).exports, module), module.exports);
};

/**
 * Exports definition helper
 */
export const defineExports = (target: any, source: any) => {
    for (var key in source) nativeDefineProperty(target, key, {
        get: source[key],
        enumerable: true,
        configurable: true,
        set: (value) => (source[key] = () => value)
    });
};

/**
 * Lazy initialization helper
 */
export const lazyInit = (fn: () => any) => {
    let result: any;
    return () => (fn && (result = fn(), (fn as any) = 0), result);
};

// const _require = createRequire(import.meta.url);

let globalReference: any;
const initGlobal = () => {
    globalReference = typeof global == "object" && global && global.Object === Object && global;
};

let root: any;
const initRoot = () => {
    initGlobal();
    const rootSelf = typeof self == "object" && self && self.Object === Object && self;
    root = globalReference || rootSelf || Function("return this")();
};

let _Symbol: any;
const initSymbol = () => {
    initRoot();
    _Symbol = root.Symbol;
};

// Initialize them once at the top level
initGlobal();
initRoot();
initSymbol();


const nativeObjectToString = Object.prototype.toString;

function getRawTag(value: any) {
    var isOwn = nativeHasOwnProperty.call(value, Symbol_toStringTag),
        tag = value[Symbol_toStringTag];
    var unmasked = false;
    try {
        value[Symbol_toStringTag] = undefined;
        unmasked = true;
    } catch (e) { }
    var result = nativeObjectToString.call(value);
    if (unmasked) {
        if (isOwn) value[Symbol_toStringTag] = tag;
        else delete value[Symbol_toStringTag];
    }
    return result;
}

let Symbol_toStringTag: any;
const initGetRawTag = lazyInit(() => {
    initSymbol();
    Symbol_toStringTag = _Symbol ? _Symbol.toStringTag : undefined;
});

function objectToString(value: any) {
    return nativeObjectToString.call(value);
}

const initObjectToString = lazyInit(() => {
    // In the original, this initialized Mz9/Oz9
    return true;
});

const NULL_TAG = "[object Null]";
const UNDEFINED_TAG = "[object Undefined]";

function getTag(value: any) {
    if (value == null) return value === undefined ? UNDEFINED_TAG : NULL_TAG;
    return Symbol_toStringTag && Symbol_toStringTag in Object(value) ? getRawTag(value) : objectToString(value);
}

const initGetTag = lazyInit(() => {
    initSymbol();
    initGetRawTag();
    initObjectToString();
    Symbol_toStringTag = _Symbol ? _Symbol.toStringTag : undefined;
});

function isObject(value: any) {
    var type = typeof value;
    return value != null && (type == "object" || type == "function");
}

const ASYNC_TAG = "[object AsyncFunction]";
const FUNC_TAG = "[object Function]";
const GEN_TAG = "[object GeneratorFunction]";
const PROXY_TAG = "[object Proxy]";

function isFunction(value: any) {
    if (!isObject(value)) return false;
    var tag = getTag(value);
    return tag == FUNC_TAG || tag == GEN_TAG || tag == ASYNC_TAG || tag == PROXY_TAG;
}

let coreJsShared: any;
const initCoreJsShared = lazyInit(() => {
    initRoot();
    coreJsShared = root["__core-js_shared__"];
});

const maskSrcKey = (function () {
    const uid = /[^.]+$/.exec(coreJsShared && coreJsShared.keys && coreJsShared.keys.IE_PROTO || "");
    return uid ? "Symbol(src)_1." + uid : "";
})();

function isMasked(func: any) {
    return !!maskSrcKey && maskSrcKey in func;
}

const nativeFunctionToString = Function.prototype.toString;

function toSource(func: any) {
    if (func != null) {
        try {
            return nativeFunctionToString.call(func);
        } catch (e) { }
        try {
            return func + "";
        } catch (e) { }
    }
    return "";
}

const reRegExpChar = /[\\^$.*+?()[\]{}|]/g;
const reIsNative = RegExp("^" +
    nativeFunctionToString.call(nativeHasOwnProperty)
        .replace(reRegExpChar, "\\$&")
        .replace(/hasOwnProperty|(function).*?(?=\\\()| for .+?(?=\\\])/g, "$1.*?") + "$");

function isNative(value: any) {
    if (!isObject(value) || isMasked(value)) return false;
    var pattern = isFunction(value) ? reIsNative : /^\[object .+?Constructor\]$/;
    return pattern.test(toSource(value));
}

function getValue(object: any, key: string) {
    return object == null ? undefined : object[key];
}

function getNative(object: any, key: string) {
    var value = getValue(object, key);
    return isNative(value) ? value : undefined;
}

const nativeMap = getNative(root, "Map");

const HASH_UNDEFINED = "__lodash_hash_undefined__";

class Hash {
    __data__!: any;
    size!: number;
    constructor(entries?: any[]) {
        this.clear();
        if (entries) {
            let index = -1;
            const length = entries.length;
            while (++index < length) {
                const entry = entries[index];
                this.set(entry[0], entry[1]);
            }
        }
    }

    clear() {
        this.__data__ = nativeCreate ? nativeCreate(null) : {};
        this.size = 0;
    }

    delete(key: string) {
        const result = this.has(key) && delete this.__data__[key];
        this.size -= result ? 1 : 0;
        return result;
    }

    get(key: string) {
        const data = this.__data__;
        if (nativeCreate) {
            const result = data[key];
            return result === HASH_UNDEFINED ? undefined : result;
        }
        return nativeHasOwnProperty.call(data, key) ? data[key] : undefined;
    }

    has(key: string) {
        const data = this.__data__;
        return nativeCreate ? data[key] !== undefined : nativeHasOwnProperty.call(data, key);
    }

    set(key: string, value: any) {
        const data = this.__data__;
        this.size += this.has(key) ? 0 : 1;
        data[key] = nativeCreate && value === undefined ? HASH_UNDEFINED : value;
        return this;
    }
}

function eq(value: any, other: any) {
    return value === other || (value !== value && other !== other);
}

function assocGet(array: any[], key: any) {
    let length = array.length;
    while (length--) {
        if (eq(array[length][0], key)) return length;
    }
    return -1;
}

class ListCache {
    __data__: any[] = [];
    size: number = 0;
    constructor(entries?: any[]) {
        this.clear();
        if (entries) {
            let index = -1;
            const length = entries.length;
            while (++index < length) {
                const entry = entries[index];
                this.set(entry[0], entry[1]);
            }
        }
    }

    clear() {
        this.__data__ = [];
        this.size = 0;
    }

    delete(key: any) {
        const data = this.__data__;
        const index = assocGet(data, key);
        if (index < 0) return false;
        const lastIndex = data.length - 1;
        if (index == lastIndex) data.pop();
        else Array.prototype.splice.call(data, index, 1);
        --this.size;
        return true;
    }

    get(key: any) {
        const data = this.__data__;
        const index = assocGet(data, key);
        return index < 0 ? undefined : data[index][1];
    }

    has(key: any) {
        return assocGet(this.__data__, key) > -1;
    }

    set(key: any, value: any) {
        const data = this.__data__;
        const index = assocGet(data, key);
        if (index < 0) {
            ++this.size;
            data.push([key, value]);
        } else {
            data[index][1] = value;
        }
        return this;
    }
}

function isKeyable(value: any) {
    const type = typeof value;
    return (type == "string" || type == "number" || type == "symbol" || type == "boolean")
        ? (value !== "__proto__")
        : (value === null);
}

function getMapData(cache: any, key: any) {
    const data = cache.__data__;
    return isKeyable(key) ? data[typeof key == "string" ? "string" : "hash"] : data.map;
}

class MapCache {
    __data__!: {
        hash: Hash;
        map: Map<any, any> | ListCache;
        string: Hash;
    };
    size!: number;
    constructor(entries?: any[]) {
        this.clear();
        if (entries) {
            let index = -1;
            const length = entries.length;
            while (++index < length) {
                const entry = entries[index];
                this.set(entry[0], entry[1]);
            }
        }
    }

    clear() {
        this.size = 0;
        this.__data__ = {
            hash: new Hash(),
            map: new (nativeMap || ListCache)(),
            string: new Hash()
        };
    }

    delete(key: any) {
        const result = getMapData(this, key).delete(key);
        this.size -= result ? 1 : 0;
        return result;
    }

    get(key: any) {
        return getMapData(this, key).get(key);
    }

    has(key: any) {
        return getMapData(this, key).has(key);
    }

    set(key: any, value: any) {
        const data = getMapData(this, key);
        const size = data.size;
        data.set(key, value);
        this.size += data.size == size ? 0 : 1;
        return this;
    }
}

export function memoize<T extends (...args: any[]) => any>(func: T, resolver?: (...args: Parameters<T>) => any): T & { cache: MapCache } {
    if (typeof func != "function" || (resolver != null && typeof resolver != "function")) {
        throw new TypeError("Expected a function");
    }
    const memoized = function (this: any, ...args: Parameters<T>) {
        const key = resolver ? resolver.apply(this, args) : args[0];
        const cache = memoized.cache;
        if (cache.has(key)) return cache.get(key);
        const result = func.apply(this, args);
        memoized.cache = cache.set(key, result) || cache;
        return result;
    };
    memoized.cache = new (memoize.Cache || MapCache)();
    return memoized as any;
}
memoize.Cache = MapCache;

export const noop = () => { };

class Stack {
    __data__: ListCache | MapCache;
    size: number = 0;
    constructor(entries?: any[]) {
        this.__data__ = new ListCache(entries);
        this.size = this.__data__.size;
    }

    clear() {
        this.__data__ = new ListCache();
        this.size = 0;
    }

    delete(key: any) {
        const data = this.__data__;
        const result = data.delete(key);
        this.size = data.size;
        return result;
    }

    get(key: any) {
        return this.__data__.get(key);
    }

    has(key: any) {
        return this.__data__.has(key);
    }

    set(key: any, value: any) {
        let data = this.__data__;
        if (data instanceof ListCache) {
            const pairs = data.__data__;
            if (!nativeMap || pairs.length < 199) {
                pairs.push([key, value]);
                this.size = ++data.size;
                return this;
            }
            data = this.__data__ = new MapCache(pairs);
        }
        data.set(key, value);
        this.size = data.size;
        return this;
    }
}

class SetCache {
    __data__: MapCache;
    constructor(values?: any[]) {
        let index = -1;
        const length = values == null ? 0 : values.length;
        this.__data__ = new MapCache();
        while (++index < length) {
            if (values) this.add(values[index]);
        }
    }

    add(value: any) {
        this.__data__.set(value, HASH_UNDEFINED);
        return this;
    }

    push(value: any) {
        return this.add(value);
    }

    has(value: any) {
        return this.__data__.has(value);
    }
}
// (SetCache.prototype as any).push = SetCache.prototype.add;

function arraySome(array: any[], predicate: Function) {
    let index = -1;
    const length = array == null ? 0 : array.length;
    while (++index < length) {
        if (predicate(array[index], index, array)) return true;
    }
    return false;
}

function cacheHas(cache: any, key: any) {
    return cache.has(key);
}

function equalArrays(array: any[], other: any[], bitmask: number, customizer: any, compareFunc: any, stack: any) {
    const isPartial = bitmask & 1;
    const arrLength = array.length;
    const othLength = other.length;
    if (arrLength != othLength && !(isPartial && othLength > arrLength)) return false;
    const stackedArr = stack.get(array);
    const stackedOth = stack.get(other);
    if (stackedArr && stackedOth) return stackedArr == other && stackedOth == array;
    let index = -1,
        result = true,
        seen = bitmask & 2 ? new SetCache() : undefined;
    stack.set(array, other);
    stack.set(other, array);
    while (++index < arrLength) {
        const arrValue = array[index],
            othValue = other[index];
        let compared: any;
        if (customizer) compared = isPartial ? customizer(othValue, arrValue, index, other, array, stack) : customizer(arrValue, othValue, index, array, other, stack);
        if (compared !== undefined) {
            if (compared) continue;
            result = false;
            break;
        }
        if (seen) {
            if (!arraySome(other, function (othValue: any, index: number) {
                if (!cacheHas(seen, index) && (arrValue === othValue || compareFunc(arrValue, othValue, bitmask, customizer, stack))) return seen.push(index);
            })) {
                result = false;
                break;
            }
        } else if (!(arrValue === othValue || compareFunc(arrValue, othValue, bitmask, customizer, stack))) {
            result = false;
            break;
        }
    }
    stack.delete(array);
    stack.delete(other);
    return result;
}

const nativeUint8Array = root.Uint8Array;

/**
 * Truncates a string to a specific length, optionally handling newlines.
 */
export function truncateString(str: string, length: number, handleNewlines = false): string {
    let result = str;
    if (handleNewlines) {
        const newlineIndex = str.indexOf("\n");
        if (newlineIndex !== -1) {
            result = str.substring(0, newlineIndex);
            if (result.length + 1 > length) return `${result.substring(0, length - 1)}…`;
            return `${result}…`;
        }
    }
    if (result.length <= length) return result;
    return `${result.substring(0, length - 1)}…`;
}

/**
 * Formats a duration in milliseconds to a human-readable string.
 */
export function formatDuration(ms: number): string {
    if (ms < 60000) {
        if (ms === 0) return "0s";
        if (ms < 1) return `${(ms / 1000).toFixed(1)}s`;
        return `${Math.round(ms / 1000).toString()}s`;
    }
    let days = Math.floor(ms / 86400000),
        hours = Math.floor(ms % 86400000 / 3600000),
        minutes = Math.floor(ms % 3600000 / 60000),
        seconds = Math.round(ms % 60000 / 1000);
    if (seconds === 60) { seconds = 0; minutes++; }
    if (minutes === 60) { minutes = 0; hours++; }
    if (hours === 24) { hours = 0; days++; }
    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m ${seconds}s`;
    if (minutes > 0) return `${minutes}m ${seconds}s`;
    return `${seconds}s`;
}

/**
 * Formats a number into a compact representation (e.g., 1.5k, 2m).
 */
export function formatCompactNumber(num: number): string {
    const isCompact = num >= 1000;
    return new Intl.NumberFormat("en", {
        notation: "compact",
        minimumFractionDigits: isCompact ? 1 : 0,
        maximumFractionDigits: 1
    }).format(num).toLowerCase();
}

/**
 * Formats a relative time string.
 */
export function formatRelativeTime(date: Date, options: { style?: "narrow" | "short" | "long"; numeric?: "always" | "auto"; now?: Date } = {}): string {
    const {
        style = "narrow",
        numeric = "always",
        now = new Date()
    } = options;
    const deltaSeconds = Math.trunc((date.getTime() - now.getTime()) / 1000);
    const units: { unit: Intl.RelativeTimeFormatUnit; seconds: number; shortUnit: string }[] = [
        { unit: "year", seconds: 31536000, shortUnit: "y" },
        { unit: "month", seconds: 2592000, shortUnit: "mo" },
        { unit: "week", seconds: 604800, shortUnit: "w" },
        { unit: "day", seconds: 86400, shortUnit: "d" },
        { unit: "hour", seconds: 3600, shortUnit: "h" },
        { unit: "minute", seconds: 60, shortUnit: "m" },
        { unit: "second", seconds: 1, shortUnit: "s" }
    ];

    for (const { unit, seconds, shortUnit } of units) {
        if (Math.abs(deltaSeconds) >= seconds) {
            const count = Math.trunc(deltaSeconds / seconds);
            if (style === "narrow") {
                return deltaSeconds < 0 ? `${Math.abs(count)}${shortUnit} ago` : `in ${count}${shortUnit}`;
            }
            return new Intl.RelativeTimeFormat("en", {
                style,
                numeric
            }).format(count, unit);
        }
    }

    if (style === "narrow") return deltaSeconds <= 0 ? "0s ago" : "in 0s";
    return new Intl.RelativeTimeFormat("en", {
        style,
        numeric
    }).format(0, "second");
}

/**
 * Formats a relative time string, defaulting to 'always' numeric mode if in the past.
 */
export function formatRelativeTimeAlways(date: Date, options: { style?: "narrow" | "short" | "long"; now?: Date } = {}): string {
    const { now = new Date(), ...rest } = options;
    if (date > now) return formatRelativeTime(date, { ...rest, now });
    return formatRelativeTime(date, { ...rest, numeric: "always", now });
}

/**
 * Formats a unix timestamp into a human-readable date/time string.
 */
export function formatTimestamp(timestampSeconds: number, includeTimeZone = false, includeTime = true): string {
    if (!timestampSeconds) return "";
    const date = new Date(timestampSeconds * 1000);
    const now = new Date();
    const minutes = date.getMinutes();
    const isFarFutureOrPast = Math.abs(date.getTime() - now.getTime()) / 3600000 > 24;

    if (isFarFutureOrPast) {
        const formatOptions: Intl.DateTimeFormatOptions = {
            month: "short",
            day: "numeric",
            hour: includeTime ? "numeric" : undefined,
            minute: !includeTime || minutes === 0 ? undefined : "2-digit",
            hour12: includeTime ? true : undefined
        };
        if (date.getFullYear() !== now.getFullYear()) {
            formatOptions.year = "numeric";
        }
        let result = date.toLocaleString("en-US", formatOptions).replace(/ ([AP]M)/i, (_, p1) => p1.toLowerCase());
        if (includeTimeZone) {
            result += ` (${Intl.DateTimeFormat().resolvedOptions().timeZone})`;
        }
        return result;
    }

    const timeStr = date.toLocaleTimeString("en-US", {
        hour: "numeric",
        minute: minutes === 0 ? undefined : "2-digit",
        hour12: true
    });
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    let result = timeStr.replace(/ ([AP]M)/i, (_, p1) => p1.toLowerCase());
    if (includeTimeZone) {
        result += ` (${tz})`;
    }
    return result;
}

/**
 * Formats a Date object into a human-readable string.
 */
export function formatDate(date: Date, includeTimeZone = false, includeTime = true): string {
    return formatTimestamp(Math.floor(date.getTime() / 1000), includeTimeZone, includeTime);
}

/**
 * Throttling wrapper for async functions to ensure sequential execution.
 */
export function batchPromise<T extends (...args: any[]) => Promise<any>>(fn: T): T {
    const queue: { args: any[], resolve: (val: any) => void, reject: (err: any) => void, context: any }[] = [];
    let isRunning = false;

    async function processQueue() {
        if (isRunning || queue.length === 0) return;
        isRunning = true;
        while (queue.length > 0) {
            const { args, resolve, reject, context } = queue.shift()!;
            try {
                const result = await fn.apply(context, args);
                resolve(result);
            } catch (err) {
                reject(err);
            }
        }
        isRunning = false;
        if (queue.length > 0) processQueue();
    }

    return function (this: any, ...args: any[]) {
        return new Promise((resolve, reject) => {
            queue.push({ args, resolve, reject, context: this });
            processQueue();
        });
    } as T;
}


export {
    Hash,
    ListCache,
    MapCache,
    SetCache,
    Stack,
    getTag,
    isObject,
    isFunction,
    isNative,
    getNative,
    eq,
    arraySome,
    equalArrays,
    nativeUint8Array as Uint8Array
};

