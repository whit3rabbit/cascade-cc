import equal from 'fast-deep-equal';

export function copy(obj: any, target: any = {}): any {
    for (const key in obj) target[key] = obj[key];
    return target;
}

export function checkDataType(type: string, data: string, strictNumbers?: boolean, isNot?: boolean): string {
    const eq = isNot ? ' !== ' : ' === ';
    const op = isNot ? ' || ' : ' && ';
    const not = isNot ? '!' : '';
    const maybeNot = isNot ? '' : '!';

    switch (type) {
        case 'null':
            return data + eq + 'null';
        case 'array':
            return not + 'Array.isArray(' + data + ')';
        case 'object':
            return '(' + not + data + op + 'typeof ' + data + eq + '"object"' + op + maybeNot + 'Array.isArray(' + data + '))';
        case 'integer':
            return '(typeof ' + data + eq + '"number"' + op + maybeNot + '(' + data + ' % 1)' + op + data + eq + data + (strictNumbers ? op + not + 'isFinite(' + data + ')' : '') + ')';
        case 'number':
            return '(typeof ' + data + eq + '"number"' + (strictNumbers ? op + not + 'isFinite(' + data + ')' : '') + ')';
        default:
            return 'typeof ' + data + eq + '"' + type + '"';
    }
}

export function checkDataTypes(types: string[], data: string, strictNumbers?: boolean): string {
    if (types.length === 1) {
        return checkDataType(types[0], data, strictNumbers, true);
    }

    let code = '';
    const typesHash = toHash(types);
    if (typesHash.array && typesHash.object) {
        code = typesHash.null ? '(' : '(!' + data + ' || ';
        code += 'typeof ' + data + ' !== "object")';
        delete typesHash.null;
        delete typesHash.array;
        delete typesHash.object;
    }
    if (typesHash.number) delete typesHash.integer;
    for (const type in typesHash) {
        code += (code ? ' && ' : '') + checkDataType(type, data, strictNumbers, true);
    }
    return code;
}

const BASIC_TYPES = toHash(['string', 'number', 'integer', 'boolean', 'null']);

export function coerceToTypes(option: any, type: string | string[]): string[] | undefined {
    if (Array.isArray(type)) {
        const res: string[] = [];
        for (const t of type) {
            if (BASIC_TYPES[t] || (option === 'array' && t === 'array')) {
                res.push(t);
            }
        }
        if (res.length) return res;
    } else if (BASIC_TYPES[type] || (option === 'array' && type === 'array')) {
        return [type as string];
    }
}

export function toHash(arr: string[]): { [key: string]: boolean } {
    const hash: { [key: string]: boolean } = {};
    for (const item of arr) hash[item] = true;
    return hash;
}

const IDENTIFIER = /^[a-z$_][a-z$_0-9]*$/i;
const SINGLE_QUOTE = /'|\\/g;

export function getProperty(key: string | number): string {
    return typeof key === 'number'
        ? '[' + key + ']'
        : IDENTIFIER.test(key)
            ? '.' + key
            : "['" + escapeQuotes(key) + "']";
}

export function escapeQuotes(str: string): string {
    return str
        .replace(SINGLE_QUOTE, '\\$&')
        .replace(/\n/g, '\\n')
        .replace(/\r/g, '\\r')
        .replace(/\f/g, '\\f')
        .replace(/\t/g, '\\t');
}

export function varOccurences(str: string, prefix: string): number {
    prefix += '[^0-9]';
    const matches = str.match(new RegExp(prefix, 'g'));
    return matches ? matches.length : 0;
}

export function varReplace(str: string, prefix: string, replacement: string): string {
    prefix += '([^0-9])';
    replacement = replacement.replace(/\$/g, '$$$$');
    return str.replace(new RegExp(prefix, 'g'), replacement + '$1');
}

export function schemaHasRules(schema: any, rules: any): boolean {
    if (typeof schema === 'boolean') return !schema;
    for (const key in schema) if (rules[key]) return true;
    return false;
}

export function schemaHasRulesExcept(schema: any, rules: any, except: string): boolean {
    if (typeof schema === 'boolean') return !schema && except !== 'not';
    for (const key in schema) if (key !== except && rules[key]) return true;
    return false;
}

export function schemaUnknownRules(schema: any, rules: any): string | undefined {
    if (typeof schema === 'boolean') return;
    for (const key in schema) if (!rules[key]) return key;
}

export function toQuotedString(str: string): string {
    return "'" + escapeQuotes(str) + "'";
}

export function getPathExpr(currentPath: string, expr: string, jsonPointers?: boolean, isIndex?: boolean): string {
    const path = jsonPointers
        ? "'/' + " + expr + (isIndex ? '' : ".replace(/~/g, '~0').replace(/\\//g, '~1')")
        : isIndex ? "'[' + " + expr + " + ']'" : "'[\\'' + " + expr + " + '\\']'";
    return joinPaths(currentPath, path);
}

export function getPath(currentPath: string, key: string | number, jsonPointers?: boolean): string {
    const path = jsonPointers ? toQuotedString('/' + escapeFragment(String(key))) : toQuotedString(getProperty(key));
    return joinPaths(currentPath, path);
}

const JSON_POINTER = /^\/(?:[^~]|~0|~1)*$/;
const RELATIVE_JSON_POINTER = /^([0-9]+)(#|\/(?:[^~]|~0|~1)*)?$/;

export function getData(ptr: string, dataLevel: number, dataPaths: string[]): string {
    let up, jsonPtr, baseData;
    if (ptr === '') return 'rootData';
    if (ptr[0] === '/') {
        if (!JSON_POINTER.test(ptr)) throw new Error('Invalid JSON-pointer: ' + ptr);
        jsonPtr = ptr;
        baseData = 'rootData';
    } else {
        const matches = ptr.match(RELATIVE_JSON_POINTER);
        if (!matches) throw new Error('Invalid JSON-pointer: ' + ptr);
        up = +matches[1];
        jsonPtr = matches[2];
        if (jsonPtr === '#') {
            if (up >= dataLevel) throw new Error('Cannot access property/index ' + up + ' levels up, current level is ' + dataLevel);
            return dataPaths[dataLevel - up];
        }
        if (up > dataLevel) throw new Error('Cannot access data ' + up + ' levels up, current level is ' + dataLevel);
        baseData = 'data' + (dataLevel - up || '');
        if (!jsonPtr) return baseData;
    }

    let res = baseData;
    const parts = jsonPtr.split('/');
    for (const part of parts) {
        if (part) {
            baseData += getProperty(unescapeJsonPointer(part));
            res += ' && ' + baseData;
        }
    }
    return res;
}

function joinPaths(a: string, b: string): string {
    if (a === '""') return b;
    return (a + ' + ' + b).replace(/([^\\])' \+ '/g, '$1');
}

export function unescapeFragment(str: string): string {
    return unescapeJsonPointer(decodeURIComponent(str));
}

export function escapeFragment(str: string): string {
    return encodeURIComponent(escapeJsonPointer(str));
}

export function escapeJsonPointer(str: string): string {
    return str.replace(/~/g, '~0').replace(/\//g, '~1');
}

export function unescapeJsonPointer(str: string): string {
    return str.replace(/~1/g, '/').replace(/~0/g, '~');
}

export function ucs2length(str: string): number {
    let length = 0;
    const len = str.length;
    let pos = 0;
    let value;
    while (pos < len) {
        length++;
        value = str.charCodeAt(pos++);
        if (value >= 0xD800 && value <= 0xDBFF && pos < len) {
            value = str.charCodeAt(pos);
            if ((value & 0xFC00) === 0xDC00) pos++;
        }
    }
    return length;
}

export { equal };
