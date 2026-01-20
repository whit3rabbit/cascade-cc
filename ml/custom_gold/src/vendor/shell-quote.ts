/**
 * Deobfuscated from chunk_70.ts (kBQ & dBQ)
 * Functional equivalent of 'shell-quote' package.
 */

export type Token = string | { op: string } | { op: 'glob', pattern: string } | { comment: string };

export function quote(args: (string | { op: string })[]): string {
    return args.map(arg => {
        if (arg === "") return "''";
        if (arg && typeof arg === "object") return arg.op.replace(/(.)/g, "\\$1");
        if (/["\s\\]/.test(arg) && !/'/.test(arg)) return "'" + arg.replace(/(['])/g, "\\$1") + "'";
        if (/["'\s]/.test(arg)) return '"' + arg.replace(/(["\\$`!])/g, "\\$1") + '"';
        return String(arg).replace(/([A-Za-z]:)?([#!"$&'()*,:;<=>?@[\\\]^`{|}])/g, "$1\\$2");
    }).join(" ");
}

const CONTROL_OPS = "(?:" + ["\\|\\|", "\\&\\&", ";;", "\\|\\&", "\\<\\(", "\\<\\<\\<", ">>", ">\\&", "<\\&", "[&;()|<>]"].join("|") + ")";
const CONTROL_RE = new RegExp("^" + CONTROL_OPS + "$");
const META_CHARS = "|&;()<> \t";
const BASH_COMMENT = /^#$/;

export function parse(command: string, env?: Record<string, any> | ((key: string) => any), options: { escape?: string } = {}): Token[] {
    const escapeChar = options.escape || "\\";
    const wordPattern = "(\\" + escapeChar + "['\"" + META_CHARS + "]|[^\\s'\"" + META_CHARS + "])+";
    const doubleQuotePattern = '"((\\\\"|[^"])*?)"';
    const singleQuotePattern = "'((\\\\'|[^'])*?)'";

    const pattern = new RegExp([
        "(" + CONTROL_OPS + ")",
        "(" + wordPattern + "|" + doubleQuotePattern + "|" + singleQuotePattern + ")+"
    ].join("|"), "g");

    const matches = Array.from(command.matchAll(pattern));
    if (matches.length === 0) return [];

    let isComment = false;
    const sentinel = `__SENTINEL_${Math.random().toString(16).slice(2)}__`;
    const sentinelRe = new RegExp("^" + sentinel);

    const formatEnv = (key: string, prefix: string) => {
        let val = typeof env === 'function' ? env(key) : env ? env[key] : undefined;
        if (val === undefined && key !== "") val = "";
        else if (val === undefined) val = "$";

        if (typeof val === 'object') return prefix + sentinel + JSON.stringify(val) + sentinel;
        return prefix + val;
    };

    const tokens = matches.map(match => {
        const tokenStr = match[0];
        if (!tokenStr || isComment) return;

        if (CONTROL_RE.test(tokenStr)) return { op: tokenStr };

        let inQuote: string | false = false;
        let isEscaped = false;
        let content = "";
        let isGlob = false;

        for (let i = 0; i < tokenStr.length; i++) {
            const char = tokenStr.charAt(i);

            if (isGlob || (!inQuote && (char === "*" || char === "?"))) {
                isGlob = true;
            }

            if (isEscaped) {
                content += char;
                isEscaped = false;
            } else if (inQuote) {
                if (char === inQuote) {
                    inQuote = false;
                } else if (inQuote === "'") {
                    content += char;
                } else if (char === escapeChar) {
                    const next = tokenStr.charAt(i + 1);
                    if (next === '"' || next === escapeChar || next === "$") {
                        content += next;
                        i++;
                    } else {
                        content += escapeChar + next;
                        i++;
                    }
                } else if (char === "$") {
                    const result = expandVar(tokenStr, i, env, formatEnv);
                    content += result.value;
                    i = result.index;
                } else {
                    content += char;
                }
            } else if (char === escapeChar) {
                isEscaped = true;
            } else if (char === "$") {
                const result = expandVar(tokenStr, i, env, formatEnv);
                content += result.value;
                i = result.index;
            } else if (char === '"' || char === "'") {
                inQuote = char;
            } else if (CONTROL_RE.test(char)) {
                return { op: tokenStr };
            } else if (BASH_COMMENT.test(char)) {
                isComment = true;
                const comment = command.slice(match.index! + i + 1);
                if (content.length) return [content, { comment }];
                return [{ comment }];
            } else {
                content += char;
            }
        }

        if (isGlob) return { op: "glob", pattern: content };
        return content;
    }).flat().filter(t => t !== undefined) as Token[];

    if (typeof env !== 'function') return tokens;

    return tokens.reduce((acc, token) => {
        if (typeof token === 'object') return acc.concat(token);
        const parts = token.split(new RegExp("(" + sentinel + ".*?" + sentinel + ")", "g"));
        if (parts.length === 1) return acc.concat(parts[0]);
        return acc.concat(parts.filter(Boolean).map(p => {
            if (sentinelRe.test(p)) return JSON.parse(p.split(sentinel)[1]);
            return p;
        }));
    }, [] as Token[]);
}

function expandVar(str: string, index: number, env: any, formatEnv: any) {
    let charIndex = index + 1;
    const char = str.charAt(charIndex);
    let key = "";

    if (char === "{") {
        charIndex++;
        if (str.charAt(charIndex) === "}") throw Error("Bad substitution: " + str.slice(charIndex - 2, charIndex + 1));
        const end = str.indexOf("}", charIndex);
        if (end < 0) throw Error("Bad substitution: " + str.slice(charIndex));
        key = str.slice(charIndex, end);
        charIndex = end;
    } else if (/[*@#?$!_-]/.test(char)) {
        key = char;
        // charIndex remains at index + 1, so we'll increment it at the end
    } else {
        const remaining = str.slice(charIndex);
        const match = remaining.match(/[^\w\d_]/);
        if (!match) {
            key = remaining;
            charIndex = str.length - 1;
        } else {
            key = remaining.slice(0, match.index);
            charIndex += (match.index || 0) - 1;
        }
    }

    return {
        value: formatEnv(key, ""),
        index: charIndex
    };
}
