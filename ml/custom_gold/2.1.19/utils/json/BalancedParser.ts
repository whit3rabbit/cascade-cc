/**
 * File: src/utils/json/BalancedParser.ts
 * Role: Permissive JSON parser that balances brackets and cleans up trailing commas. Useful for LLM streaming outputs.
 */

export interface Token {
    type: "brace" | "paren" | "separator" | "delimiter" | "string" | "number" | "name";
    value: string;
}

export function tokenize(input: string): Token[] {
    let i = 0;
    const tokens: Token[] = [];
    while (i < input.length) {
        const char = input[i];
        if (char === "\\") {
            i++;
            continue;
        }
        if (char === "{") {
            tokens.push({ type: "brace", value: "{" });
            i++;
            continue;
        }
        if (char === "}") {
            tokens.push({ type: "brace", value: "}" });
            i++;
            continue;
        }
        if (char === "[") {
            tokens.push({ type: "paren", value: "[" });
            i++;
            continue;
        }
        if (char === "]") {
            tokens.push({ type: "paren", value: "]" });
            i++;
            continue;
        }
        if (char === ":") {
            tokens.push({ type: "separator", value: ":" });
            i++;
            continue;
        }
        if (char === ",") {
            tokens.push({ type: "delimiter", value: "," });
            i++;
            continue;
        }
        if (char === '"') {
            let str = "";
            let escaped = false;
            let current = input[++i];
            while (current !== '"') {
                if (i === input.length) {
                    escaped = true;
                    break;
                }
                if (current === "\\") {
                    i++;
                    if (i === input.length) {
                        escaped = true;
                        break;
                    }
                    str += current + input[i];
                    current = input[++i];
                } else {
                    str += current;
                    current = input[++i];
                }
            }
            i++;
            if (!escaped) {
                tokens.push({ type: "string", value: str });
            }
            continue;
        }
        if (char && /\s/.test(char)) {
            i++;
            continue;
        }

        const isNum = /[0-9]/;
        if ((char && isNum.test(char)) || char === "-" || char === ".") {
            let num = "";
            let current = char;
            if (current === "-") {
                num += current;
                current = input[++i];
            }
            while (current && (isNum.test(current) || current === ".")) {
                num += current;
                current = input[++i];
            }
            tokens.push({ type: "number", value: num });
            continue;
        }

        const isAlpha = /[a-z]/i;
        if (char && isAlpha.test(char)) {
            let name = "";
            let current = char;
            while (current && isAlpha.test(current)) {
                if (i === input.length) break;
                name += current;
                current = input[++i];
            }
            if (name === "true" || name === "false" || name === "null") {
                tokens.push({ type: "name", value: name });
            } else {
                i++;
                continue;
            }
            continue;
        }
        i++;
    }
    return tokens;
}

export function cleanupTokens(tokens: Token[]): Token[] {
    if (tokens.length === 0) return tokens;
    const last = tokens[tokens.length - 1];
    switch (last.type) {
        case "separator":
            return cleanupTokens(tokens.slice(0, tokens.length - 1));
        case "number": {
            const lastChar = last.value[last.value.length - 1];
            if (lastChar === "." || lastChar === "-") {
                return cleanupTokens(tokens.slice(0, tokens.length - 1));
            }
            break;
        }
        case "string": {
            const secondLast = tokens[tokens.length - 2];
            if (secondLast?.type === "delimiter") {
                return cleanupTokens(tokens.slice(0, tokens.length - 1));
            } else if (secondLast?.type === "brace" && secondLast.value === "{") {
                return cleanupTokens(tokens.slice(0, tokens.length - 1));
            }
            break;
        }
        case "delimiter":
            return cleanupTokens(tokens.slice(0, tokens.length - 1));
    }
    return tokens;
}

export function balanceBrackets(tokens: Token[]): Token[] {
    const stack: string[] = [];
    tokens.forEach((t) => {
        if (t.type === "brace") {
            if (t.value === "{") stack.push("}");
            else {
                const idx = stack.lastIndexOf("}");
                if (idx !== -1) stack.splice(idx, 1);
            }
        }
        if (t.type === "paren") {
            if (t.value === "[") stack.push("]");
            else {
                const idx = stack.lastIndexOf("]");
                if (idx !== -1) stack.splice(idx, 1);
            }
        }
    });

    if (stack.length > 0) {
        stack.reverse().forEach((bracket) => {
            tokens.push({
                type: bracket === "}" ? "brace" : "paren",
                value: bracket,
            });
        });
    }
    return tokens;
}

export function stringifyTokens(tokens: Token[]): string {
    let result = "";
    tokens.forEach((t) => {
        switch (t.type) {
            case "string":
                result += '"' + t.value + '"';
                break;
            default:
                result += t.value;
                break;
        }
    });
    return result;
}

/**
 * Permissive JSON parse that attempts to fix common LLM streaming errors.
 */
export function parseBalancedJSON(input: string): any {
    if (!input || !input.trim()) return null;
    try {
        return JSON.parse(input);
    } catch {
        try {
            const tokens = tokenize(input);
            const cleaned = cleanupTokens(tokens);
            const balanced = balanceBrackets(cleaned);
            const stringified = stringifyTokens(balanced);
            return JSON.parse(stringified);
        } catch {
            return null;
        }
    }
}

