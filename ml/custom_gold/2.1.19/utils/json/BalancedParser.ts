/**
 * File: src/utils/json/BalancedParser.ts
 * Role: Permissive JSON parser that balances brackets and cleans up trailing commas. Useful for LLM streaming outputs.
 */

export interface Token {
    type: "brace" | "paren" | "separator" | "delimiter" | "string" | "number" | "name";
    value: string;
}

export function tokenize(input: string): Token[] {
    const tokens: Token[] = [];
    let i = 0;
    while (i < input.length) {
        const char = input[i];
        if (char === "\\") {
            i += 2;
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
            i++;
            while (i < input.length && input[i] !== '"') {
                if (input[i] === "\\") {
                    str += input[i + 1] || "";
                    i += 2;
                } else {
                    str += input[i];
                    i++;
                }
            }
            i++;
            tokens.push({ type: "string", value: str });
            continue;
        }

        if (/\s/.test(char)) {
            i++;
            continue;
        }

        const numMatch = input.slice(i).match(/^-?[0-9]*\.?[0-9]+/);
        if (numMatch) {
            tokens.push({ type: "number", value: numMatch[0] });
            i += numMatch[0].length;
            continue;
        }

        const nameMatch = input.slice(i).match(/^(true|false|null)/);
        if (nameMatch) {
            tokens.push({ type: "name", value: nameMatch[0] });
            i += nameMatch[0].length;
            continue;
        }
        i++;
    }
    return tokens;
}

export function cleanupTokens(tokens: Token[]): Token[] {
    if (tokens.length === 0) return tokens;
    const last = tokens[tokens.length - 1];
    if (last.type === "separator" || last.type === "delimiter") {
        return cleanupTokens(tokens.slice(0, -1));
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

    stack.reverse().forEach((bracket) => {
        tokens.push({ type: bracket === "}" ? "brace" : "paren", value: bracket });
    });
    return tokens;
}

export function stringifyTokens(tokens: Token[]): string {
    return tokens.map((t) => (t.type === "string" ? `"${t.value}"` : t.value)).join("");
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
            return JSON.parse(stringifyTokens(balanceBrackets(cleanupTokens(tokenize(input)))));
        } catch {
            return null;
        }
    }
}
