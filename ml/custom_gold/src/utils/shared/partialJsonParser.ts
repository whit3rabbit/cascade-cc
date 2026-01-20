/**
 * Tokenizer for JSON-like strings.
 * Deobfuscated from Z63 in chunk_224.ts.
 */
function tokenize(input: string) {
    let pos = 0;
    const tokens: any[] = [];

    while (pos < input.length) {
        let char = input[pos];

        // Escaped characters
        if (char === "\\") {
            pos++;
            continue;
        }

        // Structure
        if (char === "{") { tokens.push({ type: "brace", value: "{" }); pos++; continue; }
        if (char === "}") { tokens.push({ type: "brace", value: "}" }); pos++; continue; }
        if (char === "[") { tokens.push({ type: "paren", value: "[" }); pos++; continue; }
        if (char === "]") { tokens.push({ type: "paren", value: "]" }); pos++; continue; }
        if (char === ":") { tokens.push({ type: "separator", value: ":" }); pos++; continue; }
        if (char === ",") { tokens.push({ type: "delimiter", value: "," }); pos++; continue; }

        // Strings
        if (char === '"') {
            let value = "";
            let incomplete = false;
            pos++; // skip opening quote
            while (pos < input.length && input[pos] !== '"') {
                let current = input[pos];
                if (current === "\\") {
                    if (pos + 1 >= input.length) { incomplete = true; break; }
                    value += current + input[pos + 1];
                    pos += 2;
                } else {
                    value += current;
                    pos++;
                }
            }
            if (pos < input.length) pos++; // skip closing quote
            else incomplete = true;

            if (!incomplete) {
                tokens.push({ type: "string", value });
            }
            continue;
        }

        // Whitespace
        if (char && /\s/.test(char)) {
            pos++;
            continue;
        }

        // Numbers
        if ((char >= "0" && char <= "9") || char === "-" || char === ".") {
            let value = "";
            if (char === "-") { value += char; pos++; }
            while (pos < input.length && ((input[pos] >= "0" && input[pos] <= "9") || input[pos] === ".")) {
                value += input[pos];
                pos++;
            }
            tokens.push({ type: "number", value });
            continue;
        }

        // Booleans and Null
        if (/[a-z]/i.test(char)) {
            let value = "";
            while (pos < input.length && /[a-z]/i.test(input[pos])) {
                value += input[pos];
                pos++;
            }
            if (value === "true" || value === "false" || value === "null") {
                tokens.push({ type: "name", value });
            }
            continue;
        }

        pos++;
    }
    return tokens;
}

/**
 * Trims incomplete tokens from the end.
 * Deobfuscated from DJA in chunk_224.ts.
 */
function trimTrailingIncomplete(tokens: any[]) {
    if (tokens.length === 0) return tokens;
    const last = tokens[tokens.length - 1];

    switch (last.type) {
        case "separator":
        case "delimiter":
            return trimTrailingIncomplete(tokens.slice(0, -1));
        case "number":
            const lastChar = last.value[last.value.length - 1];
            if (lastChar === "." || lastChar === "-") {
                return trimTrailingIncomplete(tokens.slice(0, -1));
            }
            break;
        case "string":
            const prev = tokens[tokens.length - 2];
            if (prev?.type === "delimiter" || (prev?.type === "brace" && prev.value === "{")) {
                return trimTrailingIncomplete(tokens.slice(0, -1));
            }
            break;
    }
    return tokens;
}

/**
 * Balance delimiters by adding missing closing braces/brackets.
 * Deobfuscated from Y63 in chunk_224.ts.
 */
function balanceDelimiters(tokens: any[]) {
    const stack: string[] = [];
    for (const token of tokens) {
        if (token.type === "brace") {
            if (token.value === "{") stack.push("}");
            else {
                const idx = stack.lastIndexOf("}");
                if (idx !== -1) stack.splice(idx, 1);
            }
        }
        if (token.type === "paren") {
            if (token.value === "[") stack.push("]");
            else {
                const idx = stack.lastIndexOf("]");
                if (idx !== -1) stack.splice(idx, 1);
            }
        }
    }

    const balanced = [...tokens];
    while (stack.length > 0) {
        const closing = stack.pop()!;
        if (closing === "}") balanced.push({ type: "brace", value: "}" });
        else balanced.push({ type: "paren", value: "]" });
    }
    return balanced;
}

/**
 * Re-stringifies tokens into valid JSON.
 * Deobfuscated from J63 in chunk_224.ts.
 */
function stringifyTokens(tokens: any[]): string {
    let output = "";
    for (const token of tokens) {
        if (token.type === "string") {
            output += '"' + token.value + '"';
        } else {
            output += token.value;
        }
    }
    return output;
}

/**
 * Robustly parses a partial or streaming JSON string.
 * Deobfuscated from aA1 in chunk_224.ts.
 */
export function parsePartialJson(input: string): any {
    try {
        const tokens = tokenize(input);
        const trimmed = trimTrailingIncomplete(tokens);
        const balanced = balanceDelimiters(trimmed);
        const jsonStr = stringifyTokens(balanced);
        return JSON.parse(jsonStr);
    } catch (err) {
        // Final fallback: try standard parse if input is very simple
        try {
            return JSON.parse(input);
        } catch {
            throw new Error(`Failed to parse partial JSON: ${err}`);
        }
    }
}
