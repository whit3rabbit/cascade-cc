/**
 * Expands braces in strings (e.g., "a{b,c}d" -> ["abd", "acd"]).
 */
export function expandBraces(pattern: string): string[] {
    const parts: string[] = [];
    let current = "";
    let depth = 0;

    for (let i = 0; i < pattern.length; i++) {
        const char = pattern[i];
        if (char === "{") {
            depth++;
            current += char;
        } else if (char === "}") {
            depth--;
            current += char;
        } else if (char === "," && depth === 0) {
            const trimmed = current.trim();
            if (trimmed) parts.push(trimmed);
            current = "";
        } else {
            current += char;
        }
    }

    const last = current.trim();
    if (last) parts.push(last);

    return parts.filter(p => p.length > 0).flatMap(p => resolveBraces(p));
}

function resolveBraces(text: string): string[] {
    const match = text.match(/^([^{]*)\{([^}]+)\}(.*)$/);
    if (!match) return [text];

    const prefix = match[1] || "";
    const options = match[2] || "";
    const suffix = match[3] || "";

    const choices = options.split(",").map(opt => opt.trim());
    const results: string[] = [];

    for (const choice of choices) {
        const expanded = prefix + choice + suffix;
        results.push(...resolveBraces(expanded));
    }

    return results;
}
