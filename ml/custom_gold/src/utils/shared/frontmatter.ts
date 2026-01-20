/**
 * Parses frontmatter from a markdown string.
 */
export function parseFrontmatter(text: string): { frontmatter: Record<string, string>; content: string } {
    const frontmatterRegex = /^---\s*\n([\s\S]*?)---\s*\n?/;
    const match = text.match(frontmatterRegex);

    if (!match) {
        return {
            frontmatter: {},
            content: text
        };
    }

    const rawFrontmatter = match[1] || "";
    const content = text.slice(match[0].length);
    const frontmatter: Record<string, string> = {};

    const lines = rawFrontmatter.split("\n");
    for (const line of lines) {
        const colonIndex = line.indexOf(":");
        if (colonIndex > 0) {
            const key = line.slice(0, colonIndex).trim();
            const value = line.slice(colonIndex + 1).trim();
            if (key) {
                // Strip surrounding quotes
                const cleanedValue = value.replace(/^["']|["']$/g, "");
                frontmatter[key] = cleanedValue;
            }
        }
    }

    return {
        frontmatter,
        content
    };
}
