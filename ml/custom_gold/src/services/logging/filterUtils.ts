import { memoize } from "../../utils/shared/lodashLikeRuntimeAndEnv.js";

/**
 * Writes a string to stdout in chunks of 2000 characters.
 */
export function writeToStdout(content: string): void {
    for (let i = 0; i < content.length; i += 2000) {
        process.stdout.write(content.substring(i, i + 2000));
    }
}

/**
 * Writes a string to stderr in chunks of 2000 characters.
 */
export function writeToStderr(content: string): void {
    for (let i = 0; i < content.length; i += 2000) {
        process.stderr.write(content.substring(i, i + 2000));
    }
}

/**
 * Extracts MCP server name and other metadata from a log line.
 */
export function getMcpServerNameFromLog(logLine: string): string[] {
    const parts: string[] = [];
    const mcpMatch = logLine.match(/^MCP server ["']([^"']+)["']/);
    if (mcpMatch && mcpMatch[1]) {
        parts.push("mcp");
        parts.push(mcpMatch[1].toLowerCase());
    } else {
        const prefixMatch = logLine.match(/^([^:[]+):/);
        if (prefixMatch && prefixMatch[1]) {
            parts.push(prefixMatch[1].trim().toLowerCase());
        }
    }

    const tagMatch = logLine.match(/^\[([^\]]+)]/);
    if (tagMatch && tagMatch[1]) {
        parts.push(tagMatch[1].trim().toLowerCase());
    }

    if (logLine.toLowerCase().includes("statsig event:")) {
        parts.push("statsig");
    }

    const detailMatch = logLine.match(/:\s*([^:]+?)(?:\s+(?:type|mode|status|event))?:/);
    if (detailMatch && detailMatch[1]) {
        const detail = detailMatch[1].trim().toLowerCase();
        if (detail.length < 30 && !detail.includes(" ")) {
            parts.push(detail);
        }
    }

    return Array.from(new Set(parts));
}

/**
 * Determines if a log message should be included based on filters.
 */
export function shouldIncludeLog(tags: string[], filter: LogFilter | null): boolean {
    if (!filter) return true;
    if (tags.length === 0) return false;
    if (filter.isExclusive) {
        return !tags.some((tag) => filter.exclude.includes(tag));
    } else {
        return tags.some((tag) => filter.include.includes(tag));
    }
}

/**
 * Filters a log line based on a filter configuration.
 */
export function filterLogLine(logLine: string, filter: LogFilter | null): boolean {
    if (!filter) return true;
    const tags = getMcpServerNameFromLog(logLine);
    return shouldIncludeLog(tags, filter);
}

export interface LogFilter {
    include: string[];
    exclude: string[];
    isExclusive: boolean;
}

/**
 * Parses a filter string (comma-separated) into a LogFilter object.
 * Filters starting with '!' are treated as exclusions.
 */
export const parseFilterString: (filterStr: string) => LogFilter | null = memoize((filterStr: string): LogFilter | null => {
    if (!filterStr || filterStr.trim() === "") return null;
    const parts = filterStr.split(",").map((p) => p.trim()).filter(Boolean);
    if (parts.length === 0) return null;

    const hasExclusions = parts.some((p) => p.startsWith("!"));
    const hasInclusions = parts.some((p) => !p.startsWith("!"));

    // Cannot mix inclusions and exclusions in the same filter string
    if (hasExclusions && hasInclusions) return null;

    const tags = parts.map((p) => p.replace(/^!/, "").toLowerCase());
    return {
        include: hasExclusions ? [] : tags,
        exclude: hasExclusions ? tags : [],
        isExclusive: hasExclusions
    };
});
