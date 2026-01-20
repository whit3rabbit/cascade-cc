
// Logic from chunk_338.ts

export const PERMISSION_CONFIG = {
    filePatternTools: ["Read", "Write", "Edit", "Glob", "NotebookRead", "NotebookEdit"],
    bashPrefixTools: ["Bash"],
    customValidation: {
        WebSearch: (rule: string) => {
            if (rule.includes("*") || rule.includes("?")) return {
                valid: false,
                error: "WebSearch does not support wildcards",
                suggestion: "Use exact search terms without * or ?",
                examples: ["WebSearch(claude ai)", "WebSearch(typescript tutorial)"]
            };
            return { valid: true };
        },
        WebFetch: (rule: string) => {
            if (rule.includes("://") || rule.startsWith("http")) return {
                valid: false,
                error: "WebFetch permissions use domain format, not URLs",
                suggestion: 'Use "domain:hostname" format',
                examples: ["WebFetch(domain:example.com)", "WebFetch(domain:github.com)"]
            };
            if (!rule.startsWith("domain:")) return {
                valid: false,
                error: 'WebFetch permissions must use "domain:" prefix',
                suggestion: 'Use "domain:hostname" format',
                examples: ["WebFetch(domain:example.com)", "WebFetch(domain:*.google.com)"]
            };
            return { valid: true };
        }
    } as Record<string, (rule: string) => PermissionValidationResult>
};

export interface PermissionValidationResult {
    valid: boolean;
    error?: string;
    suggestion?: string;
    examples?: string[];
}

interface ParsedPermission {
    toolName: string;
    ruleContent?: string;
}

function parsePermission(rule: string): ParsedPermission {
    const match = rule.match(/^([a-zA-Z0-9_-]+)(?:\((.*)\))?$/);
    if (!match) return { toolName: rule };
    return {
        toolName: match[1],
        ruleContent: match[2]
    };
}

function parseMcpServerName(toolName: string): { serverName: string, toolName?: string } | null {
    if (toolName.startsWith("mcp__")) {
        const parts = toolName.split("__");
        if (parts.length >= 2) {
            return {
                serverName: parts[1],
                toolName: parts[2] === "*" ? undefined : parts[2]
            };
        }
    }
    return null;
}

function countOccurrences(str: string, char: string): number {
    let count = 0;
    for (let i = 0; i < str.length; i++) {
        if (str[i] === char) {
            // Check for escaped characters if necessary, simplified for now
            let slashes = 0;
            let j = i - 1;
            while (j >= 0 && str[j] === "\\") {
                slashes++;
                j--;
            }
            if (slashes % 2 === 0) count++;
        }
    }
    return count;
}

export function validatePermissionRule(rule: string): PermissionValidationResult {
    if (!rule || rule.trim() === "") {
        return { valid: false, error: "Permission rule cannot be empty" };
    }

    const open = countOccurrences(rule, "(");
    const close = countOccurrences(rule, ")");
    if (open !== close) {
        return { valid: false, error: "Mismatched parentheses", suggestion: "Ensure all opening parentheses have matching closing parentheses" };
    }

    if (rule.includes("()")) {
        const toolName = rule.substring(0, rule.indexOf("("));
        if (!toolName) {
            return { valid: false, error: "Empty parentheses with no tool name", suggestion: "Specify a tool name before the parentheses" };
        }
        return {
            valid: false,
            error: "Empty parentheses",
            suggestion: `Either specify a pattern or use just "${toolName}" without parentheses`,
            examples: [`${toolName}`, `${toolName}(some-pattern)`]
        };
    }

    const parsed = parsePermission(rule);
    const mcpInfo = parseMcpServerName(parsed.toolName);

    if (mcpInfo) {
        if (parsed.ruleContent !== undefined) {
            return {
                valid: false,
                error: "MCP rules do not support patterns in parentheses",
                suggestion: `Use "${parsed.toolName}" without parentheses, or use "mcp__${mcpInfo.serverName}__*" for all tools`,
                examples: [`mcp__${mcpInfo.serverName}`, `mcp__${mcpInfo.serverName}__*`, mcpInfo.toolName ? `mcp__${mcpInfo.serverName}__${mcpInfo.toolName}` : undefined].filter(Boolean) as string[]
            };
        }
        return { valid: true };
    }

    if (!parsed.toolName || parsed.toolName.length === 0) {
        return { valid: false, error: "Tool name cannot be empty" };
    }

    if (parsed.toolName[0] !== parsed.toolName[0].toUpperCase()) {
        return {
            valid: false,
            error: "Tool names must start with uppercase",
            suggestion: `Use "${parsed.toolName.charAt(0).toUpperCase() + parsed.toolName.slice(1)}"`
        };
    }

    const customValidator = PERMISSION_CONFIG.customValidation[parsed.toolName];
    if (customValidator && parsed.ruleContent !== undefined) {
        const result = customValidator(parsed.ruleContent);
        if (!result.valid) return result;
    }

    if (PERMISSION_CONFIG.bashPrefixTools.includes(parsed.toolName) && parsed.ruleContent !== undefined) {
        const content = parsed.ruleContent;
        if (content.includes(":*") && !content.endsWith(":*")) {
            return {
                valid: false,
                error: "The :* pattern must be at the end",
                suggestion: "Move :* to the end for prefix matching",
                examples: ["Bash(npm run:*)", "Bash(git commit:*)"]
            };
        }
        if (content.includes(" * ") && !content.endsWith(":*")) {
            return {
                valid: false,
                error: "Wildcards in the middle of commands are not supported",
                suggestion: 'Use prefix matching with ":*" or specify exact commands',
                examples: ["Bash(npm run:*) - allows any npm run command", "Bash(npm install express) - allows exact command", "Bash - allows all commands"]
            };
        }
        if (content === ":*") {
            return {
                valid: false,
                error: "Prefix cannot be empty before :*",
                suggestion: "Specify a command prefix before :*",
                examples: ["Bash(npm:*)", "Bash(git:*)"]
            };
        }

        const quotes = ['"', "'"];
        for (const q of quotes) {
            if ((content.match(new RegExp(q, "g")) || []).length % 2 !== 0) {
                return {
                    valid: false,
                    error: `Unmatched ${q} in Bash pattern`,
                    suggestion: "Ensure all quotes are properly paired"
                };
            }
        }

        if (content === "*") {
            return {
                valid: false,
                error: 'Use "Bash" without parentheses to allow all commands',
                suggestion: "Remove the parentheses or specify a command pattern",
                examples: ["Bash", "Bash(npm:*)", "Bash(npm install)"]
            };
        }

        const wildcardIndex = content.indexOf("*");
        if (wildcardIndex !== -1 && !content.includes("/")) {
            const prefix = content.substring(0, wildcardIndex);
            const isSuffix = content.substring(wildcardIndex + 1).startsWith(".");
            const isColon = prefix.endsWith(":");
            if (!isSuffix && !isColon) {
                return {
                    valid: false,
                    error: 'Use ":*" for prefix matching, not just "*"',
                    suggestion: `Change to "Bash(${content.replace(/\*/g, ":*")})" for prefix matching`,
                    examples: ["Bash(npm run:*)", "Bash(git:*)"]
                };
            }
        }
    }

    if (PERMISSION_CONFIG.filePatternTools.includes(parsed.toolName) && parsed.ruleContent !== undefined) {
        const content = parsed.ruleContent;
        if (content.includes(":*")) {
            return {
                valid: false,
                error: 'The ":*" syntax is only for Bash prefix rules',
                suggestion: 'Use glob patterns like "*" or "**" for file matching',
                examples: [`${parsed.toolName}(*.ts) - matches .ts files`, `${parsed.toolName}(src/**) - matches all files in src`, `${parsed.toolName}(**/*.test.ts) - matches test files`]
            };
        }
        if (content.includes("*") && !content.match(/^\*|\*$|\*\*|\/\*|\*\.|\*\)/) && !content.includes("**")) {
            return {
                valid: false,
                error: "Wildcard placement might be incorrect",
                suggestion: "Wildcards are typically used at path boundaries",
                examples: [`${parsed.toolName}(*.js) - all .js files`, `${parsed.toolName}(src/*) - all files directly in src`, `${parsed.toolName}(src/**) - all files recursively in src`]
            };
        }
    }

    return { valid: true };
}
