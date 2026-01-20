// Logic from chunk_465.ts (Bash Permission Coordination, Suggestions)

import React from "react";
import { Text } from "ink";
import path from "node:path";
import { getProjectRoot } from "../../utils/shared/pathUtils.js";
import { BashParser, validateCommandPaths, validateRedirections } from "./PathValidator.js";
import { validateBashCommand, checkDangerousPatterns } from "./BashValidator.js";
import { findMatchingRules, validateCommandWithRules } from "./CommandRuleValidator.js";
import { sandboxService } from "../sandbox/sandboxService.js";

// --- Coordination Logic (OE0, LE0, fk2) ---
export async function coordinateBashPermissionCheck(input: any, toolUseContext: any, getSubcommandPrefixes?: any) {
    const command = input.command.trim();
    const appState = await toolUseContext.getAppState?.();
    const permissionContext = appState?.toolPermissionContext ?? toolUseContext?.toolPermissionContext ?? {};

    const syntaxCheck = validateBashCommand(command);
    if (syntaxCheck.behavior === "deny") {
        const reason = {
            type: "other",
            reason: `Command contains malformed syntax that cannot be parsed: ${syntaxCheck.message ?? "unknown"}`
        };
        return {
            behavior: "ask",
            decisionReason: reason,
            message: syntaxCheck.message ?? "Command requires approval"
        };
    }

    if (sandboxService.isEnabled() && sandboxService.isAutoAllowBashEnabled?.() && !input?.dangerouslyDisableSandbox) {
        const autoAllow = checkExactCommandRule({ command }, permissionContext);
        if (autoAllow.behavior !== "passthrough") return autoAllow;
    }

    const exactRuleCheck = checkExactCommandRule({ command }, permissionContext);
    if (exactRuleCheck.behavior === "deny") return exactRuleCheck;

    // Handle shell operators / pipes
    const parser = await BashParser.parse(command);
    const subcommands = parser.getPipeSegments().filter((segment: string) => segment && segment.trim().length > 0);
    const cdCommands = subcommands.filter((segment: string) => segment.startsWith("cd "));

    if (cdCommands.length > 1) {
        const reason = {
            type: "other",
            reason: "Multiple directory changes in one command require approval for clarity"
        };
        return {
            behavior: "ask",
            decisionReason: reason,
            message: reason.reason
        };
    }

    const hasCd = cdCommands.length > 0;

    const subcommandResults = subcommands.map((segment: string) => {
        const mcpResult = validateMcpCommand(segment, permissionContext);
        if (mcpResult) return mcpResult;
        return checkPrefixCommandRule({ command: segment }, permissionContext, hasCd);
    });

    if (subcommandResults.find((result: any) => result.behavior === "deny")) {
        return {
            behavior: "deny",
            message: `Permission to use Bash with command ${command} has been denied.`,
            decisionReason: {
                type: "subcommandResults",
                reasons: new Map(subcommandResults.map((result: any, idx: number) => [subcommands[idx], result]))
            }
        };
    }

    const workingDirCheck = checkWorkingDirectoryAccess({ command }, permissionContext, hasCd);
    if (workingDirCheck.behavior !== "passthrough") return workingDirCheck;

    const askResult = subcommandResults.find((result: any) => result.behavior === "ask");
    if (askResult) return askResult;

    if (exactRuleCheck.behavior === "allow") return exactRuleCheck;

    const injectionCheck = checkDangerousPatterns(command);
    if (injectionCheck.behavior !== "passthrough") {
        return {
            behavior: "ask",
            message: injectionCheck.message ?? "Command contains patterns that require approval",
            decisionReason: {
                type: "other",
                reason: injectionCheck.message ?? "Command contains patterns that require approval"
            }
        };
    }

    if (subcommandResults.every((result: any) => result.behavior === "allow")) {
        return {
            behavior: "allow",
            updatedInput: input,
            decisionReason: {
                type: "subcommandResults",
                reasons: new Map(subcommandResults.map((result: any, idx: number) => [subcommands[idx], result]))
            }
        };
    }

    if (!getSubcommandPrefixes) {
        const fallthrough = buildSuggestionResult(subcommandResults, command);
        if (fallthrough) return fallthrough;
    }

    const prefixData = getSubcommandPrefixes ? await getSubcommandPrefixes(command, toolUseContext?.abortController?.signal, toolUseContext?.options?.isNonInteractiveSession) : null;
    if (toolUseContext?.abortController?.signal?.aborted) throw new Error("AbortError");

    const refreshedState = await toolUseContext.getAppState?.();
    const refreshedContext = refreshedState?.toolPermissionContext ?? permissionContext;

    if (subcommands.length === 1) {
        return validateCommandWithRules({ command: subcommands[0] }, refreshedContext, prefixData, hasCd);
    }

    const prefixMap = new Map<string, any>();
    for (const subcommand of subcommands) {
        prefixMap.set(subcommand, validateCommandWithRules({ command: subcommand }, refreshedContext, prefixData?.subcommandPrefixes?.get?.(subcommand), hasCd));
    }

    if (subcommands.every((subcommand) => prefixMap.get(subcommand)?.behavior === "allow")) {
        return {
            behavior: "allow",
            updatedInput: input,
            decisionReason: {
                type: "subcommandResults",
                reasons: prefixMap
            }
        };
    }

    const summary = buildSuggestionResult(Array.from(prefixMap.values()), command);
    if (summary) return summary;

    return {
        behavior: "passthrough",
        message: "Command requires approval",
        decisionReason: {
            type: "subcommandResults",
            reasons: prefixMap
        }
    };
}

function buildSuggestionResult(results: any[], command: string) {
    const suggestionMap = new Map<string, any>();
    for (const result of results) {
        if (result.behavior === "ask" || result.behavior === "passthrough") {
            const updates = "suggestions" in result ? result.suggestions : undefined;
            const flattened = formatSuggestionUpdates(updates);
            for (const rule of flattened) {
                const key = stringifyRule(rule);
                suggestionMap.set(key, rule);
            }
        }
    }

    const decisionReason = {
        type: "subcommandResults",
        reasons: new Map(results.map((result: any) => [command, result]))
    };

    const suggestions = suggestionMap.size > 0 ? [{
        type: "addRules",
        rules: Array.from(suggestionMap.values()),
        behavior: "allow",
        destination: "localSettings"
    }] : undefined;

    return {
        behavior: "passthrough",
        message: `Permission to use Bash with command ${command} requires approval.`,
        decisionReason,
        suggestions
    };
}

function stringifyRule(rule: any) {
    return `${rule.toolName}:${rule.ruleContent ?? ""}`;
}

function formatSuggestionUpdates(suggestions: any[] | undefined) {
    if (!suggestions) return [];
    return suggestions.flatMap((update) => {
        if (update.type === "addRules") return update.rules ?? [];
        return [];
    });
}

export function checkExactCommandRule(input: any, context: any) {
    const command = input.command.trim();
    const { matchingDenyRules, matchingAskRules, matchingAllowRules } = findMatchingRules(command, context, "exact");

    if (matchingDenyRules[0]) {
        return {
            behavior: "deny",
            message: `Permission to use Bash with command ${command} has been denied.`,
            decisionReason: {
                type: "rule",
                rule: matchingDenyRules[0]
            }
        };
    }

    if (matchingAskRules[0]) {
        return {
            behavior: "ask",
            message: "Command requires approval",
            decisionReason: {
                type: "rule",
                rule: matchingAskRules[0]
            }
        };
    }

    if (matchingAllowRules[0]) {
        return {
            behavior: "allow",
            updatedInput: input,
            decisionReason: {
                type: "rule",
                rule: matchingAllowRules[0]
            }
        };
    }

    const reason = {
        type: "other",
        reason: "This command requires approval"
    };

    return {
        behavior: "passthrough",
        message: "Command requires approval",
        decisionReason: reason,
        suggestions: buildBashRuleSuggestions(command)
    };
}

export function checkPrefixCommandRule(input: any, context: any, hasCd: boolean) {
    const command = input.command.trim();
    const exactResult = checkExactCommandRule(input, context);
    if (exactResult.behavior === "deny" || exactResult.behavior === "ask") return exactResult;

    const { matchingDenyRules, matchingAskRules, matchingAllowRules } = findMatchingRules(command, context, "prefix");

    if (matchingDenyRules[0]) {
        return {
            behavior: "deny",
            message: `Permission to use Bash with command ${command} has been denied.`,
            decisionReason: {
                type: "rule",
                rule: matchingDenyRules[0]
            }
        };
    }

    if (matchingAskRules[0]) {
        return {
            behavior: "ask",
            message: "Command requires approval",
            decisionReason: {
                type: "rule",
                rule: matchingAskRules[0]
            }
        };
    }

    const pathCheck = checkWorkingDirectoryAccess(input, context, hasCd);
    if (pathCheck.behavior !== "passthrough") return pathCheck;

    if (exactResult.behavior === "allow") return exactResult;

    if (matchingAllowRules[0]) {
        return {
            behavior: "allow",
            updatedInput: input,
            decisionReason: {
                type: "rule",
                rule: matchingAllowRules[0]
            }
        };
    }

    const commandCheck = validateCommandWithRules(input, context, null, hasCd);
    if (commandCheck.behavior !== "passthrough") return commandCheck;

    if (validateBashCommand(command).behavior === "allow") {
        return {
            behavior: "allow",
            updatedInput: input,
            decisionReason: {
                type: "other",
                reason: "Read-only command is allowed"
            }
        };
    }

    const reason = {
        type: "other",
        reason: "This command requires approval"
    };

    return {
        behavior: "passthrough",
        message: "Command requires approval",
        decisionReason: reason,
        suggestions: buildBashRuleSuggestions(command)
    };
}

function extractRedirections(command: string) {
    const matches = command.matchAll(/(?:^|\\s)([12]?>+)(\\s*\\S+)/g);
    const redirections: Array<{ operator: string; target: string }> = [];
    for (const match of matches) {
        redirections.push({ operator: match[1], target: match[2].trim() });
    }
    return redirections;
}

function checkWorkingDirectoryAccess(input: any, context: any, hasCd: boolean) {
    const cwd = getProjectRoot();
    const pathResult = validateCommandPaths(input.command, cwd, context, hasCd);
    if (pathResult?.behavior && pathResult.behavior !== "passthrough") return pathResult;

    const redirections = extractRedirections(input.command);
    const redirectResult = validateRedirections(redirections, cwd, context, hasCd);
    if (redirectResult?.behavior && redirectResult.behavior !== "passthrough") return redirectResult;

    return { behavior: "passthrough" };
}

function buildBashRuleSuggestions(command: string) {
    return [{
        type: "addRules",
        rules: [{ toolName: "Bash", ruleContent: `${command}:*` }],
        behavior: "allow",
        destination: "localSettings"
    }];
}

function validateMcpCommand(command: string, context: any) {
    if (!command.startsWith("mcp__")) return null;
    return {
        behavior: "ask",
        message: "MCP command requires approval",
        decisionReason: {
            type: "other",
            reason: "MCP tool requires permission"
        }
    };
}

function boldText(text: string) {
    return React.createElement(Text, { bold: true }, text);
}

function formatListWithBold(values: string[]) {
    if (values.length === 0) return "";
    if (values.length === 1) return boldText(values[0]);
    if (values.length === 2) {
        return React.createElement(Text, null, boldText(values[0]), " and ", boldText(values[1]));
    }
    return React.createElement(Text, null, boldText(values.slice(0, -1).join(", ")), ", and ", boldText(values.slice(-1)[0]));
}

function formatCommandSummary(values: string[]) {
    if (values.join(", ").length > 50) return "similar";
    return formatListWithBold(values);
}

function formatDirectoryList(paths: string[]) {
    if (paths.length === 0) return "";
    const names = paths.map((value) => value.split("/").pop() || value);
    if (names.length === 1) {
        return React.createElement(Text, null, boldText(names[0]), path.sep);
    }
    if (names.length === 2) {
        return React.createElement(Text, null, boldText(names[0]), path.sep, " and ", boldText(names[1]), path.sep);
    }
    return React.createElement(Text, null, boldText(names[0]), path.sep, ", ", boldText(names[1]), path.sep, " and ", String(paths.length - 2), " more");
}

function stripRulePrefix(rule: string) {
    return rule.match(/^(.+):\*$/)?.[1] ?? rule;
}

function stripRedirections(command: string) {
    const parts = command.split(/\s+[<>]/);
    return parts[0] ?? command;
}

export function formatPermissionSuggestions(suggestions: any[]) {
    const ruleUpdates = suggestions.filter((update) => update.type === "addRules").flatMap((update) => update.rules || []);
    const readRules = ruleUpdates.filter((rule) => rule.toolName === "Read");
    const bashRules = ruleUpdates.filter((rule) => rule.toolName === "Bash");
    const directoryUpdates = suggestions.filter((update) => update.type === "addDirectories").flatMap((update) => update.directories || []);

    const readDirectories = readRules
        .map((rule) => rule.ruleContent?.replace("/**", "") || "")
        .filter(Boolean);

    const bashCommands = bashRules.flatMap((rule) => {
        if (!rule.ruleContent) return [];
        const cleaned = stripRulePrefix(rule.ruleContent);
        const command = stripRedirections(cleaned);
        return [command];
    });

    const hasDirectories = directoryUpdates.length > 0;
    const hasRead = readDirectories.length > 0;
    const hasCommands = bashCommands.length > 0;

    if (hasRead && !hasDirectories && !hasCommands) {
        if (readDirectories.length === 1) {
            const single = readDirectories[0];
            const name = single.split("/").pop() || single;
            return React.createElement(Text, null, "Yes, allow reading from ", boldText(name), path.sep, " from this project");
        }
        return React.createElement(Text, null, "Yes, allow reading from ", formatDirectoryList(readDirectories), " from this project");
    }

    if (hasDirectories && !hasRead && !hasCommands) {
        if (directoryUpdates.length === 1) {
            const single = directoryUpdates[0];
            const name = single.split("/").pop() || single;
            return React.createElement(Text, null, "Yes, and always allow access to ", boldText(name), path.sep, " from this project");
        }
        return React.createElement(Text, null, "Yes, and always allow access to ", formatDirectoryList(directoryUpdates), " from this project");
    }

    if (hasCommands && !hasDirectories && !hasRead) {
        return React.createElement(Text, null, "Yes, and don't ask again for ", formatCommandSummary(bashCommands), " commands in ", boldText(path.basename(getProjectRoot())));
    }

    if ((hasDirectories || hasRead) && !hasCommands) {
        const combined = [...directoryUpdates, ...readDirectories];
        if (hasDirectories && hasRead) {
            return React.createElement(Text, null, "Yes, and always allow access to ", formatDirectoryList(combined), " from this project");
        }
    }

    if ((hasDirectories || hasRead) && hasCommands) {
        const combined = [...directoryUpdates, ...readDirectories];
        if (combined.length === 1 && bashCommands.length === 1) {
            return React.createElement(Text, null, "Yes, and allow access to ", formatDirectoryList(combined), " and ", formatCommandSummary(bashCommands), " commands");
        }
        return React.createElement(Text, null, "Yes, and allow ", formatDirectoryList(combined), " access and ", formatCommandSummary(bashCommands), " commands");
    }

    return null;
}

// --- Suggestion UI Logic (gk2, rm5) ---
export function getBashPermissionOptions({
    suggestions = [],
    onRejectFeedbackChange,
    onAcceptFeedbackChange,
    yesInputMode = false,
    noInputMode = false,
    acceptFeedbackEnabled = false
}: any) {
    const options: any[] = [];

    if (acceptFeedbackEnabled && yesInputMode) {
        options.push({
            type: "input",
            label: "Yes,",
            value: "yes",
            placeholder: "tell Claude what to do next",
            onChange: onAcceptFeedbackChange,
            allowEmptySubmit: true
        });
    } else {
        options.push({ label: "Yes", value: "yes" });
    }

    if (suggestions.length > 0) {
        const label = formatPermissionSuggestions(suggestions);
        if (label) {
            options.push({
                label,
                value: "yes-apply-suggestions"
            });
        }
    }

    if (acceptFeedbackEnabled && noInputMode) {
        options.push({
            type: "input",
            label: "No,",
            value: "no",
            placeholder: "tell Claude what to do differently",
            onChange: onRejectFeedbackChange,
            allowEmptySubmit: true
        });
    } else if (acceptFeedbackEnabled) {
        options.push({ label: "No", value: "no" });
    } else {
        options.push({
            type: "input",
            label: "No",
            value: "no",
            placeholder: "Type here to tell Claude what to do differently",
            onChange: onRejectFeedbackChange
        });
    }

    return options;
}

// --- Utils (wE0) ---
export function getCommandFromRule(rule: string) {
    return rule.match(/^(.+):\*$/)?.[1] ?? null;
}
