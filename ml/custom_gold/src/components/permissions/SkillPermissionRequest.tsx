// Logic from chunk_488.ts (Skill Permission Request Component)

import React, { useCallback, useMemo } from "react";
import { Box, Text } from "ink";
import path from "node:path";
import { PermissionDialogLayout, PermissionRuleSummary, PermissionSelect, logPermissionPrompt, logUnaryEvent } from "./PermissionComponents.js";
import { getProjectRoot } from "../../utils/shared/pathUtils.js";
import { ExecuteSkillTool } from "../../services/terminal/SlashCommandExecutor.js";

export function SkillPermissionRequest({ toolUseConfirm, onDone, onReject }: any) {
    const parsed = ExecuteSkillTool.inputSchema.safeParse(toolUseConfirm.input);
    const skillName = parsed.success ? parsed.data.skill : (console.error(`Failed to parse skill tool input: ${parsed.error.message}`), "");
    const command = toolUseConfirm.permissionResult?.behavior === "ask" && toolUseConfirm.permissionResult?.metadata && "command" in toolUseConfirm.permissionResult.metadata
        ? toolUseConfirm.permissionResult.metadata.command
        : undefined;

    const completionMeta = useMemo(() => ({
        completion_type: "tool_use_single",
        language_name: "none"
    }), []);

    logPermissionPrompt(toolUseConfirm, completionMeta);

    const projectLabel = path.basename(getProjectRoot()) || "this project";
    const options = useMemo(() => {
        const baseOptions = [{ label: "Yes", value: "yes" }];
        const exactOption = {
            label: (
                <Text>
                    Yes, and don't ask again for <Text bold>{skillName}</Text> in <Text bold>{projectLabel}</Text>
                </Text>
            ),
            value: "yes-exact"
        };
        const prefixOptions: { label: React.ReactNode; value: string }[] = [];
        const prefixIndex = skillName.indexOf(" ");
        if (prefixIndex > 0) {
            const prefix = skillName.substring(0, prefixIndex);
            prefixOptions.push({
                label: (
                    <Text>
                        Yes, and don't ask again for <Text bold>{`${prefix}:*`}</Text> commands in <Text bold>{projectLabel}</Text>
                    </Text>
                ),
                value: "yes-prefix"
            });
        }
        const noOption = {
            label: (
                <Text>
                    No, and tell Claude what to do differently <Text bold>(esc)</Text>
                </Text>
            ),
            value: "no"
        };
        return [...baseOptions, exactOption, ...prefixOptions, noOption];
    }, [projectLabel, skillName]);

    const handleAction = useCallback((action: string) => {
        switch (action) {
            case "yes":
                logUnaryEvent({
                    completionType: completionMeta.completion_type,
                    event: "accept",
                    languageName: completionMeta.language_name,
                    messageId: toolUseConfirm.assistantMessage?.message?.id
                });
                toolUseConfirm.onAllow(toolUseConfirm.input, []);
                onDone();
                break;
            case "yes-exact":
                logUnaryEvent({
                    completionType: completionMeta.completion_type,
                    event: "accept",
                    languageName: completionMeta.language_name,
                    messageId: toolUseConfirm.assistantMessage?.message?.id
                });
                toolUseConfirm.onAllow(toolUseConfirm.input, [{
                    type: "addRules",
                    rules: [{ toolName: toolUseConfirm.tool?.name, ruleContent: skillName }],
                    behavior: "allow",
                    destination: "localSettings"
                }]);
                onDone();
                break;
            case "yes-prefix": {
                logUnaryEvent({
                    completionType: completionMeta.completion_type,
                    event: "accept",
                    languageName: completionMeta.language_name,
                    messageId: toolUseConfirm.assistantMessage?.message?.id
                });
                const prefixIndex = skillName.indexOf(" ");
                const prefix = prefixIndex > 0 ? skillName.substring(0, prefixIndex) : skillName;
                toolUseConfirm.onAllow(toolUseConfirm.input, [{
                    type: "addRules",
                    rules: [{ toolName: toolUseConfirm.tool?.name, ruleContent: `${prefix}:*` }],
                    behavior: "allow",
                    destination: "localSettings"
                }]);
                onDone();
                break;
            }
            case "no":
                logUnaryEvent({
                    completionType: completionMeta.completion_type,
                    event: "reject",
                    languageName: completionMeta.language_name,
                    messageId: toolUseConfirm.assistantMessage?.message?.id
                });
                toolUseConfirm.onReject();
                onReject();
                onDone();
                break;
        }
    }, [completionMeta.completion_type, completionMeta.language_name, onDone, onReject, skillName, toolUseConfirm]);

    const isSkill = command?.loadedFrom !== "commands_DEPRECATED";
    const typeLabel = isSkill ? "skill" : "slash command";
    const typeLabelTitle = isSkill ? "Skill" : "Slash Command";

    return (
        <PermissionDialogLayout title={`Use ${typeLabel} "${skillName}"?`}>
            <Text>Claude may use instructions, code, or files from this {typeLabelTitle}.</Text>
            <Box flexDirection="column" paddingX={2} paddingY={1}>
                <Text dimColor>{command?.description}</Text>
            </Box>
            <Box flexDirection="column">
                <PermissionRuleSummary permissionResult={toolUseConfirm.permissionResult} toolType="tool" />
                <Text>Do you want to proceed?</Text>
                <PermissionSelect options={options} onChange={handleAction} onCancel={() => handleAction("no")} />
            </Box>
        </PermissionDialogLayout>
    );
}
