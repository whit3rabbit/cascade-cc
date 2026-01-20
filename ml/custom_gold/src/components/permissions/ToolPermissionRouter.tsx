// Logic from chunk_489.ts (Tool Permission Router)

import React, { useMemo } from "react";
import { useInput } from "ink";
import { FileToolPermissionRequest, GenericPermissionRequest, PermissionRequestRouter, WriteFilePermissionRequest, usePermissionPromptNotification, FetchPermissionRequest, NotebookPermissionRequest } from "./PermissionRequests.js";
import { EnterPlanModePermissionRequest, ExitPlanModePermissionRequest } from "./PlanPermissionRequests.js";
import { SkillPermissionRequest } from "./SkillPermissionRequest.js";
import { AskUserQuestionWizard } from "./AskUserQuestionWizard.js";
import { FetchTool } from "../../tools/definitions/FetchTool.js";
import { FileWriteTool } from "../../tools/definitions/fileWrite.js";
import { BashTool } from "../../tools/bash/BashTool.js";
import { ExecuteSkillTool } from "../../services/terminal/SlashCommandExecutor.js";
import { EnterPlanModeTool, ExitPlanModeTool } from "../../tools/definitions/PlanTools.js";
import { AskUserQuestionTool } from "./AskUserQuestionWizard.js";
import { NotebookEditTool } from "../../tools/definitions/NotebookEditTool.js";
import { FileReadTool } from "../../tools/definitions/fileRead.js";
import { GrepSearchTool } from "../../tools/definitions/search.js";

function isTool(tool: any, target: any, names: string[] = []) {
    if (!tool) return false;
    if (tool === target) return true;
    return Boolean(tool.name && names.includes(tool.name));
}

function getPermissionComponent(tool: any) {
    if (isTool(tool, FetchTool, ["WebFetch"])) return FetchPermissionRequest;
    if (isTool(tool, FileWriteTool, ["WriteFile", "FileWrite", "FileWriteTool"])) return WriteFilePermissionRequest;
    if (isTool(tool, BashTool, ["Bash", "bash"])) return PermissionRequestRouter;
    if (isTool(tool, ExecuteSkillTool, ["ExecuteSkill"])) return SkillPermissionRequest;
    if (isTool(tool, ExitPlanModeTool, ["ExitPlanMode"])) return ExitPlanModePermissionRequest;
    if (isTool(tool, EnterPlanModeTool, ["EnterPlanMode"])) return EnterPlanModePermissionRequest;
    if (isTool(tool, AskUserQuestionTool, ["AskUserQuestion"])) return AskUserQuestionWizard;
    if (isTool(tool, NotebookEditTool, ["NotebookEdit"])) return NotebookPermissionRequest;
    if (isTool(tool, FileReadTool, ["FileRead", "FileReadTool"])) return FileToolPermissionRequest;
    if (isTool(tool, GrepSearchTool, ["GrepSearchTool"])) return FileToolPermissionRequest;
    if (tool?.getPath) return FileToolPermissionRequest;
    return GenericPermissionRequest;
}

function getPermissionPromptTitle(toolUseConfirm: any) {
    const tool = toolUseConfirm.tool;
    if (isTool(tool, ExitPlanModeTool, ["ExitPlanMode"])) return "Claude Code needs your approval for the plan";
    if (isTool(tool, EnterPlanModeTool, ["EnterPlanMode"])) return "Claude Code wants to enter plan mode";
    const userFacingName = typeof tool?.userFacingName === "function" ? tool.userFacingName(toolUseConfirm.input) : tool?.userFacingName;
    const displayName = userFacingName ?? tool?.name ?? "";
    if (!displayName || displayName.trim() === "") return "Claude Code needs your attention";
    return `Claude needs your permission to use ${displayName}`;
}

export function ToolPermissionRouter(props: any) {
    const { toolUseConfirm, toolUseContext, onDone, onReject, verbose, workerBadge } = props;

    useInput((input, key) => {
        if (key.ctrl && input === "c") {
            onDone();
            onReject();
            toolUseConfirm.onReject();
        }
    });

    const promptTitle = useMemo(() => getPermissionPromptTitle(toolUseConfirm), [toolUseConfirm]);
    usePermissionPromptNotification(promptTitle, "permission_prompt");

    const Component = getPermissionComponent(toolUseConfirm.tool);

    return (
        <Component
            toolUseContext={toolUseContext}
            toolUseConfirm={toolUseConfirm}
            onDone={onDone}
            onReject={onReject}
            verbose={verbose}
            workerBadge={workerBadge}
        />
    );
}
