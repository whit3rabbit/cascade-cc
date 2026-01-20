// Logic from chunk_487.ts (Plan Permission Request Components)

import React, { useCallback, useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import fs from "node:fs";
import { figures } from "../../vendor/terminalFigures.js";
import { PermissionDialogLayout, PermissionRuleSummary, PermissionSelect, type PermissionOption } from "./PermissionComponents.js";
import { getEditorCommand, editInEditor, openFileInEditor } from "../../services/terminal/EditorService.js";
import { logTelemetryEvent } from "../../services/telemetry/telemetryInit.js";
import { setHasExitedPlanMode, setNeedsPlanModeExitAttachment, updatePlanModeExitAttachment } from "../../services/session/globalState.js";

function readPlanFromFile(filePath?: string) {
    if (!filePath) return null;
    try {
        return fs.readFileSync(filePath, "utf8");
    } catch {
        return null;
    }
}

// --- Enter Plan Mode Request (qd2) ---
export function EnterPlanModePermissionRequest({ toolUseConfirm, onDone, onReject }: any) {
    const toolPermissionContext = toolUseConfirm.toolUseContext?.toolPermissionContext ?? toolUseConfirm.toolPermissionContext;
    const currentMode = toolPermissionContext?.mode ?? "default";

    const handleAction = useCallback((action: string) => {
        if (action === "yes") {
            updatePlanModeExitAttachment(currentMode, "plan");
            toolUseConfirm.onAllow({}, [{ type: "setMode", mode: "plan", destination: "session" }]);
            onDone();
            return;
        }
        onDone();
        onReject();
        toolUseConfirm.onReject();
    }, [onDone, onReject, toolUseConfirm]);

    return (
        <PermissionDialogLayout title="Enter plan mode?" color="planMode">
            <Box flexDirection="column" marginTop={1} paddingX={1}>
                <Text>Claude wants to enter plan mode to explore and design an implementation approach.</Text>
                <Box marginTop={1} flexDirection="column">
                    <Text dimColor>In plan mode, Claude will:</Text>
                    <Text dimColor> · Explore the codebase thoroughly</Text>
                    <Text dimColor> · Identify existing patterns</Text>
                    <Text dimColor> · Design an implementation strategy</Text>
                    <Text dimColor> · Present a plan for your approval</Text>
                </Box>
                <Box marginTop={1}>
                    <Text dimColor>No code changes will be made until you approve the plan.</Text>
                </Box>
                <Box marginTop={1}>
                    <PermissionSelect
                        options={[
                            { label: "Yes, enter plan mode", value: "yes" },
                            { label: "No, start implementing now", value: "no" }
                        ]}
                        onChange={handleAction}
                        onCancel={() => handleAction("no")}
                    />
                </Box>
            </Box>
        </PermissionDialogLayout>
    );
}

// --- Exit Plan Mode Request (Vd2) ---
export function ExitPlanModePermissionRequest({ toolUseConfirm, onDone, onReject }: any) {
    const [rejectFeedback, setRejectFeedback] = useState("");
    const [showSaved, setShowSaved] = useState(false);
    const [, setFocusedOption] = useState<string | null>(null);

    const planFromInput = toolUseConfirm.input?.plan;
    const planFilePath = toolUseConfirm.input?.filePath;
    const planFromFile = readPlanFromFile(planFilePath) ?? undefined;
    const initialPlan = planFromInput ?? planFromFile ?? "No plan found. Please write your plan to the plan file first.";

    const [plan, setPlan] = useState(initialPlan);
    const isPlanMissing = planFromInput === undefined && !planFromFile;
    const isEmptyPlan = !plan || plan.trim() === "";

    useEffect(() => {
        if (!showSaved) return;
        const timer = setTimeout(() => setShowSaved(false), 5000);
        return () => clearTimeout(timer);
    }, [showSaved]);

    useInput((input, key) => {
        if (key.ctrl && input.toLowerCase() === "g") {
            void logTelemetryEvent("tengu_plan_external_editor_used", {});
            if (isPlanMissing && planFilePath) {
                const updated = openFileInEditor(planFilePath);
                if (updated !== null) {
                    setPlan(updated);
                    setShowSaved(true);
                }
                return;
            }
            const updated = editInEditor(plan);
            if (updated !== null && updated !== plan) {
                setPlan(updated);
                setShowSaved(true);
            }
        }
    });

    const handleAction = useCallback((action: string) => {
        const payload = isPlanMissing ? {} : { plan };

        if (action === "yes-bypass-permissions") {
            void logTelemetryEvent("tengu_plan_exit", {
                planLengthChars: plan.length,
                outcome: action
            });
            setHasExitedPlanMode(true);
            setNeedsPlanModeExitAttachment(true);
            toolUseConfirm.onAllow(payload, [{ type: "setMode", mode: "bypassPermissions", destination: "session" }]);
            onDone();
            return;
        }
        if (action === "yes-accept-edits") {
            void logTelemetryEvent("tengu_plan_exit", {
                planLengthChars: plan.length,
                outcome: action
            });
            setHasExitedPlanMode(true);
            setNeedsPlanModeExitAttachment(true);
            toolUseConfirm.onAllow(payload, [{ type: "setMode", mode: "acceptEdits", destination: "session" }]);
            onDone();
            return;
        }
        if (action === "yes-default") {
            void logTelemetryEvent("tengu_plan_exit", {
                planLengthChars: plan.length,
                outcome: action
            });
            setHasExitedPlanMode(true);
            setNeedsPlanModeExitAttachment(true);
            toolUseConfirm.onAllow(payload, [{ type: "setMode", mode: "default", destination: "session" }]);
            onDone();
            return;
        }

        const trimmed = rejectFeedback.trim();
        if (!trimmed) return;
        void logTelemetryEvent("tengu_plan_exit", {
            planLengthChars: plan.length,
            outcome: "no"
        });
        onDone();
        onReject();
        toolUseConfirm.onReject(trimmed);
    }, [isPlanMissing, onDone, onReject, plan, rejectFeedback, toolUseConfirm]);

    const handleEmptyPlanDecision = useCallback((action: string) => {
        if (action === "yes") {
            void logTelemetryEvent("tengu_plan_exit", {
                planLengthChars: 0,
                outcome: "yes-default"
            });
            setHasExitedPlanMode(true);
            setNeedsPlanModeExitAttachment(true);
            toolUseConfirm.onAllow({}, [{ type: "setMode", mode: "default", destination: "session" }]);
            onDone();
            return;
        }
        void logTelemetryEvent("tengu_plan_exit", {
            planLengthChars: 0,
            outcome: "no"
        });
        onDone();
        onReject();
        toolUseConfirm.onReject();
    }, [onDone, onReject, toolUseConfirm]);

    const toolPermissionContext = toolUseConfirm.toolUseContext?.toolPermissionContext ?? toolUseConfirm.toolPermissionContext;
    const editorLabel = getEditorCommand();

    if (isEmptyPlan) {
        return (
            <PermissionDialogLayout title="Exit plan mode?" color="planMode">
                <Box flexDirection="column" paddingX={1} marginTop={1}>
                    <Text>Claude wants to exit plan mode</Text>
                    <Box marginTop={1}>
                        <PermissionSelect
                            options={[
                                { label: "Yes", value: "yes" },
                                { label: "No", value: "no" }
                            ]}
                            onChange={handleEmptyPlanDecision}
                            onCancel={() => handleEmptyPlanDecision("no")}
                        />
                    </Box>
                </Box>
            </PermissionDialogLayout>
        );
    }

    const allowOptions = toolPermissionContext?.isBypassPermissionsModeAvailable
        ? [{ label: "Yes, and bypass permissions", value: "yes-bypass-permissions" }]
        : [{ label: "Yes, and auto-accept edits", value: "yes-accept-edits" }];

    const options: PermissionOption[] = [
        ...allowOptions,
        { label: "Yes, and manually approve edits", value: "yes-default" },
        {
            type: "input",
            label: "No, keep planning",
            value: "no",
            placeholder: "Type here to tell Claude what to change",
            onChange: setRejectFeedback
        }
    ];

    return (
        <>
            <PermissionDialogLayout title="Ready to code?" color="planMode" innerPaddingX={0}>
                <Box flexDirection="column" marginTop={1}>
                    <Box paddingX={1}>
                        <Text>Here is Claude's plan:</Text>
                    </Box>
                    <Box
                        borderDimColor
                        borderColor="subtle"
                        borderStyle="round"
                        flexDirection="column"
                        borderLeft={false}
                        borderRight={false}
                        paddingX={1}
                        marginBottom={1}
                        overflow="hidden"
                    >
                        <Text>{plan}</Text>
                    </Box>
                    <Box flexDirection="column" paddingX={1}>
                        <PermissionRuleSummary permissionResult={toolUseConfirm.permissionResult} toolType="tool" />
                        <Text dimColor>Would you like to proceed?</Text>
                        <Box marginTop={1}>
                            <PermissionSelect
                                options={options}
                                onChange={handleAction}
                                onCancel={() => {
                                    void logTelemetryEvent("tengu_plan_exit", {
                                        planLengthChars: plan.length,
                                        outcome: "no"
                                    });
                                    onDone();
                                    onReject();
                                    toolUseConfirm.onReject();
                                }}
                                onFocus={(value) => setFocusedOption(value)}
                            />
                        </Box>
                    </Box>
                </Box>
            </PermissionDialogLayout>

            {editorLabel && (
                <Box flexDirection="row" gap={1} paddingX={1} marginTop={1}>
                    <Text dimColor>ctrl-g to edit in </Text>
                    <Text bold dimColor>{editorLabel}</Text>
                    {isPlanMissing && planFilePath && <Text dimColor> · {planFilePath}</Text>}
                    {showSaved && (
                        <>
                            <Text dimColor> · </Text>
                            <Text color="success">{figures.tick}Plan saved!</Text>
                        </>
                    )}
                </Box>
            )}
        </>
    );
}
