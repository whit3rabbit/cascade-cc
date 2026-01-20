
import React from 'react';
import { Box, Text } from 'ink';

// Mock components for now
const DefaultPermissionView = (props: any) => <Box><Text>Permission needed for {props.toolUseConfirm.tool.name}</Text></Box>;

// va5 mapper logic
export function getPermissionViewForTool(tool: any): React.ComponentType<any> {
    const name = tool.name;

    switch (name) {
        // Mocked mappings based on chunk_489:560
        case "AskUserQuestion":
            // return AskUserQuestionPermissionView;
            return DefaultPermissionView;
        case "ExitPlanMode":
            // return ExitPlanModePermissionView;
            return DefaultPermissionView;
        case "EnterPlanMode":
            // return EnterPlanModePermissionView;
            return DefaultPermissionView;
        case "Skill":
            // return SkillPermissionView;
            return DefaultPermissionView;
        default:
            return DefaultPermissionView;
    }
}

// nd2 wrapper
export function ToolPermissionRouter({
    toolUseConfirm,
    onDone,
    onReject,
    ...props
}: any) {
    const View = getPermissionViewForTool(toolUseConfirm.tool);

    return (
        <View
            toolUseConfirm={toolUseConfirm}
            onDone={onDone}
            onReject={onReject}
            {...props}
        />
    );
}

export default ToolPermissionRouter;
