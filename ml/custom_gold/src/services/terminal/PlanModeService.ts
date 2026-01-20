
// Logic from chunk_535.ts

export interface PlanModeContext {
    isSubAgent: boolean;
    planExists: boolean;
    planFilePath: string;
}

export function getPlanModeSystemPrompt(context: PlanModeContext) {
    if (context.isSubAgent) return [];

    // Logic from Y97/J97
    return `Plan mode is active. The user indicated that they do not want you to execute yet -- you MUST NOT make any edits (with the exception of the plan file mentioned below), run any non-readonly tools (including changing configs or making commits), or otherwise make any changes to the system. This supercedes any other instructions you have received.

## Plan File Info:
${context.planExists ? `A plan file already exists at ${context.planFilePath}. You can read it and make incremental edits.` : `No plan file exists yet. You should create your plan at ${context.planFilePath}.`}
You should build your plan incrementally by writing to or editing this file. NOTE that this is the only file you are allowed to edit - other than this you are only allowed to take READ-ONLY actions.

## Plan Workflow
... (See original prompt for full workflow details)
`;
}

export function formatAttachment(attachment: any) {
    switch (attachment.type) {
        case "directory":
            return `[Directory Content] ${attachment.path}: \n${attachment.content}`;
        case "file":
            return `[File Content] ${attachment.filename}: \n${attachment.content.type === 'text' ? attachment.content.content : '[Binary/Image]'}`;
        case "plan_mode":
            return getPlanModeSystemPrompt(attachment);
        case "plan_file_reference":
            return `A plan file exists from plan mode at: ${attachment.planFilePath}\n\nPlan contents:\n${attachment.planContent}`;
        case "todo":
            if (attachment.itemCount === 0) return "This is a reminder that your todo list is currently empty.";
            return `Your todo list has changed: ${JSON.stringify(attachment.content)}`;
        // ... implement other cases as needed
        default:
            return `[System Attachment: ${attachment.type}]`;
    }
}
