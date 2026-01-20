
// Logic from chunk_535.ts (Attachments & Plan Mode)

import { createUserMessage } from './MessageFactory.js';

// --- Attachment Normalizer (X97) ---
export function normalizeAttachment(attachment: any) {
    switch (attachment.type) {
        case "directory":
            return createUserMessage(`Contents of ${attachment.path}:\n${attachment.content}`, { isMeta: true });
        case "file":
            return createUserMessage(`File ${attachment.filename}:\n${attachment.content.text}`, { isMeta: true });
        case "todo":
            return createUserMessage(`Todo list updated:\n${JSON.stringify(attachment.content)}`, { isMeta: true });
        case "plan_file_reference":
            return createUserMessage(`Active Plan at ${attachment.planFilePath}:\n${attachment.planContent}`, { isMeta: true });
        case "diagnostics":
            return createUserMessage(`New diagnostics detected:\n${attachment.files.join('\n')}`, { isMeta: true });
        default:
            return null;
    }
}

// --- Plan Mode Prompts (Y97 / J97) ---
export function getPlanModePrompt(isSubAgent: boolean, context: any) {
    const { planExists, planFilePath } = context;
    const base = `Plan mode is active. You MUST NOT make any changes. Only edit the plan file at ${planFilePath}.`;

    if (isSubAgent) {
        return `${base}\nAnswer comprehensively and use TaskUpdate if needed.`;
    }

    return `${base}\nPhase 1: Explore\nPhase 2: Design\nPhase 3: Review\nPhase 4: Finalize Plan\nPhase 5: Call Done`;
}
