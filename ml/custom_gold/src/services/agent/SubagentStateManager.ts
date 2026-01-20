import path from "node:path";
import fs from "node:fs";
import { z } from "zod";
import { getSessionId } from "../session/globalState.js";
import { getConfigDir } from "../../utils/shared/pathUtils.js";
import { log, logError } from "../logger/loggerService.js";

// --- Schema ---
export const taskStatusSchema = z.enum(["pending", "in_progress", "completed"]);

export const taskSchema = z.object({
    content: z.string().min(1, "Content cannot be empty"),
    status: taskStatusSchema,
    activeForm: z.string().min(1, "Active form cannot be empty")
});

export const tasksSchema = z.array(taskSchema);
export type Task = z.infer<typeof taskSchema>;

// Logic from chunk_383.ts
export function createSummaryPrompt(previousSummary: string, additionalInstructions: string = ""): string {
    const base = `Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
  - Errors that you ran into and how you fixed them
  - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
6. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
7. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
8. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.
`;

    if (!additionalInstructions) return base;

    return `${base}\nThere may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary.\n${additionalInstructions}`;
}

export function formatSummary(text: string): string {
    let formatted = text;
    const analysisMatch = formatted.match(/<analysis>([\s\S]*?)<\/analysis>/);
    if (analysisMatch) {
        const analysis = analysisMatch[1] || "";
        formatted = formatted.replace(/<analysis>[\s\S]*?<\/analysis>/, `Analysis:\n${analysis.trim()}`);
    }
    const summaryMatch = formatted.match(/<summary>([\s\S]*?)<\/summary>/);
    if (summaryMatch) {
        const summary = summaryMatch[1] || "";
        formatted = formatted.replace(/<summary>[\s\S]*?<\/summary>/, `Summary:\n${summary.trim()}`);
    }
    return formatted.replace(/\n\n+/g, "\n\n").trim();
}

/**
 * Manage persistent state for subagents (tasks/todos).
 * Subagent state is stored as JSON files in the config directory.
 */
export class SubagentStateManager {
    private getTodosDir(): string {
        const dir = path.join(getConfigDir(), "todos");
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }
        return dir;
    }

    private getAgentStatePath(agentId: string): string {
        // Filename format: sessionId-agent-agentId.json
        const sessionId = getSessionId();
        const filename = `${sessionId}-agent-${agentId}.json`;
        return path.join(this.getTodosDir(), filename);
    }

    public readAgentState(agentId: string): Task[] {
        const filePath = this.getAgentStatePath(agentId);
        if (!fs.existsSync(filePath)) {
            return [];
        }

        try {
            const content = fs.readFileSync(filePath, "utf-8");
            const parsed = JSON.parse(content);
            const result = tasksSchema.safeParse(parsed);
            if (!result.success) {
                logError("agent-state", result.error, `Failed to validate state for agent ${agentId}`);
                return [];
            }
            return result.data;
        } catch (e) {
            logError("agent-state", e, `Failed to read state for agent ${agentId}`);
            return [];
        }
    }

    public writeAgentState(agentId: string, tasks: Task[]): void {
        const filePath = this.getAgentStatePath(agentId);
        try {
            fs.writeFileSync(filePath, JSON.stringify(tasks, null, 2), "utf-8");
        } catch (e) {
            logError("agent-state", e, `Failed to write state for agent ${agentId}`);
        }
    }

    public copyAgentState(sourceId: string, targetId: string): boolean {
        try {
            const state = this.readAgentState(sourceId);
            if (state.length === 0) return false;
            this.writeAgentState(targetId, state);
            return true;
        } catch (e) {
            return false;
        }
    }
}

// Internal Markdown Utilities (used by the prompt and summary logic)
export const MarkdownUtils = {
    escape(html: string, encode: boolean = false): string {
        if (encode) {
            if (/[&<>"']/.test(html)) {
                return html.replace(/[&<>"']/g, (m) => {
                    switch (m) {
                        case '&': return '&amp;';
                        case '<': return '&lt;';
                        case '>': return '&gt;';
                        case '"': return '&quot;';
                        case "'": return '&#39;';
                        default: return m;
                    }
                });
            }
        } else {
            if (/[&<>]/.test(html)) {
                return html.replace(/[&<>]/g, (m) => {
                    switch (m) {
                        case '&': return '&amp;';
                        case '<': return '&lt;';
                        case '>': return '&gt;';
                        default: return m;
                    }
                });
            }
        }
        return html;
    },

    cleanUrl(href: string): string | null {
        try {
            return encodeURI(href).replace(/%25/g, "%");
        } catch {
            return null;
        }
    },

    splitCells(tableRow: string, count?: number): string[] {
        const row = tableRow.replace(/\\\|/g, "\u0000");
        let cells = row.split("|");
        if (!cells[0].trim()) cells.shift();
        if (cells.length > 0 && !cells[cells.length - 1].trim()) cells.pop();

        if (count) {
            if (cells.length > count) {
                cells.splice(count);
            } else {
                while (cells.length < count) cells.push("");
            }
        }

        for (let i = 0; i < cells.length; i++) {
            cells[i] = cells[i].trim().replace(/\u0000/g, "|");
        }
        return cells;
    }
};
