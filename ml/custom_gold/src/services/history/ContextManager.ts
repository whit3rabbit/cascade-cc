
// Logic from chunk_446.ts (Context Management, Notes, Effort Level)

import { resolve, join } from "path";
// Stub imports

// Logic for Notes Template (Fy5)
export const NOTES_TEMPLATE = `
# Session Title
_A short and distinctive 5-10 word descriptive title for the session. Super info dense, no filler_

# Current State
_What is actively being worked on right now? Pending tasks not yet completed. Immediate next steps._

# Task specification
_What did the user ask to build? Any design decisions or other explanatory context_

# Files and Functions
_What are the important files? In short, what do they contain and why are they relevant?_

# Workflow
_What bash commands are usually run and in what order? How to interpret their output if not obvious?_

# Errors & Corrections
_Errors encountered and how they were fixed. What did the user correct? What approaches failed and should not be tried again?_

# Codebase and System Documentation
_What are the important system components? How do they work/fit together?_

# Learnings
_What has worked well? What has not? What to avoid? Do not duplicate items from other sections_

# Key results
_If the user asked a specific output such as an answer to a question, a table, or other document, repeat the exact result here_

# Worklog
_Step by step, what was attempted, done? Very terse summary for each step_
`;

export function getPromptTemplate() {
    return "IMPORTANT: This message and these instructions are NOT part of the actual user conversation...";
}

// Logic for Effort Level (hH0)
export function getEffortLevel() {
    const fromEnv = process.env.CLAUDE_CODE_EFFORT_LEVEL;
    if (fromEnv) {
        if (fromEnv === 'unset') return undefined;
        const num = parseInt(fromEnv, 10);
        if (!isNaN(num) && Number.isInteger(num)) return num;
        if (['low', 'medium', 'high'].includes(fromEnv)) return fromEnv;
    }
    // Stub for config check
    return undefined;
}

// Logic for Context Management (T9A)
export function calculateContextUsage(totalTokens: number) {
    // Stub
    const limit = 200000; // Mock limit
    const percentLeft = Math.max(0, Math.round((limit - totalTokens) / limit * 100));
    return {
        percentLeft,
        isAboveWarningThreshold: totalTokens > limit * 0.8,
        isAboveErrorThreshold: totalTokens > limit * 0.95,
        isAboveAutoCompactThreshold: totalTokens > limit * 0.7
    };
}

export function shouldAutoCompact(totalTokens: number) {
    const usage = calculateContextUsage(totalTokens);
    return usage.isAboveAutoCompactThreshold;
}
