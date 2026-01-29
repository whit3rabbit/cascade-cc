/**
 * File: src/commands/pr_comments.ts
 * Role: Implementation of the /pr-comments command for fetching GitHub review comments.
 */

import { createCommandHelper, CommandContext } from './helpers.js';

/**
 * Command definition for fetching pull request comments from GitHub.
 * 
 * Uses the GitHub CLI (gh) to retrieve PR-level and review comments.
 */
export const prCommentsCommandDefinition = createCommandHelper("pr-comments", "Fetch and display comments from a GitHub pull request", {
    async getPromptForCommand(userInput: string, _context: CommandContext) {
        return [
            {
                type: "text",
                text: `You are an AI assistant integrated into a git-based version control system. Your task is to fetch and display comments from a GitHub pull request.

Workflow:
1.  **Identify PR**: Use \`gh pr view --json number,headRepository\` to get the PR number and repository info.
2.  **Fetch Comments**:
    -   Use \`gh api /repos/{owner}/{repo}/issues/{number}/comments\` for PR-level comments.
    -   Use \`gh api /repos/{owner}/{repo}/pulls/{number}/comments\` for review (inline) comments.
3.  **Context Enrichment**: For review comments, pay attention to \`body\`, \`diff_hunk\`, \`path\`, and \`line\`. If the comment references specific code, prefer to fetch that context for clarity.
4.  **Format Output**: Return a clean, readable summary of all comments, grouped by thread.

Formatting Template:
## Comments

[For each thread:]
- @author path/to/file.ext#line:
  \`\`\`diff
  [diff_hunk]
  \`\`\`
  > [comment body]
  
  (Indented replies if any)

If no comments are found, return "No comments found."

Rules:
- Only return the formatted comments, no extra chat.
- Preserved threading is essential.
- Use jq to parse API responses where needed.

${userInput ? `Additional User Request: ${userInput}` : ""}
`
            }
        ];
    },
    userFacingName() {
        return "pr-comments";
    }
});
