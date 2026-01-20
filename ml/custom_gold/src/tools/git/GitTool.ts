
import { z } from "zod";
import { runBashCommand } from "../../services/terminal/BashExecutor.js";
import { Tool } from "../definitions/tool.js";

export const GitTool: Tool = {
    name: "git",
    description: async () => "Performs git operations (log, diff, status, commit, etc.)",
    inputSchema: z.object({
        action: z.string().describe("Git action to perform (e.g. status, log, diff)"),
        args: z.array(z.string()).optional().describe("Arguments for the git action")
    }),
    userFacingName: () => "Git",
    isEnabled: () => true,
    isConcurrencySafe: () => false,
    isReadOnly: () => false,
    prompt: async () => "Perform a git operation. Use this tool to check status, diff, log, commit, etc. For complex operations, consider using bash directly.",
    isSearchOrReadCommand: () => ({ isSearch: false, isRead: false }),
    async call(input: { action: string, args?: string[] }, _context: any) {
        const { action, args = [] } = input;

        // Prevent dangerous commands? 
        // For now, we trust the sandbox policies managed in GitPolicyService,
        // but we should at least block 'push' without confirmation if possible.
        // But for this direct tool implementation, we'll just forward to bash.

        const command = `git ${action} ${args.join(" ")}`;

        try {
            const { result } = await runBashCommand(command);
            const { stdout, stderr, code } = (await result) as any;

            if (code !== 0) {
                return {
                    is_error: true,
                    content: `Git command failed with exit code ${code}:\n${stderr}\n${stdout}`
                };
            }

            return {
                data: {
                    success: true,
                    action,
                    output: stdout
                }
            };
        } catch (error: any) {
            return {
                is_error: true,
                content: `Error executing git command: ${error.message}`
            };
        }
    },

    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        if (result.is_error) {
            return {
                tool_use_id: toolUseId,
                type: "tool_result",
                content: result.content,
                is_error: true
            };
        }
        return {
            tool_use_id: toolUseId,
            type: "tool_result",
            content: result.data.output
        };
    }
}
