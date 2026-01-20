
// Logic from chunk_547.ts (GitHub Action Setup Service)

import { execSync } from "node:child_process";

/**
 * Executes a gh command and returns the result.
 */
async function runGhCommand(args: string[]) {
    try {
        const stdout = execSync(`gh ${args.join(" ")}`, { encoding: "utf8", stdio: "pipe" });
        return { code: 0, stdout, stderr: "" };
    } catch (err: any) {
        return {
            code: err.status || 1,
            stdout: err.stdout || "",
            stderr: err.stderr || String(err)
        };
    }
}

/**
 * Sets up GitHub Actions for a repository.
 */
export async function setupGitHubActions(options: {
    repo: string;
    branch: string;
    secretName: string;
    apiKey?: string;
    skipWorkflow: boolean;
    authType: string;
    selectedWorkflows: string[];
    telemetryContext?: any;
}) {
    const { repo, branch, secretName, apiKey, skipWorkflow, authType, selectedWorkflows, telemetryContext } = options;

    try {
        // 1. Set secret if provided
        if (apiKey) {
            const secretResult = await runGhCommand([
                "secret", "set", secretName,
                "--body", apiKey,
                "--repo", repo
            ]);

            if (secretResult.code !== 0) {
                const helpText = `
Need help? Common issues:
• Permission denied → Run: gh auth refresh -h github.com -s repo
• Not authorized → Ensure you have admin access to the repository
• For manual setup → Visit: https://github.com/anthropics/claude-code-action`;

                throw new Error(`Failed to set API key secret: ${secretResult.stderr || "Unknown error"}${helpText}`);
            }
        }

        // 2. Handle PR creation if not skipped
        if (!skipWorkflow && branch) {
            const prTitle = "Add Claude GitHub Action";
            const prBody = "This PR adds Claude Code to your GitHub Actions workflow for PR reviews and automated assistance.";
            const prUrl = `https://github.com/${repo}/compare/${branch}?quick_pull=1&title=${encodeURIComponent(prTitle)}&body=${encodeURIComponent(prBody)}`;

            // In actual implementation, we might use open(prUrl)
            console.log(`PR Page URL: ${prUrl}`);
        }

        return {
            success: true,
            repo,
            skipWorkflow,
            authType
        };
    } catch (err) {
        throw err;
    }
}
