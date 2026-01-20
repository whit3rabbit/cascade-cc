import { spawn } from "child_process";
// import { findExecutable } from ...

export async function checkGitInstalled(): Promise<boolean> {
    // Simplified logic
    return new Promise((resolve) => {
        const p = spawn("git", ["--version"]);
        p.on("close", (code) => {
            resolve(code === 0);
        });
        p.on("error", () => resolve(false));
    });
}

export interface GitResult {
    code: number;
    stdout: string;
    stderr: string;
}

export async function runGitCommand(args: string[], cwd: string, env: any = {}): Promise<GitResult> {
    return new Promise((resolve, reject) => {
        const p = spawn("git", args, { cwd, env: { ...process.env, ...env } });
        let stdout = "";
        let stderr = "";
        p.stdout.on("data", (d) => stdout += d.toString());
        p.stderr.on("data", (d) => stderr += d.toString());
        p.on("close", (code) => resolve({ code: code || 0, stdout, stderr }));
        p.on("error", (err) => reject(err));
    });
}

export async function updateGitRepository(repoPath: string, ref?: string): Promise<GitResult> {
    if (ref) {
        // Fetch specific ref logic
        const fetch = await runGitCommand(["fetch", "origin", ref], repoPath);
        if (fetch.code !== 0) return fetch;

        const checkout = await runGitCommand(["checkout", ref], repoPath);
        if (checkout.code !== 0) return checkout;

        const pull = await runGitCommand(["pull", "origin", "HEAD"], repoPath);
        return processGitError(pull);
    }

    const pull = await runGitCommand(["pull", "origin", "HEAD"], repoPath);
    return processGitError(pull);
}

export function processGitError(result: GitResult): GitResult {
    if (result.code !== 0 && result.stderr) {
        if (result.stderr.includes("Permission denied (publickey)") || result.stderr.includes("Could not read from remote repository")) {
            return { ...result, stderr: `SSH authentication failed. \nOriginal: ${result.stderr}` };
        }
        if (result.stderr.includes("timed out") || result.stderr.includes("Could not resolve host")) {
            return { ...result, stderr: `Network error. \nOriginal: ${result.stderr}` };
        }
    }
    return result;
}

export interface GitStatus {
    hasUncommitted: boolean;
    hasUnpushed: boolean;
    commitsAheadOfDefaultBranch: number;
}

export async function getGitStatus(cwd: string): Promise<GitStatus> {
    // Check for uncommitted
    const status = await runGitCommand(["status", "--porcelain"], cwd);
    const hasUncommitted = status.stdout.trim().length > 0;

    // Check for unpushed
    // Try getting upstream first
    const branchName = await getBranchName(cwd);
    let hasUnpushed = false;
    let commitsAhead = 0;

    if (branchName) {
        try {
            const unpushed = await runGitCommand(["log", "@{u}..HEAD", "--oneline"], cwd);
            if (unpushed.code === 0) {
                const lines = unpushed.stdout.trim().split("\n").filter(Boolean);
                hasUnpushed = lines.length > 0;
                commitsAhead = lines.length;
            }
        } catch (e) {
            // No upstream or other error, assume unpushed if commits exist?
            // Simplified logic: ignore for now
        }
    }

    return { hasUncommitted, hasUnpushed, commitsAheadOfDefaultBranch: commitsAhead };
}

export async function getBranchName(cwd: string): Promise<string> {
    const result = await runGitCommand(["rev-parse", "--abbrev-ref", "HEAD"], cwd);
    if (result.code !== 0) return "";
    return result.stdout.trim();
}

export async function getDefaultBranch(cwd: string): Promise<string> {
    // Simplified heuristic
    const branches = ["main", "master", "trunk", "development"];
    for (const branch of branches) {
        // check if branch exists locally
        const res = await runGitCommand(["rev-parse", "--verify", branch], cwd);
        if (res.code === 0) return branch;
    }
    return "main";
}
