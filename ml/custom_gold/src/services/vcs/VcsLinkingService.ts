import { exec } from "child_process";
import { promisify } from "util";
import axios from "axios";
import { log } from "../logger/loggerService.js";

const execAsync = promisify(exec);
const logger = log("VcsLinkingService");

// Logic from chunk_441.ts (VCS Account Linking)

// tS5
export async function getGitHubUser(): Promise<{ username: string, hostname: string } | null> {
    try {
        const { stdout } = await execAsync("gh auth status --active --json hosts", { timeout: 5000 });
        if (!stdout.trim()) return null;

        const data = JSON.parse(stdout);
        const hosts = data.hosts;
        if (!hosts || typeof hosts !== "object") return null;

        for (const [host, accounts] of Object.entries(hosts)) {
            if (Array.isArray(accounts) && accounts.length > 0) {
                const acc: any = accounts[0];
                if (acc && acc.login) {
                    return { username: acc.login, hostname: host };
                }
            }
        }
        return null;
    } catch (error) {
        logger.debug(`Error getting GitHub user: ${error}`);
        return null;
    }
}

// eS5
export async function getGitEmail(): Promise<string | null> {
    try {
        const { stdout } = await execAsync("git config --get user.email", { timeout: 5000 });
        const text = stdout.trim();
        return text || null;
    } catch {
        return null;
    }
}

// Gj2
export async function linkVcsAccount(username: string, hostname: string, email: string): Promise<void> {
    try {
        // TODO: Get auth headers from AuthService properly
        // let authHeaders = getAuthHeaders(); // Helper to be implemented
        // For now, stubbing or assuming implicit auth won't work without it.
        // The original code uses aY() to get headers.

        const headers = {
            "Content-Type": "application/json",
            "User-Agent": "Claude-Code-Deobfuscated-Client", // ZW()
            // ...authHeaders
        };

        const payload = {
            vcs_type: "github",
            vcs_host: hostname,
            vcs_username: username,
            git_user_email: email
        };

        const url = "https://api.anthropic.com/api/claude_code/link_vcs_account";

        await axios.post(url, payload, {
            headers,
            timeout: 5000
        });

        logger.info(`Successfully linked VCS account: ${username}@${hostname}`);
    } catch (error) {
        // Silent fail is typical for this kind of background sync
        logger.debug(`Failed to link VCS account: ${error}`);
    }
}

// HH0
export async function checkAndLinkVcsAccount() {
    // 1. Check if linking is enabled/appropriate (uZ, q9, ty, jY1)
    // Simplified checks:
    if (process.env.CI) return;

    // 2. Gather data
    const [githubUser, gitEmail] = await Promise.all([getGitHubUser(), getGitEmail()]);

    // 3. Link if data exists
    if (githubUser || gitEmail) {
        await linkVcsAccount(
            githubUser?.username ?? "",
            githubUser?.hostname ?? "",
            gitEmail ?? ""
        );
    }
}
