/**
 * File: src/services/auth/azureCredentials.ts
 * Role: Implementation of Azure CLI and Developer CLI token credentials.
 */

import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFilePromise = promisify(execFile);

export interface AccessToken {
    token: string;
    expiresOnTimestamp: number;
}

/**
 * Validation stub for scopes.
 */
function validateScopes(scopes: string | string[], _logger: any): void {
    // Logic from official Azure SDK would go here
    if (!scopes || (Array.isArray(scopes) && scopes.length === 0)) {
        throw new Error("Scopes are required");
    }
}

/**
 * Map scopes to resource URLs for legacy Azure CLI.
 */
function getResourceForScope(scopes: string | string[]): string {
    const scope = Array.isArray(scopes) ? scopes[0] : scopes;
    return scope.replace(/\/.default$/, "");
}

/**
 * Azure CLI Credential implementation.
 */
export class AzureCliCredential {
    private tenantId?: string;
    private subscription?: string;
    private timeout?: number;

    constructor(options: { tenantId?: string; subscription?: string; processTimeoutInMs?: number } = {}) {
        this.tenantId = options.tenantId;
        this.subscription = options.subscription;
        this.timeout = options.processTimeoutInMs;
    }

    async getToken(scopes: string | string[]): Promise<AccessToken> {
        validateScopes(scopes, console);
        const resource = getResourceForScope(scopes);

        const args = ["account", "get-access-token", "--output", "json", "--resource", resource];
        if (this.tenantId) args.push("--tenant", this.tenantId);
        if (this.subscription) args.push("--subscription", `"${this.subscription}"`);

        try {
            const { stdout } = await execFilePromise("az", args, { timeout: this.timeout, shell: true });
            const parsed = JSON.parse(stdout);

            let expiresOn = Number.parseInt(parsed.expires_on, 10) * 1000;
            if (isNaN(expiresOn)) {
                expiresOn = new Date(parsed.expiresOn).getTime();
            }

            return {
                token: parsed.accessToken,
                expiresOnTimestamp: expiresOn
            };
        } catch (error: any) {
            if (error.stderr?.includes("az login")) {
                throw new Error("Please run 'az login' to authenticate.");
            }
            throw new Error(`Azure CLI failed: ${error.message}`);
        }
    }
}

/**
 * Azure Developer CLI (azd) Credential implementation.
 */
export class AzureDeveloperCliCredential {
    private tenantId?: string;
    private timeout?: number;

    constructor(options: { tenantId?: string; processTimeoutInMs?: number } = {}) {
        this.tenantId = options.tenantId;
        this.timeout = options.processTimeoutInMs;
    }

    async getToken(scopes: string | string[]): Promise<AccessToken> {
        const scopesArray = typeof scopes === "string" ? [scopes] : scopes;
        const args = ["auth", "token", "--output", "json"];

        if (this.tenantId) args.push("--tenant-id", this.tenantId);
        for (const s of scopesArray) args.push("--scope", s);

        try {
            const { stdout } = await execFilePromise("azd", args, { timeout: this.timeout });
            const parsed = JSON.parse(stdout);

            return {
                token: parsed.token,
                expiresOnTimestamp: new Date(parsed.expiresOn).getTime()
            };
        } catch (error: any) {
            if (error.stderr?.includes("azd login") || error.stderr?.includes("azd auth login")) {
                throw new Error("Please run 'azd auth login' to authenticate.");
            }
            throw new Error(`Azure Developer CLI failed: ${error.message}`);
        }
    }
}
