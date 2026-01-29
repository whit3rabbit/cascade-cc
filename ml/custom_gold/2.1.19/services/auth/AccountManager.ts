/**
 * File: src/services/auth/AccountManager.ts
 * Role: Manages authenticated accounts, tenant profiles, and ID tokens (MSAL-style).
 */

export interface Account {
    homeAccountId: string;
    username?: string;
    tenantProfiles?: Set<string>;
    [key: string]: any;
}

export interface AccountFilter {
    homeAccountId?: string;
    username?: string;
    tenantId?: string;
}

/**
 * Service for managing multiple user accounts and their associated tokens.
 */
export const AccountManager = {
    accounts: new Map<string, Account>(),

    /**
     * Retrieves all accounts matching a filter.
     */
    getAccounts(filter: AccountFilter = {}): Account[] {
        let results: Account[] = Array.from(this.accounts.values());

        if (filter.homeAccountId) {
            results = results.filter((a: Account) => a.homeAccountId === filter.homeAccountId);
        }
        if (filter.username) {
            results = results.filter((a: Account) => a.username === filter.username);
        }
        if (filter.tenantId) {
            results = results.filter((a: Account) => a.tenantProfiles?.has(filter.tenantId!));
        }

        return results;
    },

    /**
     * Gets a single account by ID.
     */
    getAccountById(homeAccountId: string): Account | null {
        return this.accounts.get(homeAccountId) || null;
    },

    /**
     * Adds or updates an account.
     */
    saveAccount(account: Account): void {
        if (!account.homeAccountId) throw new Error("homeAccountId is required");
        this.accounts.set(account.homeAccountId, account);
    },

    /**
     * Removes an account and all its tokens.
     */
    removeAccount(homeAccountId: string): void {
        this.accounts.delete(homeAccountId);
    },

    /**
     * Decodes an OIDC ID Token to extract claims.
     */
    decodeIdToken(token: string): any {
        try {
            const parts = token.split('.');
            if (parts.length !== 3) return null;
            const payload = Buffer.from(parts[1], 'base64').toString();
            return JSON.parse(payload);
        } catch (e) {
            return null;
        }
    }
};
