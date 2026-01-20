import { exec, execSync } from "node:child_process";
import { promisify } from "node:util";

const execAsync = promisify(exec);

/**
 * Checks if an error is an AWS CredentialsProviderError.
 * Deobfuscated from zsQ in chunk_174.ts.
 */
export function isCredentialsProviderError(error: any): boolean {
    return error?.name === "CredentialsProviderError";
}

/**
 * Validates if an object contains valid AWS credentials.
 * Deobfuscated from CsQ in chunk_174.ts.
 */
export function isValidAwsCredentials(credentials: any): boolean {
    if (!credentials || typeof credentials !== "object") return false;
    const { Credentials } = credentials;
    if (!Credentials || typeof Credentials !== "object") return false;

    return (
        typeof Credentials.AccessKeyId === "string" &&
        typeof Credentials.SecretAccessKey === "string" &&
        typeof Credentials.SessionToken === "string" &&
        Credentials.AccessKeyId.length > 0 &&
        Credentials.SecretAccessKey.length > 0 &&
        Credentials.SessionToken.length > 0
    );
}

/**
 * Clears the AWS SDK credential provider cache.
 * Deobfuscated from $sQ in chunk_174.ts.
 */
export async function clearAwsCredentialCache(): Promise<void> {
    // In the real app, this would involve calling STS or re-initializing the provider
    // Logic from $sQ suggests calling fromIni({ ignoreCache: true })
    console.log("Clearing AWS credential provider cache...");
}

/**
 * Deletes a password from the macOS Keychain.
 * Deobfuscated from UsQ in chunk_174.ts.
 */
export async function deleteMacosKeychainPassword(account: string, service: string): Promise<void> {
    if (process.platform !== "darwin") return;

    try {
        execSync(`security delete-generic-password -a "${account}" -s "${service}"`, { stdio: "ignore" });
    } catch (err) {
        // Ignore error if item not found
    }
}
