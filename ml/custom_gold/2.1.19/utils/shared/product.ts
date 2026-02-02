/**
 * File: src/utils/shared/product.ts
 * Role: Unified product naming and system user identification.
 */

import { createHash } from "node:crypto";

const BASE_PRODUCT_NAME = "Claude Code";

/**
 * Returns the product name, optionally specialized with a unique ID derived from the config path.
 * 
 * @param suffix - Optional suffix to append to the name.
 * @returns {string} The product name.
 */
export function getProductName(suffix = ""): string {
    const configDirEnv = process.env.CLAUDE_CONFIG_DIR;
    let uniqueId = "";

    if (configDirEnv) {
        uniqueId = "-" + createHash("sha256")
            .update(configDirEnv)
            .digest("hex")
            .substring(0, 8);
    }

    return `${BASE_PRODUCT_NAME}${suffix}${uniqueId}`;
}

/**
 * Returns the current OS user name using fallback environment variables.
 * 
 * @returns {string} The system user name.
 */
export function getSystemUser(): string {
    return process.env.USER || process.env.USERNAME || "claude-code-user";
}

/**
 * Returns the current product version.
 */
export function getProductVersion(): string {
    return "0.0.1"; // Placeholder
}
