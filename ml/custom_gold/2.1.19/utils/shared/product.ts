/**
 * File: src/utils/shared/product.ts
 * Role: Unified product naming and system user identification.
 */

import { createHash } from "node:crypto";
import { PRODUCT_NAME } from "../../constants/product.js";
import { getOAuthConfig } from "../../constants/oauth.js";
import { BUILD_INFO } from "../../constants/build.js";

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

    const oauthSuffix = getOAuthConfig().OAUTH_FILE_SUFFIX;
    return `${PRODUCT_NAME}${oauthSuffix}${suffix}${uniqueId}`;
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
    return BUILD_INFO.VERSION;
}
