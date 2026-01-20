
export function formatPluginError(error: any): string {
    if (!error || typeof error !== 'object') return String(error);

    switch (error.type) {
        case "generic-error": return error.error;
        case "path-not-found": return `Path not found: ${error.path} (${error.component})`;
        case "git-auth-failed": return `Git authentication failed (${error.authType}): ${error.gitUrl}`;
        case "git-timeout": return `Git ${error.operation} timeout: ${error.gitUrl}`;
        case "network-error": return `Network error: ${error.url}${error.details ? ` - ${error.details}` : ""}`;
        case "manifest-parse-error": return `Manifest parse error: ${error.parseError}`;
        case "manifest-validation-error": return `Manifest validation failed: ${error.validationErrors.join(", ")}`;
        case "plugin-not-found": return `Plugin ${error.pluginId} not found in marketplace ${error.marketplace}`;
        case "marketplace-not-found": return `Marketplace ${error.marketplace} not found`;
        case "marketplace-load-failed": return `Marketplace ${error.marketplace} failed to load: ${error.reason}`;
        // ... add others from PR
        default: return error.message || String(error);
    }
}
