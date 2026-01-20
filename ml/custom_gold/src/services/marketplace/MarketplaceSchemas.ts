
import { z } from "zod";

export const ANTHROPIC_ORG = "anthropics";

export const PluginManifestSchema = z.object({
    name: z.string().min(1, "Plugin name cannot be empty").refine((val) => !val.includes(" "), {
        message: 'Plugin name cannot contain spaces. Use kebab-case (e.g., "my-plugin")'
    }).describe("Unique identifier for the plugin, used for namespacing (prefer kebab-case)"),
    version: z.string().optional().describe("Semantic version (e.g., 1.2.3) following semver.org specification"),
    description: z.string().optional().describe("Brief, user-facing explanation of what the plugin provides"),
    homepage: z.string().url().optional().describe("Plugin homepage or documentation URL"),
    repository: z.string().optional().describe("Source code repository URL"),
    license: z.string().optional().describe("SPDX license identifier (e.g., MIT, Apache-2.0)"),
    keywords: z.array(z.string()).optional().describe("Tags for plugin discovery and categorization")
});

export const MarketplacePluginEntrySchema = PluginManifestSchema.partial().extend({
    name: z.string().min(1, "Plugin name cannot be empty").describe("Unique identifier matching the plugin name"),
    source: z.union([
        z.string().describe("Relative path to plugin directory inside the marketplace"),
        z.object({
            source: z.literal("github"),
            repo: z.string(),
            ref: z.string().optional(),
            path: z.string().optional()
        }),
        z.object({
            source: z.literal("git"),
            url: z.string(),
            ref: z.string().optional(),
            path: z.string().optional()
        }),
        z.object({
            source: z.literal("url"),
            url: z.string(),
            headers: z.record(z.string(), z.string()).optional()
        }),
        z.object({
            source: z.literal("npm"),
            package: z.string()
        })
    ]).describe("Source location of the plugin"),
    category: z.string().optional().describe('Category for organizing plugins'),
    tags: z.array(z.string()).optional().describe("Tags for searchability and discovery"),
    strict: z.boolean().optional().default(true).describe("Require the plugin manifest to be present in the plugin folder")
}).strict();

export const MarketplaceManifestSchema = z.object({
    name: z.string().min(1, "Marketplace must have a name"),
    plugins: z.array(MarketplacePluginEntrySchema).describe("Collection of available plugins in this marketplace"),
    metadata: z.object({
        pluginRoot: z.string().optional().describe("Base path for relative plugin sources"),
        version: z.string().optional().describe("Marketplace version"),
        description: z.string().optional().describe("Marketplace description")
    }).optional().describe("Optional marketplace metadata")
});

export const PluginInstallationV1Schema = z.object({
    version: z.string().describe("Currently installed version"),
    installedAt: z.string().describe("ISO 8601 timestamp of installation"),
    lastUpdated: z.string().optional().describe("ISO 8601 timestamp of last update"),
    installPath: z.string().describe("Absolute path to the installed plugin directory"),
    gitCommitSha: z.string().optional().describe("Git commit SHA for git-based plugins (for version tracking)"),
    isLocal: z.boolean().optional().describe("True if plugin is local (in marketplace directory). Local plugins should not be deleted on uninstall.")
});

export const InstalledPluginsV1Schema = z.object({
    version: z.literal(1).describe("Schema version 1"),
    plugins: z.record(z.string(), PluginInstallationV1Schema).describe("Map of plugin IDs to their installation metadata")
});

export const PluginInstallationV2Schema = z.object({
    scope: z.enum(["managed", "user", "project", "local"]).describe("Installation scope"),
    projectPath: z.string().optional().describe("Project path (required for project/local scopes)"),
    installPath: z.string().describe("Absolute path to the versioned plugin directory"),
    version: z.string().optional().describe("Currently installed version"),
    installedAt: z.string().optional().describe("ISO 8601 timestamp of installation"),
    lastUpdated: z.string().optional().describe("ISO 8601 timestamp of last update"),
    gitCommitSha: z.string().optional().describe("Git commit SHA for git-based plugins"),
    isLocal: z.boolean().optional().describe("True if plugin is in marketplace directory")
});

export const InstalledPluginsV2Schema = z.object({
    version: z.literal(2).describe("Schema version 2"),
    plugins: z.record(z.string(), z.array(PluginInstallationV2Schema)).describe("Map of plugin IDs to arrays of installation entries")
});

export const InstalledPluginsDbSchema = z.union([InstalledPluginsV1Schema, InstalledPluginsV2Schema]);

export const MarketplaceSourceSchema = z.discriminatedUnion("source", [
    z.object({
        source: z.literal("github"),
        repo: z.string().describe("GitHub repository (user/repo)"),
        ref: z.string().optional().describe("Git reference (branch, tag, commit)"),
        path: z.string().optional().describe("Path within the repo to the marketplace file")
    }),
    z.object({
        source: z.literal("git"),
        url: z.string().describe("Git repository URL"),
        ref: z.string().optional().describe("Git reference (branch, tag, commit)"),
        path: z.string().optional().describe("Path within the repo to the marketplace file")
    }),
    z.object({
        source: z.literal("url"),
        url: z.string().url().describe("Direct URL to the marketplace JSON file"),
        headers: z.record(z.string(), z.string()).optional().describe("Optional HTTP headers")
    }),
    z.object({
        source: z.literal("file"),
        path: z.string().describe("Local file path to the marketplace JSON file")
    }),
    z.object({
        source: z.literal("directory"),
        path: z.string().describe("Local directory path containing .claude-plugin/marketplace.json")
    }),
    z.object({
        source: z.literal("npm"),
        package: z.string().describe("NPM package name")
    })
]).describe("Source configuration for a marketplace");

export type MarketplaceSource = z.infer<typeof MarketplaceSourceSchema>;
