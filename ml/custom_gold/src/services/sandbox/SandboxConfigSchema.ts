import { z } from "zod";

export const NetworkConfigSchema = z.object({
    allowedDomains: z.array(z.string()).optional(),
    allowUnixSockets: z.array(z.string()).optional(),
    allowAllUnixSockets: z.boolean().optional(),
    allowLocalBinding: z.boolean().optional(),
    httpProxyPort: z.number().optional(),
    socksProxyPort: z.number().optional()
}).optional();

export const SandboxConfigSchema = z.object({
    enabled: z.boolean().optional(),
    autoAllowBashIfSandboxed: z.boolean().optional(),
    allowUnsandboxedCommands: z.boolean().optional().describe("Allow commands to run outside the sandbox via the dangerouslyDisableSandbox parameter. When false, the dangerouslyDisableSandbox parameter is completely ignored and all commands must run sandboxed. Default: true."),
    network: NetworkConfigSchema,
    ignoreViolations: z.record(z.string(), z.array(z.string())).optional(),
    enableWeakerNestedSandbox: z.boolean().optional(),
    excludedCommands: z.array(z.string()).optional(),
    ripgrep: z.object({
        command: z.string(),
        args: z.array(z.string()).optional()
    }).optional().describe("Custom ripgrep configuration for bundled ripgrep support")
}).passthrough();

export type SandboxConfig = z.infer<typeof SandboxConfigSchema>;
