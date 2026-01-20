
import React from 'react';
import { Box, Text } from 'ink';
import { performWebFetch, ALLOWED_DOMAINS } from '../../services/web/WebFetchService.js';
import { getGlobalState as getAppState } from '../../services/session/sessionStore.js';
import { z } from 'zod'; // Assuming zod is used via 'm'
import { ToolUseConfirm } from '../../components/permissions/ToolUseConfirm.js';
import { PermissionSelect } from '../../components/permissions/PermissionComponents.js';

// Input Schema (wi5)
const inputSchema = z.strictObject({
    url: z.string().url().describe("The URL to fetch content from"),
    prompt: z.string().describe("The prompt to run on the fetched content")
});

const outputSchema = z.object({
    bytes: z.number().describe("Size of the fetched content in bytes"),
    code: z.number().describe("HTTP response code"),
    codeText: z.string().describe("HTTP response code text"),
    result: z.string().describe("Processed result from applying the prompt to the content"),
    durationMs: z.number().describe("Time taken to fetch and process the content"),
    url: z.string().describe("The URL that was fetched")
});

export const FetchTool = {
    name: "Fetch",
    description: async (args: any) => {
        try {
            return `Claude wants to fetch content from ${new URL(args.url).hostname}`;
        } catch {
            return "Claude wants to fetch content from this URL";
        }
    },
    userFacingName: () => "Fetch",
    isEnabled: () => true,
    inputSchema,
    outputSchema,
    isConcurrencySafe: () => true,
    isReadOnly: () => true,

    // Permission Check (checkPermissions in chunk_476)
    async checkPermissions(input: any, context: any) {
        const globalState = getAppState();
        const toolPermissionContext = globalState.toolPermissionContext || {}; // stub

        try {
            const { url } = input;
            const u = new URL(url);
            const hostname = u.hostname;
            const pathname = u.pathname;

            for (const allowed of ALLOWED_DOMAINS) {
                if (allowed.includes('/')) {
                    const [domain, ...pathParts] = allowed.split('/');
                    const pathPrefix = '/' + pathParts.join('/');
                    if (hostname === domain && pathname.startsWith(pathPrefix)) {
                        return { behavior: 'allow', updatedInput: input, decisionReason: { type: 'other', reason: 'Preapproved host and path' } };
                    }
                } else if (hostname === allowed) {
                    return { behavior: 'allow', updatedInput: input, decisionReason: { type: 'other', reason: 'Preapproved host' } };
                }
            }
        } catch { }

        // Rule checks (ES logic) - placeholder
        // Using common logic or implementing here

        return { behavior: 'ask', message: `Claude requested permissions to use Fetch, but you haven't granted it yet.` };
    },

    async call(input: any, context: any) {
        const start = Date.now();
        const { url, prompt } = input;
        const result = await performWebFetch(url, context.abortController.signal);

        if (result.type === 'redirect') {
            // Handle redirect message
            return {
                data: {
                    bytes: Buffer.byteLength(result.redirectUrl), // approximate
                    code: result.statusCode,
                    codeText: "Redirect",
                    result: `REDIRECT DETECTED... (full message logic)`,
                    durationMs: Date.now() - start,
                    url
                }
            };
        }

        // Process prompt if needed (Rg2 in chunk)
        // For now return content
        return {
            data: {
                ...result,
                durationMs: Date.now() - start,
                url
            }
        };
    },

    // Render functions (jg2, Tg2, Pg2, Sg2, xg2)
    renderToolUseMessage: (input: any) => `Fetch ${input.url}`, // Simplified
    renderToolUseRejectedMessage: () => <Text color="red">Fetch rejected</Text>,
    renderToolUseErrorMessage: (err: any) => <Text color="red">Error: {JSON.stringify(err)}</Text>,
    renderToolUseProgressMessage: () => <Text dimColor>Fetching…</Text>,
    renderToolResultMessage: (res: any) => <Text>Received {res.bytes} bytes</Text>
};

// Permission View (kg2)
export function FetchToolPermissionView({ toolUseConfirm, onDone, onReject }: any) {
    const url = toolUseConfirm.input.url;
    const hostname = new URL(url).hostname;

    const options = [
        { label: "Yes", value: "yes" },
        { label: `Yes, and don't ask again for ${hostname}`, value: "yes-dont-ask-again-domain" },
        { label: "No", value: "no" }
    ];

    const handleChange = (value: string) => {
        if (value === 'yes') {
            toolUseConfirm.onAllow(toolUseConfirm.input, []);
            onDone();
        } else if (value === 'yes-dont-ask-again-domain') {
            // Add rule logic
            const rule = { toolName: 'Fetch', ruleContent: `domain:${hostname}` };
            toolUseConfirm.onAllow(toolUseConfirm.input, [{ type: 'addRules', rules: [rule], behavior: 'allow', destination: 'localSettings' }]);
            onDone();
        } else {
            toolUseConfirm.onReject();
            onReject();
        }
    };

    return (
        <Box flexDirection="column" paddingX={1} borderStyle="round" borderColor="gray">
            <Text bold>Fetch</Text>
            <Text>Claude wants to fetch content from {hostname}</Text>
            <Text dimColor>{toolUseConfirm.description}</Text>
            <Box marginTop={1}>
                <PermissionSelect
                    options={options}
                    onChange={handleChange}
                    onCancel={() => handleChange('no')}
                />
            </Box>
        </Box>
    );
}
