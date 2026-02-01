import { spawn } from 'node:child_process';
import { getSettings } from '../config/SettingsService.js';
import { HookEvent, HookInput, HookOutput, HookOutputSchema, HookConfigEntry, HookInputSchema } from './HookTypes.js';

export class HookService {
    /**
     * Dispatches a hook event, executing all configured hooks for that event.
     */
    async dispatch(event: HookEvent, input: HookInput): Promise<HookOutput[]> {
        const settings = getSettings();
        if (!settings.hooks) {
            return [];
        }

        const hooks = settings.hooks[event];
        if (!hooks || hooks.length === 0) {
            return [];
        }

        if (settings.allowManagedHooksOnly) {
            // In a real implementation, we would filter hooks by source (policy vs user).
            // For now, we are just acknowledging the setting exists and is checked here.
            // console.warn("[HookService] allowManagedHooksOnly is enabled. Verifying hooks...");
        }

        // Validate input (optional, but good practice)
        const parseResult = HookInputSchema.safeParse(input);
        if (!parseResult.success) {
            console.warn(`[HookService] Invalid input for event ${event}:`, parseResult.error);
            // Proceeding with provided input anyway, or could return empty
        }

        const results: HookOutput[] = [];
        for (const hook of hooks) {
            try {
                const result = await this.executeHook(hook, input);
                results.push(result);

                // Check for control flow (stop/interruption)
                if (result.continue === false) {
                    // If one hook says stop, do we stop other hooks? 
                    // Reference suggests "continuation" usually refers to the main Claude flow, not necessarily the hook chain.
                    // But if 'decision' is 'block', we might want to flag it.
                }
            } catch (error) {
                console.error(`[HookService] Error executing hook for ${event}:`, error);
                results.push({
                    continue: true,
                    systemMessage: `Hook execution failed: ${error instanceof Error ? error.message : String(error)}`,
                    decision: 'approve' // Fail open by default? Or closed?
                });
            }
        }
        return results;
    }

    /**
     * Executes a single hook configuration.
     */
    private async executeHook(hookConfig: HookConfigEntry, input: HookInput): Promise<HookOutput> {
        const commandStr = hookConfig.command || (Array.isArray(hookConfig.commands) ? hookConfig.commands[0] : hookConfig.commands as string);
        if (!commandStr) {
            throw new Error("No command specified in hook configuration");
        }

        // Prepare input JSON
        const inputJson = JSON.stringify(input);

        return new Promise((resolve, reject) => {
            const timeoutMs = hookConfig.timeout ? hookConfig.timeout * 1000 : 10000; // Default 10s
            const abortController = new AbortController();
            const timeoutId = setTimeout(() => {
                abortController.abort();
            }, timeoutMs);

            const cwd = hookConfig.cwd || process.cwd();

            // Use shell to execute command
            const child = spawn(commandStr, [], {
                cwd,
                shell: true,
                signal: abortController.signal,
                stdio: ['pipe', 'pipe', 'pipe']
            });

            let stdout = '';
            let stderr = '';

            if (child.stdin) {
                child.stdin.write(inputJson);
                child.stdin.end();
            }

            child.stdout?.on('data', (data) => {
                stdout += data.toString();
            });

            child.stderr?.on('data', (data) => {
                stderr += data.toString();
            });

            child.on('close', (code) => {
                clearTimeout(timeoutId);
                if (code !== 0) {
                    // Hook failed?
                    console.warn(`[HookService] Hook exited with code ${code}. Stderr: ${stderr}`);
                    // We interpret non-zero as failure but maybe still try to parse output?
                    // If output is empty but code is non-zero, it's an error.
                    if (!stdout.trim()) {
                        resolve({
                            continue: true,
                            decision: 'approve',
                            systemMessage: `Hook exited with code ${code}: ${stderr}`
                        });
                        return;
                    }
                }

                // Parse Output
                try {
                    // Attempt to parse stdout as JSON
                    // Sometimes hooks might output raw text logs before JSON. 
                    // Reference implementation seems to look for JSON.
                    const jsonStart = stdout.indexOf('{');
                    const jsonEnd = stdout.lastIndexOf('}');
                    if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
                        const jsonStr = stdout.substring(jsonStart, jsonEnd + 1);
                        const parsed = JSON.parse(jsonStr);
                        const result = HookOutputSchema.safeParse(parsed);
                        if (result.success) {
                            resolve(result.data);
                        } else {
                            // Could not validate schema
                            console.warn(`[HookService] Invalid hook output schema:`, result.error);
                            resolve({
                                continue: true,
                                systemMessage: `Invalid hook output schema`,
                                decision: 'approve'
                            });
                        }
                    } else {
                        // No JSON found, treat as success/continue but log output?
                        resolve({
                            continue: true,
                            decision: 'approve',
                            // plainText: stdout // If we want to capture it
                        });
                    }
                } catch (e) {
                    console.error(`[HookService] Error parsing hook output:`, e);
                    resolve({
                        continue: true,
                        decision: 'approve',
                        systemMessage: `Error parsing hook output`
                    });
                }
            });

            child.on('error', (err) => {
                clearTimeout(timeoutId);
                if (err.name === 'AbortError') {
                    resolve({
                        continue: true,
                        decision: 'approve',
                        systemMessage: `Hook execution timed out after ${timeoutMs}ms`
                    });
                } else {
                    reject(err);
                }
            });
        });
    }
}

export const hookService = new HookService();
