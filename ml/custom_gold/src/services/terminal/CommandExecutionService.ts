
import { getGlobalState as getAppState, getOriginalCwd } from '../session/sessionStore.js';
import { evaluateBashCommandSafety } from '../bash/BashSafetyService.js';
import { validateCommandWithRules } from '../validation/CommandRuleValidator.js';
// import { getOriginalCwd } from '../../utils/shared/pathUtils.js'; // Moved to sessionStore

// Types corresponding to internal Permission structures
interface PermissionResult {
    behavior: 'allow' | 'ask' | 'deny' | 'passthrough';
    message?: string;
    decisionReason?: any;
    updatedInput?: any;
    suggestions?: string[];
}

interface CommandInput {
    command: string;
    // other fields?
}

// Logic from chunk_465.ts
export class CommandExecutionService {
    // OE0
    async evaluateCommand(input: CommandInput, context: any, executor: (cmd: string) => Promise<any>): Promise<PermissionResult> {
        const appState = getAppState();
        const { command } = input;

        // sI: Parse checks
        // We use a simplified check here or import a robust parser
        // For now, assume basic non-empty check or implementation from BashSafetyService
        if (!command || command.trim() === '') {
            // Basic pass
        }

        // 1. Check if sandboxing is enabled and auto-allow logic (am5)
        // if (_B.isSandboxingEnabled() && _B.isAutoAllowBashIfSandboxedEnabled() && mHA(A)) ...
        // Implementing basic check:
        // const safety = evaluateBashCommandSafety(command);
        // if (safety.behavior !== 'passthrough') return safety;

        // 2. LE0: Global deny/allow lists
        // const ruleResult = validateCommandAgainstRules(command);
        // if (ruleResult.behavior === 'deny') return ruleResult;

        // 3. Subcommand breakdown ($k2 loop)
        // Recursively check piped commands if needed.

        // 4. Check for multiple directory changes (cd ... && cd ...)
        const subCommands = this.extractSubCommands(command);
        const cdCommands = subCommands.filter(c => c.trim().startsWith('cd '));
        if (cdCommands.length > 1) {
            return {
                behavior: 'ask',
                decisionReason: { type: 'other', reason: "Multiple directory changes in one command require approval for clarity" },
                message: "Multiple directory changes in one command require approval for clarity"
            };
        }

        // 5. Validate individual commands against rules (nm5, fk2)
        const validations = subCommands.map(cmd => validateCommandWithRules(cmd, context, appState)); // Assuming context/state needed
        if (validations.some(v => v.behavior === 'deny')) {
            return {
                behavior: 'deny',
                message: `Permission to use command ${command} has been denied.`
            };
        }

        // 6. Check for dangerous patterns (gX1, qd)
        // const dangerousPatternCheck = evaluateBashCommandSafety(command);
        // if (dangerousPatternCheck.behavior !== 'passthrough') return dangerousPatternCheck;

        // 7. If all passed, return allow or ask based on cumulative results
        if (validations.every(v => v.behavior === 'allow')) {
            return { behavior: 'allow', updatedInput: input };
        }

        // Default ask behavior if nothing explicitly allowed or denied but safety checks passed?
        // Or passthrough to let the user decide?
        // For now, aligning with the chunk logic which falls back to 'ask' or executes if safe.

        return { behavior: 'ask', updatedInput: input };
    }

    private extractSubCommands(command: string): string[] {
        // Simplified split by &&, ;, |
        // Real implementation should use tree-sitter or tokenizer
        return command.split(/&&|;|\|/).map(s => s.trim()).filter(s => s.length > 0);
    }
}

export const commandExecutionService = new CommandExecutionService();
