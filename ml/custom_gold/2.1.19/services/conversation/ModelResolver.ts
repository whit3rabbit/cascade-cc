
import { EnvService } from '../config/EnvService.js';
import { DEFAULT_AGENT_MODEL } from '../../types/AgentTypes.js';

/**
 * Resolves the user-facing model alias (e.g., 'sonnet', 'opus') to the specific API model identifier.
 * Handles 'opusplan' logic and '[1m]' context windows.
 */
export class ModelResolver {

    /**
     * returns the actual model ID (e.g. claude-3-5-sonnet-20241022)
     */
    static resolveModel(modelAlias: string, isPlanMode: boolean): string {
        const cleanAlias = modelAlias.toLowerCase().trim();

        // Check for opusplan special case
        if (cleanAlias === 'opusplan') {
            if (isPlanMode) {
                return EnvService.get('ANTHROPIC_DEFAULT_OPUS_MODEL') || 'claude-3-opus-20240229';
            } else {
                return EnvService.get('ANTHROPIC_DEFAULT_SONNET_MODEL') || DEFAULT_AGENT_MODEL;
            }
        }

        // Check for direct environment variable mappings
        // This is a simplified lookup map as per docs
        const map: Record<string, string> = {
            'haiku': EnvService.get('ANTHROPIC_DEFAULT_HAIKU_MODEL') || 'claude-3-5-haiku-20241022',
            'sonnet': EnvService.get('ANTHROPIC_DEFAULT_SONNET_MODEL') || DEFAULT_AGENT_MODEL,
            'opus': EnvService.get('ANTHROPIC_DEFAULT_OPUS_MODEL') || 'claude-3-opus-20240229',
            'subagent': EnvService.get('CLAUDE_CODE_SUBAGENT_MODEL') || 'claude-3-5-haiku-20241022'
        };

        if (map[cleanAlias]) {
            return map[cleanAlias];
        }

        return cleanAlias; // return as-is if no alias match
    }

    /**
     * Determines if extended context '[1m]' suffix is present.
     */
    static isExtendedContext(modelAlias: string): boolean {
        return modelAlias.endsWith('[1m]');
    }

    /**
     * Strips the '[1m]' suffix for resolving the base model.
     */
    static stripExtendedSuffix(modelAlias: string): string {
        return modelAlias.replace(/\[1m\]$/, '');
    }
}
