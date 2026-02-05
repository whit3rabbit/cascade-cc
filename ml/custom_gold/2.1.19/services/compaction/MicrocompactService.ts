/**
 * File: src/services/compaction/MicrocompactService.ts
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

// Constants and Thresholds based on chunk1081
const PREFIX = '󱐋 '; // TP1 prefix
const VIEW_TOOL = 'FileRead'; // Tool to suggest for viewing
const WINDOW_SIZE = 3; // mW2 - keep last 3 tool results uncompacted
const WARNING_THRESHOLD = 20000; // BW2
const AUTO_COMPACT_LIMIT = 40000; // uW2
const TOKEN_ESTIMATION_MULTIPLIER = 1.3333333333333333;

// Tools subject to micro-compaction (based on gW2)
const TOOLS_TO_COMPACT = new Set([
    'FileRead',
    'Glob',
    'Grep',
    'WebFetch',
    'WebSearch',
    'Lsp',
    'Bash'
]);

export class MicrocompactService {
    private compactedToolUseIds: Set<string> = new Set();
    private tempDir: string;

    constructor() {
        this.tempDir = path.join(os.tmpdir(), 'claude-code-microcompact');
        if (!fs.existsSync(this.tempDir)) {
            fs.mkdirSync(this.tempDir, { recursive: true });
        }
    }

    private async saveToolResultToFile(content: string, toolUseId: string): Promise<string> {
        const filename = `${toolUseId}.txt`;
        const filepath = path.join(this.tempDir, filename);
        fs.writeFileSync(filepath, content);
        return filepath;
    }

    private estimateTokens(text: string): number {
        return Math.ceil(text.length * TOKEN_ESTIMATION_MULTIPLIER);
    }

    async fetchContextData(messages: any[], trigger: 'auto' | 'manual' = 'auto'): Promise<{ messages: any[], boundaryMessage?: string }> {
        if (process.env.DISABLE_MICROCOMPACT) {
            return { messages };
        }

        const toolIdsFound: string[] = [];
        const toolIdToTokens: Map<string, number> = new Map();
        const toolIdToResult: Map<string, any> = new Map();

        // 1. Scan for eligible tool results
        for (const msg of messages) {
            const content = msg.message?.content || msg.content;
            if (!Array.isArray(content)) continue;

            for (const item of content) {
                if (item.type === 'tool_use' && TOOLS_TO_COMPACT.has(item.name)) {
                    if (!this.compactedToolUseIds.has(item.id)) {
                        toolIdsFound.push(item.id);
                    }
                } else if (item.type === 'tool_result' && toolIdsFound.includes(item.toolUseId || item.tool_use_id)) {
                    // Normalize toolUseId
                    const id = item.toolUseId || item.tool_use_id;
                    const tokens = this.estimateTokens(JSON.stringify(item.content));
                    toolIdToTokens.set(id, tokens);
                    toolIdToResult.set(id, item);
                }
            }
        }

        // 2. Identify results to compact (Gold Reference Logic)
        const slidingWindow = toolIdsFound.slice(-WINDOW_SIZE);
        const totalEligibleTokens = Array.from(toolIdToTokens.values()).reduce((a, b) => a + b, 0);

        let tokensSaved = 0;
        const idsToCompact = new Set<string>();

        // Compact oldest first until remaining tokens <= AUTO_COMPACT_LIMIT
        for (const id of toolIdsFound) {
            if (slidingWindow.includes(id)) continue;

            if ((totalEligibleTokens - tokensSaved) > AUTO_COMPACT_LIMIT) {
                idsToCompact.add(id);
                tokensSaved += toolIdToTokens.get(id) || 0;
            }
        }

        // 3. Safety Check: If auto-mode, ignore if savings are small or total conversation is small
        if (trigger === 'auto') {
            if (tokensSaved < WARNING_THRESHOLD) {
                idsToCompact.clear();
                tokensSaved = 0;
            }
        }

        if (idsToCompact.size === 0) {
            return { messages };
        }

        // 4. Apply compaction
        let compactionOccurred = false;
        const newMessages = await Promise.all(messages.map(async (msg) => {
            const content = msg.message?.content || msg.content;
            if (!Array.isArray(content)) return msg;

            // Deep clone to avoid mutating original
            const contentArr = [...content];
            let changed = false;

            const newContent = await Promise.all(contentArr.map(async (item) => {
                const id = item.toolUseId || item.tool_use_id;
                if (item.type === 'tool_result' && id && idsToCompact.has(id)) {
                    const alreadyCompacted = this.compactedToolUseIds.has(id);
                    if (alreadyCompacted) return item;

                    compactionOccurred = true;
                    this.compactedToolUseIds.add(id);
                    const filepath = await this.saveToolResultToFile(JSON.stringify(item.content), id);

                    changed = true;
                    return {
                        ...item,
                        content: `${PREFIX}Tool result saved to: ${filepath}\n\nUse ${VIEW_TOOL} to view`
                    };
                }
                return item;
            }));

            if (!changed) return msg;

            const updatedMsg = { ...msg };
            if (updatedMsg.message) {
                updatedMsg.message = { ...updatedMsg.message, content: newContent };
            } else {
                updatedMsg.content = newContent;
            }
            return updatedMsg;
        }));

        if (compactionOccurred) {
            // Boundary marker mimicking chunk1124 clearConversation_16
            const boundaryMessage = {
                type: "system",
                subtype: "microcompact_boundary",
                content: "Context microcompacted",
                isMeta: false,
                timestamp: new Date().toISOString(),
                uuid: Math.random().toString(36).substring(7),
                level: "info",
                microcompactMetadata: {
                    trigger,
                    preTokens: totalEligibleTokens,
                    tokensSaved,
                    compactedToolIds: Array.from(idsToCompact),
                    clearedAttachmentUUIDs: []
                }
            };

            return {
                messages: newMessages,
                boundaryMessage: JSON.stringify(boundaryMessage)
            };
        }

        return { messages };
    }
}

export const microcompactService = new MicrocompactService();
