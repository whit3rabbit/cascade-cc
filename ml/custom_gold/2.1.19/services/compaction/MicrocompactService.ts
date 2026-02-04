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
    'Lsp'
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

    async fetchContextData(messages: any[]): Promise<{ messages: any[], boundaryMessage?: string }> {
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
                } else if (item.type === 'tool_result' && toolIdsFound.includes(item.tool_use_id)) {
                    const tokens = this.estimateTokens(JSON.stringify(item.content));
                    toolIdToTokens.set(item.tool_use_id, tokens);
                    toolIdToResult.set(item.tool_use_id, item);
                }
            }
        }

        // 2. Identify results to compact using sliding window
        // Keep the last WINDOW_SIZE results untouched.
        const slidingWindow = toolIdsFound.slice(-WINDOW_SIZE);
        const totalUncompactedTokens = Array.from(toolIdToTokens.values()).reduce((a, b) => a + b, 0);

        let tokensSaved = 0;
        const idsToCompact = new Set<string>();

        // Logic from chunk1081: Compact if total savings > threshold OR (in auto mode) if above warning threshold
        // Here we simplify to: compact if not in sliding window and we save tokens.
        for (const id of toolIdsFound) {
            if (slidingWindow.includes(id)) continue;

            // In chunk1081, it checks if (X - O > z) where z is 40k.
            // Simplified: If we found many tokens, compact old ones.
            idsToCompact.add(id);
            tokensSaved += toolIdToTokens.get(id) || 0;
        }

        // Only compact if it significantly reduces context or we are above warning threshold
        if (idsToCompact.size === 0 || (totalUncompactedTokens < WARNING_THRESHOLD && idsToCompact.size < 5)) {
            return { messages };
        }

        // 3. Apply compaction
        let compactionOccurred = false;
        const newMessages = await Promise.all(messages.map(async (msg) => {
            const content = msg.message?.content || msg.content;
            if (!Array.isArray(content)) return msg;

            const newContent = await Promise.all(content.map(async (item) => {
                if (item.type === 'tool_result' && idsToCompact.has(item.tool_use_id)) {
                    const alreadyCompacted = this.compactedToolUseIds.has(item.tool_use_id);
                    if (alreadyCompacted) return item; // Safety, though idsToCompact excludes them

                    compactionOccurred = true;
                    this.compactedToolUseIds.add(item.tool_use_id);
                    const filepath = await this.saveToolResultToFile(JSON.stringify(item.content), item.tool_use_id);

                    return {
                        ...item,
                        content: `${PREFIX}Tool result saved to: ${filepath}\n\nUse ${VIEW_TOOL} to view`
                    };
                }
                return item;
            }));

            const updatedMsg = { ...msg };
            if (updatedMsg.message) {
                updatedMsg.message.content = newContent;
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
                    trigger: "auto", // default for background microcompact
                    preTokens: Array.from(toolIdToTokens.values()).reduce((a, b) => a + b, 0),
                    tokensSaved,
                    compactedToolIds: Array.from(idsToCompact),
                    clearedAttachmentUUIDs: [] // placeholder
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
