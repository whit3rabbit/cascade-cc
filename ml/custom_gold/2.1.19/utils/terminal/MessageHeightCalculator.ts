import { MeasuredText } from './MeasuredText.js';

export interface RenderableMessage {
    role: string;
    content: any;
    uuid?: string;
}

/**
 * Utility for estimating the height of a rendered message in the terminal.
 * This mirrors the rendering logic in MessageRenderer.tsx.
 */
export class MessageHeightCalculator {
    /**
     * Calculates the estimated height (in lines) of a message.
     */
    static calculateHeight(message: RenderableMessage, columns: number): number {
        let totalHeight = 0;

        // Message header (e.g., "User" or "Claude") + margin
        totalHeight += 1; // Header line
        totalHeight += 2; // Margin from Box (marginY=1 adds a line before and after)

        const contentColumns = Math.max(1, columns - 2); // Accounting for paddingLeft={2}

        if (Array.isArray(message.content)) {
            for (const block of message.content) {
                totalHeight += this.calculateBlockHeight(block, contentColumns);
            }
        } else {
            totalHeight += MeasuredText.measureHeight(String(message.content), contentColumns);
        }

        return totalHeight;
    }

    private static calculateBlockHeight(block: any, columns: number): number {
        let blockHeight = 0;

        if (block.type === 'text') {
            blockHeight += MeasuredText.measureHeight(block.text, columns);
        } else if (block.type === 'tool_use') {
            // Box with borderStyle="round" adds 2 lines for borders + content
            blockHeight += 2; // Borders
            blockHeight += 1; // "Tool Use: [name]" line
            // Input JSON might be rendered if verbose, but we'll assume standard height for virtualization
            blockHeight += 1; // Extra margin/padding
        } else if (block.type === 'tool_result') {
            // Box with borderStyle="round" adds 2 lines for borders + content
            blockHeight += 2; // Borders
            blockHeight += 1; // "Tool Result: [id]" line

            const content = String(block.content);
            // Tool results often contain many lines
            blockHeight += MeasuredText.measureHeight(content, columns);
            blockHeight += 1; // Extra margin/padding
        }

        return blockHeight;
    }
}
