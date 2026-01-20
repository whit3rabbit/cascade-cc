import { InkNode, extractTextStyles } from "./inkDomLayer.js";
import { wrapAnsi, truncateAnsi } from "./ansiTextUtilities.js";
import { applyStyles } from "../utils/shared/theme.js";

/**
 * Interface for a single cell in the screen buffer.
 */
export interface ScreenCell {
    char: string;
    styleId: number;
    width: number;
    hyperlink?: string;
}

/**
 * Manages a grid-based screen buffer for CLI rendering.
 */
export class ScreenBuffer {
    public cells: ScreenCell[];
    public width: number;
    public height: number;
    public dirty = true;

    constructor(width: number, height: number) {
        this.width = width;
        this.height = height;
        this.cells = Array(width * height).fill(null).map(() => ({
            char: " ",
            styleId: 0,
            width: 1
        }));
    }

    write(x: number, y: number, char: string, styleId = 0): void {
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) return;
        const idx = y * this.width + x;
        this.cells[idx] = { char, styleId, width: 1 };
        this.dirty = true;
    }

    read(x: number, y: number): ScreenCell | undefined {
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) return undefined;
        return this.cells[y * this.width + x];
    }

    clear(): void {
        this.cells.forEach(cell => {
            cell.char = " ";
            cell.styleId = 0;
            cell.width = 1;
        });
        this.dirty = true;
    }
}

/**
 * Core rendering entry point. Recursively renders Ink nodes to a screen buffer.
 * Deobfuscated from ec1 in chunk_199.ts.
 */
export function renderNodeToScreen(
    node: InkNode,
    buffer: ScreenBuffer,
    offsetX = 0,
    offsetY = 0
): void {
    // 1. Calculate computed positions (simplified placeholder for Yoga logic)
    const x = offsetX + (node.style.left || 0);
    const y = offsetY + (node.style.top || 0);

    if (node.nodeName === "ink-text") {
        const textStyles = extractTextStyles(node);
        let currentX = x;
        let currentY = y;

        for (const { text, styles } of textStyles) {
            const styledText = applyStyles(text, styles);
            // In reality, this would handle line breaks and wrapping
            for (const char of text) {
                buffer.write(currentX++, currentY, char);
            }
        }
    } else if (node.nodeName === "ink-box") {
        // Render border if present
        if (node.style.borderStyle) {
            renderBorder(node, buffer, x, y);
        }

        // Recurse into children
        for (const child of node.childNodes) {
            renderNodeToScreen(child, buffer, x, y);
        }
    } else if (node.nodeName === "ink-root") {
        for (const child of node.childNodes) {
            renderNodeToScreen(child, buffer, x, y);
        }
    }
}

/**
 * Draws a border around a box component.
 * Deobfuscated from ns8 in chunk_199.ts.
 */
export function renderBorder(node: InkNode, buffer: ScreenBuffer, x: number, y: number): void {
    const width = node.style.width || 10; // Fallback
    const height = node.style.height || 5;   // Fallback

    // Simplified box-drawing chars
    const chars = {
        topLeft: "┌",
        topRight: "┐",
        bottomLeft: "└",
        bottomRight: "┘",
        horizontal: "─",
        vertical: "│"
    };

    // Top
    buffer.write(x, y, chars.topLeft);
    for (let i = 1; i < width - 1; i++) buffer.write(x + i, y, chars.horizontal);
    buffer.write(x + width - 1, y, chars.topRight);

    // Bottom
    buffer.write(x, y + height - 1, chars.bottomLeft);
    for (let i = 1; i < width - 1; i++) buffer.write(x + i, y + height - 1, chars.horizontal);
    buffer.write(x + width - 1, y + height - 1, chars.bottomRight);

    // Sides
    for (let i = 1; i < height - 1; i++) {
        buffer.write(x, y + i, chars.vertical);
        buffer.write(x + width - 1, y + i, chars.vertical);
    }
}

/**
 * Diffs two screen buffers to minimize terminal output.
 * Deobfuscated from Vl1 in chunk_199.ts.
 */
export function diffScreens(oldBuffer: ScreenBuffer, newBuffer: ScreenBuffer): { x: number, y: number, cell: ScreenCell }[] {
    const changes = [];
    const length = Math.min(oldBuffer.cells.length, newBuffer.cells.length);

    for (let i = 0; i < length; i++) {
        if (JSON.stringify(oldBuffer.cells[i]) !== JSON.stringify(newBuffer.cells[i])) {
            changes.push({
                x: i % newBuffer.width,
                y: Math.floor(i / newBuffer.width),
                cell: newBuffer.cells[i]
            });
        }
    }
    return changes;
}
