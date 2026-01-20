
import { Box, Text } from "ink";

// Use global Intl.Segmenter. If not available, we might need a polyfill,
// but for Node.js 16+ it should be present.
const segmenter = new (Intl as any).Segmenter(undefined, { granularity: "grapheme" });
const wordSegmenter = new (Intl as any).Segmenter(undefined, { granularity: "word" });

/**
 * Helper to get display width of a string in terminal.
 */
function getStringDisplayWidth(str: string): number {
    // This is a simplified version; real one would use wcwidth
    // For now, let's assume 1 char = 1 width, except maybe for some emojis/CJK
    return str.length;
}

export class MeasuredLine {
    constructor(
        public text: string,
        public startOffset: number,
        public isPrecededByNewline: boolean,
        public endsWithNewline: boolean = false
    ) { }

    equals(other: MeasuredLine) {
        return this.text === other.text && this.startOffset === other.startOffset;
    }

    get length() {
        return this.text.length + (this.endsWithNewline ? 1 : 0);
    }
}

export class MeasuredText {
    private _wrappedLines: MeasuredLine[] | null = null;
    private navigationCache = new Map<string, number>();
    private graphemeBoundaries: number[] | null = null;
    private wordBoundariesCache: Array<{ start: number, end: number, isWordLike: boolean }> | null = null;

    constructor(
        public readonly text: string,
        public readonly columns: number
    ) {
        this.text = text.normalize("NFC");
    }

    get wrappedLines() {
        if (!this._wrappedLines) this._wrappedLines = this.measureWrappedText();
        return this._wrappedLines;
    }

    getGraphemeBoundaries(): number[] {
        if (!this.graphemeBoundaries) {
            this.graphemeBoundaries = [];
            for (const { index } of segmenter.segment(this.text)) {
                this.graphemeBoundaries.push(index);
            }
            this.graphemeBoundaries.push(this.text.length);
        }
        return this.graphemeBoundaries;
    }

    getWordBoundaries() {
        if (!this.wordBoundariesCache) {
            this.wordBoundariesCache = [];
            for (const seg of wordSegmenter.segment(this.text)) {
                this.wordBoundariesCache.push({
                    start: seg.index,
                    end: seg.index + seg.segment.length,
                    isWordLike: seg.isWordLike ?? false
                });
            }
        }
        return this.wordBoundariesCache;
    }

    private binarySearchBoundary(boundaries: number[], currentOffset: number, forward: boolean): number {
        let low = 0;
        let high = boundaries.length - 1;
        let result = forward ? this.text.length : 0;

        while (low <= high) {
            const mid = Math.floor((low + high) / 2);
            const boundary = boundaries[mid];
            if (boundary === undefined) break;

            if (forward) {
                if (boundary > currentOffset) {
                    result = boundary;
                    high = mid - 1;
                } else {
                    low = mid + 1;
                }
            } else {
                if (boundary < currentOffset) {
                    result = boundary;
                    low = mid + 1;
                } else {
                    high = mid - 1;
                }
            }
        }
        return result;
    }

    stringIndexToDisplayWidth(text: string, index: number): number {
        if (index <= 0) return 0;
        if (index >= text.length) return getStringDisplayWidth(text);
        return getStringDisplayWidth(text.substring(0, index));
    }

    displayWidthToStringIndex(lineText: string, width: number): number {
        if (width <= 0 || !lineText) return 0;
        // If the line is the whole text, we use the global offset
        if (lineText === this.text) return this.offsetAtDisplayWidth(width);

        let currentWidth = 0;
        let lastIndex = 0;
        for (const { segment, index } of segmenter.segment(lineText)) {
            const segWidth = getStringDisplayWidth(segment);
            if (currentWidth + segWidth > width) break;
            currentWidth += segWidth;
            lastIndex = index + segment.length;
        }
        return lastIndex;
    }

    offsetAtDisplayWidth(width: number): number {
        if (width <= 0) return 0;
        let currentWidth = 0;
        const boundaries = this.getGraphemeBoundaries();
        for (let i = 0; i < boundaries.length - 1; i++) {
            const start = boundaries[i];
            const end = boundaries[i + 1];
            if (start === undefined || end === undefined) continue;
            const char = this.text.substring(start, end);
            const charWidth = getStringDisplayWidth(char);
            if (currentWidth + charWidth > width) return start;
            currentWidth += charWidth;
        }
        return this.text.length;
    }

    private measureWrappedText(): MeasuredLine[] {
        // Simplified wrap logic for now. 
        // In real Claude, this likely uses 'wrap-ansi'.
        const lines: MeasuredLine[] = [];
        let offset = 0;
        const rawLines = this.text.split("\n");

        for (let i = 0; i < rawLines.length; i++) {
            const lineText = rawLines[i];
            const isFirst = i === 0;
            // Real implementation would handle columns here.
            // For now, let's treat each physical line as a wrapped line if it fits.
            lines.push(new MeasuredLine(lineText, offset, isFirst, i < rawLines.length - 1));
            offset += lineText.length + 1;
        }
        return lines;
    }

    getWrappedText(): string[] {
        return this.wrappedLines.map(line => line.isPrecededByNewline ? line.text : line.text.trimStart());
    }

    getLine(idx: number): MeasuredLine {
        const lines = this.wrappedLines;
        return lines[Math.max(0, Math.min(idx, lines.length - 1))];
    }

    getOffsetFromPosition(pos: { line: number, column: number }): number {
        const line = this.getLine(pos.line);
        if (line.text.length === 0 && line.endsWithNewline) return line.startOffset;

        const indent = line.isPrecededByNewline ? 0 : line.text.length - line.text.trimStart().length;
        const targetCol = pos.column + indent;
        const strIndex = this.displayWidthToStringIndex(line.text, targetCol);

        const offset = line.startOffset + strIndex;
        const lineEnd = line.startOffset + line.text.length;

        let finalOffset = offset;
        const lineWidth = getStringDisplayWidth(line.text);
        if (line.endsWithNewline && pos.column > lineWidth) {
            finalOffset = lineEnd + 1;
        }
        return Math.min(finalOffset, lineEnd);
    }

    getLineLength(idx: number): number {
        const line = this.getLine(idx);
        return getStringDisplayWidth(line.text);
    }

    getPositionFromOffset(offset: number): { line: number, column: number } {
        const lines = this.wrappedLines;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const nextLine = lines[i + 1];
            if (offset >= line.startOffset && (!nextLine || offset < nextLine.startOffset)) {
                const relativeOffset = offset - line.startOffset;
                let col: number;
                if (line.isPrecededByNewline) {
                    col = this.stringIndexToDisplayWidth(line.text, relativeOffset);
                } else {
                    const indent = line.text.length - line.text.trimStart().length;
                    if (relativeOffset < indent) {
                        col = 0;
                    } else {
                        const trimmed = line.text.trimStart();
                        col = this.stringIndexToDisplayWidth(trimmed, relativeOffset - indent);
                    }
                }
                return { line: i, column: Math.max(0, col) };
            }
        }
        const lastIdx = lines.length - 1;
        const lastLine = lines[lastIdx];
        return { line: lastIdx, column: getStringDisplayWidth(lastLine.text) };
    }

    get lineCount() {
        return this.wrappedLines.length;
    }

    private withCache(key: string, fn: () => number): number {
        if (this.navigationCache.has(key)) return this.navigationCache.get(key)!;
        const val = fn();
        this.navigationCache.set(key, val);
        return val;
    }

    nextOffset(offset: number): number {
        return this.withCache(`next:${offset}`, () => {
            const boundaries = this.getGraphemeBoundaries();
            return this.binarySearchBoundary(boundaries, offset, true);
        });
    }

    prevOffset(offset: number): number {
        if (offset <= 0) return 0;
        return this.withCache(`prev:${offset}`, () => {
            const boundaries = this.getGraphemeBoundaries();
            return this.binarySearchBoundary(boundaries, offset, false);
        });
    }
}
