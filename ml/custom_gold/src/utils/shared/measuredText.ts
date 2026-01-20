import wrapAnsi from "wrap-ansi";
// @ts-ignore
import wcwidth from "wcwidth";

/**
 * Grapheme-aware line info for wrapped text.
 * Deobfuscated from OeA in chunk_206/207.ts.
 */
export class WrappedLine {
    constructor(
        public text: string,
        public startOffset: number,
        public isPrecededByNewline: boolean,
        public endsWithNewline: boolean = false
    ) { }

    get length(): number {
        return this.text.length + (this.endsWithNewline ? 1 : 0);
    }
}

/**
 * Calculates text layout, wrapping, and grapheme boundaries.
 * Deobfuscated from rKB in chunk_207.ts.
 */
export class MeasuredText {
    private _wrappedLines?: WrappedLine[];
    private graphemeBoundaries?: number[];
    private wordBoundariesCache?: Array<{ start: number; end: number; isWordLike: boolean }>;
    private navigationCache = new Map<string, number>();

    constructor(public text: string, public columns: number) {
        this.text = text.normalize("NFC");
    }

    get wrappedLines(): WrappedLine[] {
        if (!this._wrappedLines) {
            this._wrappedLines = this.measureWrappedText();
        }
        return this._wrappedLines;
    }

    getGraphemeBoundaries(): number[] {
        if (!this.graphemeBoundaries) {
            // @ts-ignore
            const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });
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
            // @ts-ignore
            const segmenter = new Intl.Segmenter(undefined, { granularity: "word" });
            this.wordBoundariesCache = [];
            for (const seg of segmenter.segment(this.text)) {
                this.wordBoundariesCache.push({
                    start: seg.index,
                    end: seg.index + seg.segment.length,
                    isWordLike: (seg as any).isWordLike ?? false
                });
            }
        }
        return this.wordBoundariesCache;
    }

    private measureWrappedText(): WrappedLine[] {
        const wrapped = wrapAnsi(this.text, this.columns, { hard: true, trim: false });
        const lines: WrappedLine[] = [];
        let currentOffset = 0;
        let lastNewlineIdx = -1;

        const parts = wrapped.split("\n");
        for (let i = 0; i < parts.length; i++) {
            const part = parts[i];
            const isPrecededByNewline = (idx: number) => i === 0 || (idx > 0 && this.text[idx - 1] === "\n");

            if (part.length === 0) {
                lastNewlineIdx = this.text.indexOf("\n", lastNewlineIdx + 1);
                if (lastNewlineIdx !== -1) {
                    lines.push(new WrappedLine(part, lastNewlineIdx, isPrecededByNewline(lastNewlineIdx), true));
                } else {
                    lines.push(new WrappedLine(part, this.text.length, isPrecededByNewline(this.text.length), false));
                }
            } else {
                const offset = this.text.indexOf(part.substring(0, 10), currentOffset); // Use partial match to find start
                if (offset === -1) throw new Error("Failed to find wrapped line in text");

                currentOffset = offset + part.length;
                const endsWithNewline = currentOffset < this.text.length && this.text[currentOffset] === "\n";
                if (endsWithNewline) {
                    lastNewlineIdx = currentOffset;
                }
                lines.push(new WrappedLine(part, offset, isPrecededByNewline(offset), endsWithNewline));
            }
        }
        return lines;
    }

    getWrappedText(): string[] {
        return this.wrappedLines.map(l => l.isPrecededByNewline ? l.text : l.text.trimStart());
    }

    getLine(index: number): WrappedLine {
        const lines = this.wrappedLines;
        return lines[Math.max(0, Math.min(index, lines.length - 1))];
    }

    getOffsetFromPosition(pos: { line: number; column: number }): number {
        const line = this.getLine(pos.line);
        if (line.text.length === 0 && line.endsWithNewline) return line.startOffset;

        const leadingSpace = line.isPrecededByNewline ? 0 : line.text.length - line.text.trimStart().length;
        const col = pos.column + leadingSpace;
        const idxInLine = this.displayWidthToStringIndex(line.text, col);
        const offset = line.startOffset + idxInLine;
        const lineEnd = line.startOffset + line.text.length;

        let result = offset;
        const lineWidth = wcwidth(line.text);
        if (line.endsWithNewline && pos.column > lineWidth) {
            result = lineEnd + 1;
        }
        return Math.min(result, lineEnd);
    }

    getPositionFromOffset(offset: number): { line: number; column: number } {
        const lines = this.wrappedLines;
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const nextLine = lines[i + 1];
            if (offset >= line.startOffset && (!nextLine || offset < nextLine.startOffset)) {
                const localOffset = offset - line.startOffset;
                let column: number;
                if (line.isPrecededByNewline) {
                    column = this.stringIndexToDisplayWidth(line.text, localOffset);
                } else {
                    const trimLen = line.text.length - line.text.trimStart().length;
                    if (localOffset < trimLen) {
                        column = 0;
                    } else {
                        column = this.stringIndexToDisplayWidth(line.text.trimStart(), localOffset - trimLen);
                    }
                }
                return { line: i, column: Math.max(0, column) };
            }
        }
        const lastIdx = lines.length - 1;
        return { line: lastIdx, column: wcwidth(lines[lastIdx].text) };
    }

    get lineCount(): number {
        return this.wrappedLines.length;
    }

    private displayWidthToStringIndex(text: string, width: number): number {
        if (width <= 0) return 0;
        let currentWidth = 0;
        let currentIndex = 0;
        // @ts-ignore
        const segmenter = new Intl.Segmenter(undefined, { granularity: "grapheme" });
        for (const { segment, index } of segmenter.segment(text)) {
            const segWidth = wcwidth(segment);
            if (currentWidth + segWidth > width) break;
            currentWidth += segWidth;
            currentIndex = index + segment.length;
        }
        return currentIndex;
    }

    private stringIndexToDisplayWidth(text: string, index: number): number {
        if (index <= 0) return 0;
        if (index >= text.length) return wcwidth(text);
        return wcwidth(text.substring(0, index));
    }

    nextOffset(offset: number): number {
        const key = `next:${offset}`;
        if (this.navigationCache.has(key)) return this.navigationCache.get(key)!;

        const boundaries = this.getGraphemeBoundaries();
        let result = this.text.length;
        for (const b of boundaries) {
            if (b > offset) {
                result = b;
                break;
            }
        }
        this.navigationCache.set(key, result);
        return result;
    }

    prevOffset(offset: number): number {
        if (offset <= 0) return 0;
        const key = `prev:${offset}`;
        if (this.navigationCache.has(key)) return this.navigationCache.get(key)!;

        const boundaries = this.getGraphemeBoundaries();
        let result = 0;
        for (let i = boundaries.length - 1; i >= 0; i--) {
            if (boundaries[i] < offset) {
                result = boundaries[i];
                break;
            }
        }
        this.navigationCache.set(key, result);
        return result;
    }
}
