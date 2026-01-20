
import { MeasuredText } from "./MeasuredText.js";

export class TextCursor {
    public readonly measuredText: MeasuredText;
    public readonly selection: number;
    public readonly offset: number;

    constructor(text: MeasuredText, offset: number = 0, selection: number = 0) {
        this.measuredText = text;
        this.selection = selection;
        this.offset = Math.max(0, Math.min(this.text.length, offset));
    }

    static fromText(text: string, columns: number, offset: number = 0, selection: number = 0) {
        // -1 to account for terminal border or similar in original code
        return new TextCursor(new MeasuredText(text, columns - 1), offset, selection);
    }

    get text() { return this.measuredText.text; }
    get columns() { return this.measuredText.columns + 1; }

    getPosition() {
        return this.measuredText.getPositionFromOffset(this.offset);
    }

    getOffset(pos: { line: number, column: number }) {
        return this.measuredText.getOffsetFromPosition(pos);
    }

    render(options: {
        cursorChar?: string,
        maskChar?: string,
        invert?: (s: string) => string
    } = {}): string {
        const { cursorChar, maskChar, invert } = options;
        const { line: cursorLine, column: cursorCol } = this.getPosition();

        return this.measuredText.getWrappedText().map((textLine: string, lineIdx: number, allLines: string[]) => {
            let renderedLine = textLine;

            // Masking
            if (maskChar) {
                renderedLine = maskChar.repeat(renderedLine.length);
            }

            if (cursorLine !== lineIdx) return renderedLine.trimEnd();

            // Cursor rendering logic
            // In a real terminal, we'd use ANSI or Ink components, 
            // but this helper produces the string representation.

            // This part is complex in original (segmenting carefully)
            // For now, let's do a basic version:
            const before = renderedLine.slice(0, cursorCol);
            const charAtCursor = renderedLine[cursorCol] || (cursorChar || " ");
            const after = renderedLine.slice(cursorCol + 1);

            const styledCursor = invert ? invert(charAtCursor) : charAtCursor;
            return (before + styledCursor + after).trimEnd();
        }).join("\n");
    }

    left() {
        if (this.offset === 0) return this;
        const newOffset = this.measuredText.prevOffset(this.offset);
        return new TextCursor(this.measuredText, newOffset);
    }

    right() {
        if (this.offset >= this.text.length) return this;
        const newOffset = this.measuredText.nextOffset(this.offset);
        return new TextCursor(this.measuredText, Math.min(newOffset, this.text.length));
    }

    up() {
        const { line, column } = this.getPosition();
        if (line === 0) return this;

        const lineAbove = this.measuredText.getLine(line - 1);
        if (!lineAbove) return this;

        const lineAboveLen = this.measuredText.getLineLength(line - 1);
        const targetCol = Math.min(column, lineAboveLen);
        const newOffset = this.getOffset({ line: line - 1, column: targetCol });
        return new TextCursor(this.measuredText, newOffset, 0);
    }

    down() {
        const { line, column } = this.getPosition();
        if (line >= this.measuredText.lineCount - 1) return this;

        const lineBelow = this.measuredText.getLine(line + 1);
        if (!lineBelow) return this;

        const lineBelowLen = this.measuredText.getLineLength(line + 1);
        const targetCol = Math.min(column, lineBelowLen);
        const newOffset = this.getOffset({ line: line + 1, column: targetCol });
        return new TextCursor(this.measuredText, newOffset, 0);
    }

    startOfCurrentLine() {
        const { line } = this.getPosition();
        return new TextCursor(this.measuredText, this.getOffset({ line, column: 0 }), 0);
    }

    startOfLine() {
        const { line, column } = this.getPosition();
        if (column === 0 && line > 0) {
            return new TextCursor(this.measuredText, this.getOffset({ line: line - 1, column: 0 }), 0);
        }
        return this.startOfCurrentLine();
    }

    firstNonBlankInLine() {
        const { line } = this.getPosition();
        const lineText = this.measuredText.getLine(line).text || "";
        const match = lineText.match(/^\s*\S/);
        const col = match?.index ? match.index + match[0].length - 1 : 0;
        const offset = this.getOffset({ line, column: col });
        return new TextCursor(this.measuredText, offset, 0);
    }

    endOfLine() {
        const { line } = this.getPosition();
        const len = this.measuredText.getLineLength(line);
        const offset = this.getOffset({ line, column: len });
        return new TextCursor(this.measuredText, offset, 0);
    }

    findLogicalLineStart(atOffset: number = this.offset) {
        const idx = this.text.lastIndexOf("\n", atOffset - 1);
        return idx === -1 ? 0 : idx + 1;
    }

    findLogicalLineEnd(atOffset: number = this.offset) {
        const idx = this.text.indexOf("\n", atOffset);
        return idx === -1 ? this.text.length : idx;
    }

    getLogicalLineBounds() {
        return {
            start: this.findLogicalLineStart(),
            end: this.findLogicalLineEnd()
        };
    }

    endOfLogicalLine() {
        return new TextCursor(this.measuredText, this.findLogicalLineEnd(), 0);
    }

    startOfLogicalLine() {
        return new TextCursor(this.measuredText, this.findLogicalLineStart(), 0);
    }

    firstNonBlankInLogicalLine() {
        const { start, end } = this.getLogicalLineBounds();
        const lineText = this.text.slice(start, end);
        const match = lineText.match(/\S/);
        const offset = start + (match?.index ?? 0);
        return new TextCursor(this.measuredText, offset, 0);
    }

    upLogicalLine() {
        const { start } = this.getLogicalLineBounds();
        if (start === 0) return new TextCursor(this.measuredText, 0, 0);

        const relativeOffset = this.offset - start;
        const prevLineEnd = start - 1;
        const prevLineStart = this.findLogicalLineStart(prevLineEnd);
        const prevLineLen = prevLineEnd - prevLineStart;

        const targetOffset = prevLineStart + Math.min(relativeOffset, prevLineLen);
        return new TextCursor(this.measuredText, targetOffset, 0);
    }

    downLogicalLine() {
        const { start, end } = this.getLogicalLineBounds();
        if (end >= this.text.length) return new TextCursor(this.measuredText, this.text.length, 0);

        const relativeOffset = this.offset - start;
        const nextLineStart = end + 1;
        const nextLineEnd = this.findLogicalLineEnd(nextLineStart);
        const nextLineLen = nextLineEnd - nextLineStart;

        const targetOffset = nextLineStart + Math.min(relativeOffset, nextLineLen);
        return new TextCursor(this.measuredText, targetOffset, 0);
    }

    nextWord() {
        if (this.isAtEnd()) return this;
        const boundaries = this.measuredText.getWordBoundaries();
        for (const b of boundaries) {
            if (b.isWordLike && b.start > this.offset) return new TextCursor(this.measuredText, b.start);
        }
        return new TextCursor(this.measuredText, this.text.length);
    }

    endOfWord() {
        if (this.isAtEnd()) return this;
        const boundaries = this.measuredText.getWordBoundaries();
        for (const b of boundaries) {
            if (!b.isWordLike) continue;
            if (this.offset >= b.start && this.offset < b.end - 1) return new TextCursor(this.measuredText, b.end - 1);
            if (this.offset === b.end - 1) {
                // Find next word end
                for (const nextB of boundaries) {
                    if (nextB.isWordLike && nextB.start > this.offset) return new TextCursor(this.measuredText, nextB.end - 1);
                }
                return this;
            }
        }
        // Fallback for case where we are after last word start
        for (const b of boundaries) {
            if (b.isWordLike && b.start > this.offset) return new TextCursor(this.measuredText, b.end - 1);
        }
        return this;
    }

    prevWord() {
        if (this.isAtStart()) return this;
        const boundaries = this.measuredText.getWordBoundaries();
        let lastWordStart = null;
        for (const b of boundaries) {
            if (!b.isWordLike) continue;
            if (b.start < this.offset) {
                if (this.offset > b.start && this.offset <= b.end) return new TextCursor(this.measuredText, b.start);
                lastWordStart = b.start;
            }
        }
        if (lastWordStart !== null) return new TextCursor(this.measuredText, lastWordStart);
        return new TextCursor(this.measuredText, 0);
    }

    nextWORD() {
        let cursor: TextCursor = this;
        while (!cursor.isOverWhitespace() && !cursor.isAtEnd()) cursor = cursor.right();
        while (cursor.isOverWhitespace() && !cursor.isAtEnd()) cursor = cursor.right();
        return cursor;
    }

    endOfWORD(): TextCursor {
        let cursor: TextCursor = this;
        if (!cursor.isOverWhitespace() && (cursor.right().isOverWhitespace() || cursor.right().isAtEnd())) {
            cursor = cursor.right();
            return cursor.endOfWORD();
        }
        if (cursor.isOverWhitespace()) cursor = cursor.nextWORD();
        while (!cursor.right().isOverWhitespace() && !cursor.isAtEnd()) cursor = cursor.right();
        return cursor;
    }

    prevWORD() {
        let cursor: TextCursor = this;
        if (cursor.left().isOverWhitespace()) cursor = cursor.left();
        while (cursor.isOverWhitespace() && !cursor.isAtStart()) cursor = cursor.left();
        if (!cursor.isOverWhitespace()) {
            while (!cursor.left().isOverWhitespace() && !cursor.isAtStart()) cursor = cursor.left();
        }
        return cursor;
    }

    modifyText(target: TextCursor, replacement: string = ""): TextCursor {
        const start = this.offset;
        const end = target.offset;
        const newText = this.text.slice(0, start) + replacement + this.text.slice(end);
        return TextCursor.fromText(newText, this.columns, start + replacement.normalize("NFC").length);
    }

    insert(text: string) {
        return this.modifyText(this, text);
    }

    del() {
        if (this.isAtEnd()) return this;
        return this.modifyText(this.right());
    }

    backspace() {
        if (this.isAtStart()) return this;
        return this.left().modifyText(this);
    }

    deleteToLineStart(): { cursor: TextCursor, killed: string } {
        const start = this.startOfCurrentLine();
        const killed = this.text.slice(start.offset, this.offset);
        return {
            cursor: start.modifyText(this),
            killed
        };
    }

    deleteToLineEnd(): { cursor: TextCursor, killed: string } {
        if (this.text[this.offset] === "\n") {
            return {
                cursor: this.modifyText(this.right()),
                killed: "\n"
            };
        }
        const end = this.endOfLine();
        const killed = this.text.slice(this.offset, end.offset);
        return {
            cursor: this.modifyText(end),
            killed
        };
    }

    deleteToLogicalLineEnd() {
        if (this.text[this.offset] === "\n") return this.modifyText(this.right());
        return this.modifyText(this.endOfLogicalLine());
    }

    deleteWordBefore(): { cursor: TextCursor, killed: string } {
        if (this.isAtStart()) return { cursor: this, killed: "" };
        const start = this.prevWord();
        const killed = this.text.slice(start.offset, this.offset);
        return {
            cursor: start.modifyText(this),
            killed
        };
    }

    deleteWordAfter() {
        if (this.isAtEnd()) return this;
        return this.modifyText(this.nextWord());
    }

    isOverWhitespace() {
        const char = this.text[this.offset] ?? "";
        return /\s/.test(char);
    }

    equals(other: TextCursor) {
        return this.offset === other.offset &&
            this.text === other.text &&
            this.measuredText.columns === other.measuredText.columns;
    }

    isAtStart() { return this.offset === 0; }
    isAtEnd() { return this.offset >= this.text.length; }

    startOfFirstLine() {
        return new TextCursor(this.measuredText, 0, 0);
    }

    startOfLastLine() {
        const idx = this.text.lastIndexOf("\n");
        if (idx === -1) return this.startOfLine();
        return new TextCursor(this.measuredText, idx + 1, 0);
    }
}
