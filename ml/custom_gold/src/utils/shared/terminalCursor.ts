import { MeasuredText, WrappedLine } from "./measuredText.js";

/**
 * Logic-rich class for terminal text editing and movement.
 * Deobfuscated from _8 in chunk_206.ts.
 */

export class TerminalCursor {
    constructor(
        public measuredText: MeasuredText,
        public offset: number = 0,
        public selection: number = 0
    ) {
        this.offset = Math.max(0, Math.min(this.text.length, offset));
    }

    static fromText(text: string, columns: number, offset: number = 0, selection: number = 0): TerminalCursor {
        return new TerminalCursor(new MeasuredText(text, columns), offset, selection);
    }

    get text(): string {
        return this.measuredText.text;
    }

    get columns(): number {
        return this.measuredText.columns + 1;
    }

    isAtStart() { return this.offset === 0; }
    isAtEnd() { return this.offset >= this.text.length; }

    left(): TerminalCursor {
        if (this.isAtStart()) return this;
        // Handle surrogate pairs/graphemes via measuredText
        const newOffset = this.measuredText.prevOffset ? this.measuredText.prevOffset(this.offset) : this.offset - 1;
        return new TerminalCursor(this.measuredText, newOffset);
    }

    right(): TerminalCursor {
        if (this.isAtEnd()) return this;
        const newOffset = this.measuredText.nextOffset ? this.measuredText.nextOffset(this.offset) : this.offset + 1;
        return new TerminalCursor(this.measuredText, Math.min(newOffset, this.text.length));
    }

    // Multi-line navigation (logical vs wrapped)
    // These require the wrapped text lines from measuredText

    startOfLogicalLine(): TerminalCursor {
        const start = this.text.lastIndexOf("\n", this.offset - 1);
        return new TerminalCursor(this.measuredText, start === -1 ? 0 : start + 1);
    }

    endOfLogicalLine(): TerminalCursor {
        const end = this.text.indexOf("\n", this.offset);
        return new TerminalCursor(this.measuredText, end === -1 ? this.text.length : end);
    }

    upLogicalLine(): TerminalCursor {
        const { start } = this.getLogicalLineBounds();
        if (start === 0) return new TerminalCursor(this.measuredText, 0);

        const column = this.offset - start;
        const prevLineEnd = start - 1;
        const prevLineStart = this.text.lastIndexOf("\n", prevLineEnd - 1) + 1;
        const prevLineLength = prevLineEnd - prevLineStart;

        return new TerminalCursor(this.measuredText, prevLineStart + Math.min(column, prevLineLength));
    }

    downLogicalLine(): TerminalCursor {
        const { end } = this.getLogicalLineBounds();
        if (end >= this.text.length) return new TerminalCursor(this.measuredText, this.text.length);

        const column = this.offset - this.findLogicalLineStart();
        const nextLineStart = end + 1;
        const nextLineEndIdx = this.text.indexOf("\n", nextLineStart);
        const nextLineEnd = nextLineEndIdx === -1 ? this.text.length : nextLineEndIdx;
        const nextLineLength = nextLineEnd - nextLineStart;

        return new TerminalCursor(this.measuredText, nextLineStart + Math.min(column, nextLineLength));
    }

    getLogicalLineBounds() {
        return {
            start: this.findLogicalLineStart(),
            end: this.findLogicalLineEnd()
        };
    }

    findLogicalLineStart(offset: number = this.offset): number {
        const idx = this.text.lastIndexOf("\n", offset - 1);
        return idx === -1 ? 0 : idx + 1;
    }

    findLogicalLineEnd(offset: number = this.offset): number {
        const idx = this.text.indexOf("\n", offset);
        return idx === -1 ? this.text.length : idx;
    }

    // Word movement
    nextWord(): TerminalCursor {
        if (this.isAtEnd()) return this;
        const match = this.text.slice(this.offset).match(/\b\w/);
        if (!match) return new TerminalCursor(this.measuredText, this.text.length);
        return new TerminalCursor(this.measuredText, this.offset + match.index!);
    }

    prevWord(): TerminalCursor {
        if (this.isAtStart()) return this;
        const match = this.text.slice(0, this.offset).match(/\w\b/g);
        if (!match) return new TerminalCursor(this.measuredText, 0);
        // Find the last boundary before offset
        let last = 0;
        const regex = /\w\b/g;
        let m;
        while ((m = regex.exec(this.text.slice(0, this.offset))) !== null) {
            last = m.index;
        }
        // Need to find start of word
        const startMatch = this.text.slice(0, last + 1).match(/\b\w\w*$/);
        return new TerminalCursor(this.measuredText, startMatch ? startMatch.index! : last);
    }

    // Text modification
    modifyText(target: TerminalCursor, replacement: string = ""): TerminalCursor {
        const start = Math.min(this.offset, target.offset);
        const end = Math.max(this.offset, target.offset);
        const newText = this.text.slice(0, start) + replacement + this.text.slice(end);
        return TerminalCursor.fromText(newText, this.columns, start + replacement.length);
    }

    insert(text: string): TerminalCursor {
        return this.modifyText(this, text);
    }

    backspace(): TerminalCursor {
        if (this.isAtStart()) return this;
        return this.left().modifyText(this);
    }

    del(): TerminalCursor {
        if (this.isAtEnd()) return this;
        return this.modifyText(this.right());
    }

    deleteToLineEnd(): { cursor: TerminalCursor; killed: string } {
        const end = this.endOfLogicalLine();
        if (this.offset === end.offset && !this.isAtEnd()) {
            // Delete the newline itself
            const next = this.right();
            return { cursor: this.modifyText(next), killed: "\n" };
        }
        const killed = this.text.slice(this.offset, end.offset);
        return { cursor: this.modifyText(end), killed };
    }

    deleteWordBefore(): { cursor: TerminalCursor; killed: string } {
        if (this.isAtStart()) return { cursor: this, killed: "" };
        const prev = this.prevWord();
        const killed = this.text.slice(prev.offset, this.offset);
        return { cursor: prev.modifyText(this), killed };
    }
}
