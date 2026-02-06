import wrapAnsi from 'wrap-ansi';

/**
 * Utility class for measuring text dimensions in the terminal.
 * Heavily inspired by chunk381 from the 2.1.19 gold reference.
 */
export class MeasuredText {
    private readonly text: string;
    private readonly columns: number;
    private _lineCount: number | null = null;
    private _wrappedText: string | null = null;

    constructor(text: string, columns: number) {
        this.text = text.normalize('NFC');
        this.columns = Math.max(1, columns);
    }

    /**
     * Gets the wrapped text according to terminal columns.
     */
    get wrappedText(): string {
        if (this._wrappedText === null) {
            // Using wrapAnsi with hard wrapping and no trim to accurately simulate terminal behavior
            this._wrappedText = wrapAnsi(this.text, this.columns, {
                hard: true,
                trim: false
            });
        }
        return this._wrappedText;
    }

    /**
     * Gets the total number of lines after wrapping.
     */
    get lineCount(): number {
        if (this._lineCount === null) {
            if (this.text.length === 0) {
                this._lineCount = 0;
            } else {
                // Split by newline and count fragments
                const lines = this.wrappedText.split('\n');
                this._lineCount = lines.length;
            }
        }
        return this._lineCount;
    }

    /**
     * Static helper for quick height calculation.
     */
    static measureHeight(text: string, columns: number): number {
        return new MeasuredText(text, columns).lineCount;
    }
}
