import { useState, useCallback, useRef, useEffect } from 'react';
import { setCursorStyle, resetCursorStyle } from '../utils/terminal/cursorStyle.js';

export type VimMode = 'INSERT' | 'NORMAL' | 'VISUAL';

export interface UseVimModeProps {
    enabled?: boolean;
    onModeChange?: (mode: VimMode) => void;
    value: string;
    onChange: (value: string) => void;
    cursorOffset: number;
    setCursorOffset: (offset: number) => void;
    killRing: any;
    macros: any;
    onSubmit?: (value: string) => void;
    onExit?: () => void;
    onUndo?: () => void;
    columns?: number; // Added for movement logic
}

export type VimCommandState =
    | { type: 'idle' }
    | { type: 'count'; digits: string }
    | { type: 'operator'; operator: string; count: number }
    | { type: 'operatorCount'; operator: string; count: number; digits: string }
    | { type: 'operatorFind'; operator: string; count: number; findChar: string }
    | { type: 'operatorTextObj'; operator: string; count: number; scope: 'inner' | 'around' }
    | { type: 'find'; findChar: string; count: number }
    | { type: 'g'; count: number }
    | { type: 'operatorG'; operator: string; count: number }
    | { type: 'replace'; count: number }
    | { type: 'indent'; operator: string; count: number }
    | { type: 'macroRecord' }
    | { type: 'macroPlay' }
    | { type: 'colon'; buffer: string };

type VimOperator = 'delete' | 'change' | 'yank';

const MAX_COUNT = 10000;
const OPERATOR_MAP: Record<string, VimOperator> = {
    d: 'delete',
    c: 'change',
    y: 'yank'
};
const MOTIONS = new Set(['h', 'l', 'j', 'k', 'w', 'b', 'e', 'W', 'B', 'E', '0', '^', '$']);
const FIND_MOTIONS = new Set(['f', 'F', 't', 'T']);
const TEXT_OBJECT_SCOPES: Record<string, 'inner' | 'around'> = { i: 'inner', a: 'around' };
const TEXT_OBJECTS = new Set(['w', 'W', '"', '\'', '`', '(', ')', 'b', '[', ']', '{', '}', 'B', '<', '>']);

const lineOffsetFromLines = (lines: string[], line: number) => {
    let offset = 0;
    for (let i = 0; i < line; i++) {
        offset += (lines[i]?.length ?? 0) + 1;
    }
    return offset;
};

const WORD_CHAR = /^[\p{L}\p{N}\p{M}_]$/u;
const isWordChar = (c: string) => WORD_CHAR.test(c);
const isWhitespace = (c: string) => /\s/.test(c);
const isPunctuation = (c: string) => c.length > 0 && !isWordChar(c) && !isWhitespace(c);

export function useVimMode({ enabled, onModeChange, value, onChange, cursorOffset, setCursorOffset, killRing: _killRing, macros, onSubmit, onExit, onUndo, columns: _columns = 80 }: UseVimModeProps) {
    const [mode, setMode] = useState<VimMode>('INSERT');
    const [visualStart, setVisualStart] = useState<number | null>(null);
    const commandStateRef = useRef<VimCommandState>({ type: 'idle' });
    const insertStateRef = useRef({ insertedText: '' });
    const internalVimStateRef = useRef({
        lastChange: null as any | null,
        lastFind: null as { type: string; char: string } | null,
        register: '',
        registerIsLinewise: false
    });

    // Update hardware cursor style
    useEffect(() => {
        if (!enabled) {
            resetCursorStyle();
            return;
        }

        switch (mode) {
            case 'INSERT':
                setCursorStyle('beam');
                break;
            case 'NORMAL':
            case 'VISUAL':
                setCursorStyle('block');
                break;
        }

        return () => {
            resetCursorStyle();
        };
    }, [enabled, mode]);

    const getLogicalLineBounds = useCallback((offset: number) => {
        const start = value.lastIndexOf('\n', offset - 1) + 1;
        const end = value.indexOf('\n', offset);
        return { start, end: end === -1 ? value.length : end };
    }, [value]);

    const getPosition = useCallback((offset: number) => {
        const lines = value.slice(0, offset).split('\n');
        const line = lines.length - 1;
        const column = lines[line].length;
        return { line, column };
    }, [value]);

    const getOffset = useCallback((line: number, column: number) => {
        const lines = value.split('\n');
        const targetLine = Math.max(0, Math.min(line, lines.length - 1));
        let offset = 0;
        for (let i = 0; i < targetLine; i++) {
            offset += lines[i].length + 1;
        }
        return offset + Math.min(column, lines[targetLine]?.length || 0);
    }, [value]);

    const findCharacter = useCallback((char: string, type: string, count: number, startOffset: number) => {
        let offset = startOffset;
        const isForward = type === 'f' || type === 't';
        const isTill = type === 't' || type === 'T';
        let foundCount = 0;

        if (isForward) {
            for (let i = offset + 1; i < value.length; i++) {
                if (value[i] === char) {
                    foundCount++;
                    if (foundCount === count) {
                        return isTill ? Math.max(offset, i - 1) : i;
                    }
                }
            }
        } else {
            for (let i = offset - 1; i >= 0; i--) {
                if (value[i] === char) {
                    foundCount++;
                    if (foundCount === count) {
                        return isTill ? Math.min(offset, i + 1) : i;
                    }
                }
            }
        }
        return null;
    }, [value]);

    const moveNextVimWord = useCallback((offset: number, count: number) => {
        let k = offset;
        for (let i = 0; i < count; i++) {
            if (k >= value.length) break;
            const char = value[k];
            if (isWordChar(char)) {
                while (k < value.length && isWordChar(value[k])) k++;
            } else if (isPunctuation(char)) {
                while (k < value.length && isPunctuation(value[k])) k++;
            }
            while (k < value.length && isWhitespace(value[k])) k++;
        }
        const maxOffset = Math.max(0, value.length - 1);
        return Math.min(k, maxOffset);
    }, [value]);

    const movePrevVimWord = useCallback((offset: number, count: number) => {
        let k = offset;
        for (let i = 0; i < count; i++) {
            if (k <= 0) break;
            k--;
            while (k > 0 && isWhitespace(value[k])) k--;
            if (k === 0 && isWhitespace(value[0])) break;
            const char = value[k];
            if (isWordChar(char)) {
                while (k > 0 && isWordChar(value[k - 1])) k--;
            } else if (isPunctuation(char)) {
                while (k > 0 && isPunctuation(value[k - 1])) k--;
            }
        }
        return k;
    }, [value]);

    const moveEndOfVimWord = useCallback((offset: number, count: number) => {
        let k = offset;
        for (let i = 0; i < count; i++) {
            if (k >= value.length - 1) break;
            k++;
            while (k < value.length && isWhitespace(value[k])) k++;
            if (k >= value.length) break;
            const char = value[k];
            if (isWordChar(char)) {
                while (k < value.length - 1 && isWordChar(value[k + 1])) k++;
            } else if (isPunctuation(char)) {
                while (k < value.length - 1 && isPunctuation(value[k + 1])) k++;
            }
        }
        const maxOffset = Math.max(0, value.length - 1);
        return Math.min(k, maxOffset);
    }, [value]);

    const moveNextWORD = useCallback((offset: number, count: number) => {
        let k = offset;
        for (let i = 0; i < count; i++) {
            while (k < value.length && !isWhitespace(value[k])) k++;
            while (k < value.length && isWhitespace(value[k])) k++;
        }
        const maxOffset = Math.max(0, value.length - 1);
        return Math.min(k, maxOffset);
    }, [value]);

    const movePrevWORD = useCallback((offset: number, count: number) => {
        let k = offset;
        for (let i = 0; i < count; i++) {
            if (k <= 0) break;
            k--;
            while (k > 0 && isWhitespace(value[k])) k--;
            while (k > 0 && !isWhitespace(value[k - 1])) k--;
        }
        return k;
    }, [value]);

    const moveEndOfWORD = useCallback((offset: number, count: number) => {
        let k = offset;
        for (let i = 0; i < count; i++) {
            if (k >= value.length - 1) break;
            k++;
            while (k < value.length && isWhitespace(value[k])) k++;
            while (k < value.length - 1 && !isWhitespace(value[k + 1])) k++;
        }
        const maxOffset = Math.max(0, value.length - 1);
        return Math.min(k, maxOffset);
    }, [value]);

    const getTextObjectRange = useCallback((offset: number, obj: string, scope: 'inner' | 'around') => {
        const text = value;
        const isInner = scope === 'inner';
        const charAt = (idx: number) => text[idx] ?? '';

        if (obj === 'w' || obj === 'W') {
            const isW = obj === 'W';
            const isWordLike = isW ? (c: string) => !isWhitespace(c) : isWordChar;
            const isSpace = isWhitespace;
            const isPunct = (c: string) => c.length > 0 && !isWordChar(c) && !isSpace(c);
            let start = offset;
            let end = offset;

            if (isWordLike(charAt(offset))) {
                while (start > 0 && isWordLike(charAt(start - 1))) start--;
                while (end < text.length && isWordLike(charAt(end))) end++;
            } else if (isSpace(charAt(offset))) {
                while (start > 0 && isSpace(charAt(start - 1))) start--;
                while (end < text.length && isSpace(charAt(end))) end++;
                return { start, end };
            } else if (isPunct(charAt(offset))) {
                while (start > 0 && isPunct(charAt(start - 1))) start--;
                while (end < text.length && isPunct(charAt(end))) end++;
            }

            if (!isInner) {
                if (end < text.length && isSpace(charAt(end))) {
                    while (end < text.length && isSpace(charAt(end))) end++;
                } else if (start > 0 && isSpace(charAt(start - 1))) {
                    while (start > 0 && isSpace(charAt(start - 1))) start--;
                }
            }
            return { start, end };
        }

        const pairs: { [key: string]: [string, string] } = {
            '(': ['(', ')'], ')': ['(', ')'], 'b': ['(', ')'],
            '[': ['[', ']'], ']': ['[', ']'],
            '{': ['{', '}'], '}': ['{', '}'], 'B': ['{', '}'],
            '<': ['<', '>'], '>': ['<', '>'],
            '"': ['"', '"'], "'": ["'", "'"], '`': ['`', '`']
        };
        const pair = pairs[obj];
        if (!pair) return null;

        const [open, close] = pair;
        if (open === close) {
            const lineStart = text.lastIndexOf('\n', offset - 1) + 1;
            const lineEnd = text.indexOf('\n', offset);
            const actualLineEnd = lineEnd === -1 ? text.length : lineEnd;
            const line = text.slice(lineStart, actualLineEnd);
            const relOffset = offset - lineStart;
            const indices: number[] = [];
            for (let i = 0; i < line.length; i++) {
                if (line[i] === open) indices.push(i);
            }
            for (let i = 0; i < indices.length - 1; i += 2) {
                if (indices[i] <= relOffset && relOffset <= indices[i + 1]) {
                    return isInner
                        ? { start: lineStart + indices[i] + 1, end: lineStart + indices[i + 1] }
                        : { start: lineStart + indices[i], end: lineStart + indices[i + 1] + 1 };
                }
            }
            return null;
        }

        let depth = 0;
        let start = -1;
        for (let i = offset; i >= 0; i--) {
            if (text[i] === close && i !== offset) depth++;
            else if (text[i] === open) {
                if (depth === 0) { start = i; break; }
                depth--;
            }
        }
        if (start === -1) return null;
        depth = 0;
        let end = -1;
        for (let i = start + 1; i < text.length; i++) {
            if (text[i] === open) depth++;
            else if (text[i] === close) {
                if (depth === 0) { end = i; break; }
                depth--;
            }
        }
        if (end === -1) return null;
        return isInner
            ? { start: start + 1, end }
            : { start, end: end + 1 };
    }, [value]);

    const setVimMode = useCallback((nextMode: VimMode) => {
        if (nextMode === 'INSERT') {
            insertStateRef.current.insertedText = '';
        }
        commandStateRef.current = { type: 'idle' };
        setMode(nextMode);
        onModeChange?.(nextMode);
    }, [onModeChange]);

    const enterInsert = useCallback((offset: number) => {
        setCursorOffset(offset);
        setVimMode('INSERT');
    }, [setCursorOffset, setVimMode]);

    const enterNormal = useCallback((offset?: number) => {
        if (offset !== undefined) {
            setCursorOffset(offset);
        }
        setVimMode('NORMAL');
    }, [setCursorOffset, setVimMode]);

    const performOperator = useCallback((op: VimOperator, start: number, end: number, linewise: boolean, recordInfo?: any, record: boolean = true) => {
        const from = Math.min(start, end);
        const to = Math.max(start, end);
        let killed = value.slice(from, to);
        if (linewise && !killed.endsWith('\n')) killed += '\n';

        internalVimStateRef.current.register = killed;
        internalVimStateRef.current.registerIsLinewise = linewise;

        if (op === 'yank') {
            setCursorOffset(from);
        } else if (op === 'delete') {
            const newValue = value.slice(0, from) + value.slice(to);
            onChange(newValue);
            setCursorOffset(Math.min(from, Math.max(0, newValue.length - 1)));
        } else if (op === 'change') {
            const newValue = value.slice(0, from) + value.slice(to);
            onChange(newValue);
            enterInsert(from);
        }
        if (record && recordInfo) {
            internalVimStateRef.current.lastChange = recordInfo;
        }
    }, [value, onChange, setCursorOffset, enterInsert]);

    const handleNormalModeKey = useCallback((key: any, input: string) => {
        const state = commandStateRef.current;
        const internal = internalVimStateRef.current;

        const recordChange = (change: any) => {
            internal.lastChange = change;
        };

        const resetState = () => {
            commandStateRef.current = { type: 'idle' };
        };

        const clampCount = (count: number) => Math.min(count, MAX_COUNT);
        const maxOffset = Math.max(0, value.length - 1);

        const executeMotion = (motion: string, count: number, startOffset: number = cursorOffset) => {
            let offset = startOffset;
            switch (motion) {
                case 'h':
                    offset = Math.max(0, startOffset - count);
                    break;
                case 'l':
                    offset = Math.min(maxOffset, startOffset + count);
                    break;
                case 'j': {
                    const { line, column } = getPosition(startOffset);
                    offset = getOffset(line + count, column);
                    break;
                }
                case 'k': {
                    const { line, column } = getPosition(startOffset);
                    offset = getOffset(line - count, column);
                    break;
                }
                case 'w':
                    offset = moveNextVimWord(startOffset, count);
                    break;
                case 'b':
                    offset = movePrevVimWord(startOffset, count);
                    break;
                case 'e':
                    offset = moveEndOfVimWord(startOffset, count);
                    break;
                case 'W':
                    offset = moveNextWORD(startOffset, count);
                    break;
                case 'B':
                    offset = movePrevWORD(startOffset, count);
                    break;
                case 'E':
                    offset = moveEndOfWORD(startOffset, count);
                    break;
                case '0':
                    offset = getLogicalLineBounds(startOffset).start;
                    break;
                case '^': {
                    const { start, end } = getLogicalLineBounds(startOffset);
                    const lineText = value.slice(start, end);
                    const nonBlank = lineText.search(/\S/);
                    offset = start + (nonBlank === -1 ? 0 : nonBlank);
                    break;
                }
                case '$': {
                    const { end } = getLogicalLineBounds(startOffset);
                    offset = Math.max(0, end - 1);
                    break;
                }
                case 'G': {
                    const lines = value.split('\n');
                    const target = count === 1 ? lines.length - 1 : Math.min(count - 1, lines.length - 1);
                    offset = lineOffsetFromLines(lines, target);
                    break;
                }
                case 'gg': {
                    const lines = value.split('\n');
                    const target = count === 1 ? 0 : Math.min(count - 1, lines.length - 1);
                    offset = lineOffsetFromLines(lines, target);
                    break;
                }
            }
            return offset;
        };

        const computeOperatorRange = (op: VimOperator, motion: string, count: number, startOffset: number, motionOffset: number) => {
            let from = Math.min(startOffset, motionOffset);
            let to = Math.max(startOffset, motionOffset);
            let linewise = false;

            if (op === 'change' && (motion === 'w' || motion === 'W')) {
                const nextWord = motion === 'w' ? moveNextVimWord : moveNextWORD;
                const endWord = motion === 'w' ? moveEndOfVimWord : moveEndOfWORD;
                let offset = startOffset;
                if (count > 1) {
                    offset = nextWord(offset, count - 1);
                }
                const endOffset = endWord(offset, 1);
                from = startOffset;
                to = Math.min(value.length, endOffset + 1);
            } else if (motion === 'j' || motion === 'k' || motion === 'G' || motion === 'gg') {
                linewise = true;
                let rangeEnd = to;
                const nextBreak = value.indexOf('\n', rangeEnd);
                if (nextBreak === -1) {
                    rangeEnd = value.length;
                    if (from > 0 && value[from - 1] === '\n') {
                        from -= 1;
                    }
                } else {
                    rangeEnd = nextBreak + 1;
                }
                to = rangeEnd;
            } else if ((motion === 'e' || motion === 'E' || motion === '$') && startOffset <= motionOffset) {
                to = Math.min(value.length, to + 1);
            }
            return { from, to, linewise };
        };

        const applyDeleteChar = (count: number, record: boolean = true) => {
            const start = cursorOffset;
            const end = Math.min(start + count, value.length);
            if (start >= value.length) return;
            const deleted = value.slice(start, end);
            internal.register = deleted;
            internal.registerIsLinewise = false;
            const newValue = value.slice(0, start) + value.slice(end);
            onChange(newValue);
            setCursorOffset(Math.min(start, Math.max(0, newValue.length - 1)));
            if (record) {
                recordChange({ type: 'x', count });
            }
        };

        const applyToggleCase = (count: number, record: boolean = true) => {
            let start = cursorOffset;
            let end = Math.min(start + count, value.length);
            if (start >= value.length) return;
            let text = value;
            for (let i = start; i < end; i++) {
                const c = text[i];
                const swapped = c === c.toUpperCase() ? c.toLowerCase() : c.toUpperCase();
                text = text.slice(0, i) + swapped + text.slice(i + 1);
            }
            onChange(text);
            setCursorOffset(end);
            if (record) {
                recordChange({ type: 'toggleCase', count });
            }
        };

        const applyReplaceChar = (char: string, count: number, record: boolean = true) => {
            const start = cursorOffset;
            const end = Math.min(start + count, value.length);
            if (start >= value.length) return;
            let text = value;
            for (let i = start; i < end; i++) {
                text = text.slice(0, i) + char + text.slice(i + 1);
            }
            onChange(text);
            setCursorOffset(Math.max(0, end - 1));
            if (record) {
                recordChange({ type: 'replace', char, count });
            }
        };

        const applyIndent = (dir: '>' | '<', count: number, record: boolean = true) => {
            const lines = value.split('\n');
            const { line } = getPosition(cursorOffset);
            const lineCount = Math.min(count, lines.length - line);
            for (let i = 0; i < lineCount; i++) {
                const idx = line + i;
                const current = lines[idx] ?? '';
                if (dir === '>') {
                    lines[idx] = '  ' + current;
                } else if (current.startsWith('  ')) {
                    lines[idx] = current.slice(2);
                } else if (current.startsWith('\t')) {
                    lines[idx] = current.slice(1);
                } else {
                    let removed = 0;
                    let pos = 0;
                    while (pos < current.length && removed < 2 && /\s/.test(current[pos])) {
                        removed++;
                        pos++;
                    }
                    lines[idx] = current.slice(pos);
                }
            }
            const nextValue = lines.join('\n');
            const leadingWhitespace = ((lines[line] ?? '').match(/^\s*/)?.[0] ?? '').length;
            onChange(nextValue);
            setCursorOffset(lineOffsetFromLines(lines, line) + leadingWhitespace);
            if (record) {
                recordChange({ type: 'indent', dir, count });
            }
        };

        const applyJoin = (count: number, record: boolean = true) => {
            const lines = value.split('\n');
            const { line } = getPosition(cursorOffset);
            if (line >= lines.length - 1) return;
            const toJoin = Math.min(count, lines.length - line - 1);
            let joined = lines[line];
            const originalLength = joined.length;
            for (let i = 1; i <= toJoin; i++) {
                const nextLine = (lines[line + i] ?? '').trimStart();
                if (nextLine.length > 0) {
                    if (!joined.endsWith(' ') && joined.length > 0) {
                        joined += ' ';
                    }
                    joined += nextLine;
                }
            }
            const nextLines = [...lines.slice(0, line), joined, ...lines.slice(line + toJoin + 1)];
            onChange(nextLines.join('\n'));
            setCursorOffset(lineOffsetFromLines(nextLines, line) + originalLength);
            if (record) {
                recordChange({ type: 'join', count });
            }
        };

        const applyOpenLine = (direction: 'above' | 'below', record: boolean = true) => {
            const lines = value.split('\n');
            const { line } = getPosition(cursorOffset);
            const target = direction === 'below' ? line + 1 : line;
            const nextLines = [...lines.slice(0, target), '', ...lines.slice(target)];
            onChange(nextLines.join('\n'));
            enterInsert(lineOffsetFromLines(nextLines, target));
            if (record) {
                recordChange({ type: 'openLine', direction });
            }
        };

        const applyPaste = (after: boolean, count: number) => {
            const reg = internal.register;
            if (!reg) return;
            const isLinewise = reg.endsWith('\n');
            if (isLinewise) {
                const lines = value.split('\n');
                const { line } = getPosition(cursorOffset);
                const target = after ? line + 1 : line;
                const regLines = reg.slice(0, -1).split('\n');
                const insertLines: string[] = [];
                for (let i = 0; i < count; i++) {
                    insertLines.push(...regLines);
                }
                const nextLines = [...lines.slice(0, target), ...insertLines, ...lines.slice(target)];
                onChange(nextLines.join('\n'));
                setCursorOffset(lineOffsetFromLines(nextLines, target));
            } else {
                const insertion = reg.repeat(count);
                const pos = after && cursorOffset < value.length ? cursorOffset + 1 : cursorOffset;
                const nextValue = value.slice(0, pos) + insertion + value.slice(pos);
                const nextOffset = Math.max(pos, pos + insertion.length - 1);
                onChange(nextValue);
                setCursorOffset(nextOffset);
            }
        };

        const applyLinewiseOperator = (op: VimOperator, count: number, record: boolean = true) => {
            const lines = value.split('\n');
            const { line } = getPosition(cursorOffset);
            const lineCount = Math.min(count, lines.length - line);
            const lineStart = lineOffsetFromLines(lines, line);
            let endOffset = lineStart;
            for (let i = 0; i < lineCount; i++) {
                const nextBreak = value.indexOf('\n', endOffset);
                endOffset = nextBreak === -1 ? value.length : nextBreak + 1;
            }
            let grabbed = value.slice(lineStart, endOffset);
            if (!grabbed.endsWith('\n')) grabbed += '\n';
            internal.register = grabbed;
            internal.registerIsLinewise = true;

            if (op === 'yank') {
                setCursorOffset(lineStart);
            } else if (op === 'delete') {
                let from = lineStart;
                let to = endOffset;
                if (to === value.length && from > 0 && value[from - 1] === '\n') {
                    from -= 1;
                }
                const nextValue = value.slice(0, from) + value.slice(to);
                onChange(nextValue || '');
                setCursorOffset(Math.min(from, Math.max(0, nextValue.length - 1)));
            } else if (op === 'change') {
                if (lines.length === 1) {
                    onChange('');
                    enterInsert(0);
                } else {
                    const before = lines.slice(0, line);
                    const after = lines.slice(line + lineCount);
                    const nextLines = [...before, '', ...after];
                    onChange(nextLines.join('\n'));
                    enterInsert(lineStart);
                }
            }
            if (record) {
                recordChange({ type: 'operator', clearConversation: op, motion: op[0], count });
            }
        };

        const applyOperatorMotion = (op: VimOperator, motion: string, count: number, record: boolean = true) => {
            const startOffset = cursorOffset;
            const motionOffset = executeMotion(motion, count, startOffset);
            const allowSameSpot = (op === 'change' && (motion === 'w' || motion === 'W')) || motion === 'e' || motion === 'E' || motion === '$';
            if (motionOffset === startOffset && !allowSameSpot) return;
            const range = computeOperatorRange(op, motion, count, startOffset, motionOffset);
            performOperator(op, range.from, range.to, range.linewise, {
                type: 'operator',
                clearConversation: op,
                motion,
                count
            }, record);
        };

        const applyOperatorToLine = (op: VimOperator, motion: 'G' | 'gg', count: number, record: boolean = true) => {
            const startOffset = cursorOffset;
            const motionOffset = executeMotion(motion, count, startOffset);
            const range = computeOperatorRange(op, motion, count, startOffset, motionOffset);
            performOperator(op, range.from, range.to, range.linewise, {
                type: 'operator',
                clearConversation: op,
                motion,
                count
            }, record);
        };

        const applyOperatorFind = (op: VimOperator, findType: string, char: string, count: number, record: boolean = true) => {
            const motionOffset = findCharacter(char, findType, count, cursorOffset);
            if (motionOffset === null) return;
            const from = Math.min(cursorOffset, motionOffset);
            const to = Math.max(cursorOffset, motionOffset) + 1;
            performOperator(op, from, to, false, {
                type: 'operatorFind',
                clearConversation: op,
                find: findType,
                char,
                count
            }, record);
            internal.lastFind = { type: findType, char };
        };

        const applyOperatorTextObj = (op: VimOperator, scope: 'inner' | 'around', objType: string, count: number, record: boolean = true) => {
            const range = getTextObjectRange(cursorOffset, objType, scope);
            if (!range) return;
            performOperator(op, range.start, range.end, false, {
                type: 'operatorTextObj',
                clearConversation: op,
                scope,
                objType,
                count
            }, record);
        };

        const repeatLastFind = (reverse: boolean, count: number) => {
            const last = internal.lastFind;
            if (!last) return;
            let type = last.type;
            if (reverse) {
                type = { f: 'F', F: 'f', t: 'T', T: 't' }[type] as string;
            }
            const nextOffset = findCharacter(last.char, type, count, cursorOffset);
            if (nextOffset !== null) {
                setCursorOffset(nextOffset);
            }
        };

        const repeatLastChange = () => {
            const change = internal.lastChange;
            if (!change) return;
            switch (change.type) {
                case 'insert': {
                    if (change.text) {
                        const nextValue = value.slice(0, cursorOffset) + change.text + value.slice(cursorOffset);
                        onChange(nextValue);
                        setCursorOffset(cursorOffset + change.text.length);
                    }
                    break;
                }
                case 'x':
                    applyDeleteChar(change.count, false);
                    break;
                case 'replace':
                    applyReplaceChar(change.char, change.count, false);
                    break;
                case 'toggleCase':
                    applyToggleCase(change.count, false);
                    break;
                case 'indent':
                    applyIndent(change.dir, change.count, false);
                    break;
                case 'join':
                    applyJoin(change.count, false);
                    break;
                case 'openLine':
                    applyOpenLine(change.direction, false);
                    break;
                case 'operator':
                    applyOperatorMotion(change.clearConversation, change.motion, change.count, false);
                    break;
                case 'operatorFind':
                    applyOperatorFind(change.clearConversation, change.find, change.char, change.count, false);
                    break;
                case 'operatorTextObj':
                    applyOperatorTextObj(change.clearConversation, change.scope, change.objType, change.count, false);
                    break;
            }
        };

        // Handle Escape or Ctrl-C/G to reset state
        if (key.escape || (key.ctrl && (input === 'c' || input === 'g'))) {
            resetState();
            return;
        }

        // Handle Colon Mode
        if (state.type === 'colon') {
            if (key.return) {
                if (state.buffer === 'w' || state.buffer === 'wq') onSubmit?.(value);
                if (state.buffer === 'q' || state.buffer === 'wq') onExit?.();
                resetState();
            } else if (input.length === 1) {
                commandStateRef.current = { type: 'colon', buffer: state.buffer + input };
            }
            return;
        }

        // Handle Macro states
        if (state.type === 'macroRecord') {
            if (input.length === 1) {
                macros.startRecording(input);
                resetState();
            }
            return;
        }
        if (state.type === 'macroPlay') {
            if (input.length === 1) {
                macros.playMacro(input);
                resetState();
            }
            return;
        }

        let mappedInput = input;
        if (key.leftArrow) mappedInput = 'h';
        if (key.rightArrow) mappedInput = 'l';
        if (key.upArrow) mappedInput = 'k';
        if (key.downArrow) mappedInput = 'j';

        const handleIdleCommand = (ch: string, count: number): { execute?: () => void; next?: VimCommandState } | null => {
            const op = OPERATOR_MAP[ch];
            if (op) {
                return { next: { type: 'operator', operator: op, count } };
            }
            if (MOTIONS.has(ch)) {
                return { execute: () => setCursorOffset(executeMotion(ch, count)) };
            }
            if (FIND_MOTIONS.has(ch)) {
                return { next: { type: 'find', findChar: ch, count } };
            }
            if (ch === 'g') {
                return { next: { type: 'g', count } };
            }
            if (ch === 'r') {
                return { next: { type: 'replace', count } };
            }
            if (ch === '>' || ch === '<') {
                return { next: { type: 'indent', operator: ch, count } };
            }
            if (ch === '~') {
                return { execute: () => applyToggleCase(count) };
            }
            if (ch === 'x') {
                return { execute: () => applyDeleteChar(count) };
            }
            if (ch === 'J') {
                return { execute: () => applyJoin(count) };
            }
            if (ch === 'p' || ch === 'P') {
                return { execute: () => applyPaste(ch === 'p', count) };
            }
            if (ch === 'D') {
                return { execute: () => applyOperatorMotion('delete', '$', 1) };
            }
            if (ch === 'C') {
                return { execute: () => applyOperatorMotion('change', '$', 1) };
            }
            if (ch === 'Y') {
                return { execute: () => applyLinewiseOperator('yank', count) };
            }
            if (ch === 'G') {
                return { execute: () => setCursorOffset(executeMotion('G', count)) };
            }
            if (ch === '.') {
                return { execute: () => repeatLastChange() };
            }
            if (ch === ';' || ch === ',') {
                return { execute: () => repeatLastFind(ch === ',', count) };
            }
            if (ch === 'u') {
                return { execute: () => onUndo?.() };
            }
            if (ch === 'i') {
                return { execute: () => enterInsert(cursorOffset) };
            }
            if (ch === 'I') {
                return {
                    execute: () => {
                        const { start, end } = getLogicalLineBounds(cursorOffset);
                        const lineText = value.slice(start, end);
                        const firstNonBlank = lineText.search(/\S/);
                        enterInsert(start + (firstNonBlank === -1 ? 0 : firstNonBlank));
                    }
                };
            }
            if (ch === 'a') {
                return {
                    execute: () => {
                        const nextOffset = cursorOffset >= value.length ? cursorOffset : cursorOffset + 1;
                        enterInsert(nextOffset);
                    }
                };
            }
            if (ch === 'A') {
                return {
                    execute: () => {
                        const { end } = getLogicalLineBounds(cursorOffset);
                        enterInsert(end);
                    }
                };
            }
            if (ch === 'o') {
                return { execute: () => applyOpenLine('below') };
            }
            if (ch === 'O') {
                return { execute: () => applyOpenLine('above') };
            }
            if (ch === 'v') {
                return {
                    execute: () => {
                        setVimMode('VISUAL');
                        setVisualStart(cursorOffset);
                    }
                };
            }
            if (ch === ':') {
                return { next: { type: 'colon', buffer: '' } };
            }
            if (ch === 'q') {
                if (macros.isRecording) {
                    macros.stopRecording();
                } else {
                    return { next: { type: 'macroRecord' } };
                }
                return { execute: () => {} };
            }
            if (ch === '@') {
                return { next: { type: 'macroPlay' } };
            }
            return null;
        };

        const handleOperatorCommand = (op: VimOperator, count: number, ch: string): { execute?: () => void; next?: VimCommandState } | null => {
            if (TEXT_OBJECT_SCOPES[ch]) {
                return { next: { type: 'operatorTextObj', operator: op, count, scope: TEXT_OBJECT_SCOPES[ch] } };
            }
            if (FIND_MOTIONS.has(ch)) {
                return { next: { type: 'operatorFind', operator: op, count, findChar: ch } };
            }
            if (MOTIONS.has(ch)) {
                return { execute: () => applyOperatorMotion(op, ch, count) };
            }
            if (ch === 'G') {
                return { execute: () => applyOperatorToLine(op, 'G', count) };
            }
            if (ch === 'g') {
                return { next: { type: 'operatorG', operator: op, count } };
            }
            return null;
        };

        let result: { execute?: () => void; next?: VimCommandState } | null = null;

        if (state.type === 'idle') {
            if (/[1-9]/.test(mappedInput)) {
                result = { next: { type: 'count', digits: mappedInput } };
            } else if (mappedInput === '0') {
                result = { execute: () => setCursorOffset(executeMotion('0', 1)) };
            } else {
                result = handleIdleCommand(mappedInput, 1);
            }
        } else if (state.type === 'count') {
            if (/[0-9]/.test(mappedInput)) {
                const digits = String(clampCount(parseInt(state.digits + mappedInput, 10)));
                result = { next: { type: 'count', digits } };
            } else {
                const count = parseInt(state.digits, 10);
                result = handleIdleCommand(mappedInput, count) || { next: { type: 'idle' } };
            }
        } else if (state.type === 'operator') {
            const operatorKey = state.operator[0];
            if (mappedInput === operatorKey) {
                result = { execute: () => applyLinewiseOperator(state.operator as VimOperator, state.count) };
            } else if (/[0-9]/.test(mappedInput)) {
                result = { next: { type: 'operatorCount', operator: state.operator, count: state.count, digits: mappedInput } };
            } else {
                result = handleOperatorCommand(state.operator as VimOperator, state.count, mappedInput) || { next: { type: 'idle' } };
            }
        } else if (state.type === 'operatorCount') {
            if (/[0-9]/.test(mappedInput)) {
                const digits = String(clampCount(parseInt(state.digits + mappedInput, 10)));
                result = { next: { ...state, digits } };
            } else {
                const count = state.count * parseInt(state.digits, 10);
                result = handleOperatorCommand(state.operator as VimOperator, count, mappedInput) || { next: { type: 'idle' } };
            }
        } else if (state.type === 'find') {
            result = {
                execute: () => {
                    const nextOffset = findCharacter(mappedInput, state.findChar, state.count, cursorOffset);
                    if (nextOffset !== null) {
                        setCursorOffset(nextOffset);
                        internal.lastFind = { type: state.findChar, char: mappedInput };
                    }
                }
            };
        } else if (state.type === 'g') {
            if (mappedInput === 'g') {
                result = { execute: () => setCursorOffset(executeMotion('gg', state.count)) };
            } else {
                result = { next: { type: 'idle' } };
            }
        } else if (state.type === 'operatorG') {
            if (mappedInput === 'g') {
                result = { execute: () => applyOperatorToLine(state.operator as VimOperator, 'gg', state.count) };
            } else {
                result = { next: { type: 'idle' } };
            }
        } else if (state.type === 'operatorFind') {
            result = { execute: () => applyOperatorFind(state.operator as VimOperator, state.findChar, mappedInput, state.count) };
        } else if (state.type === 'operatorTextObj') {
            if (TEXT_OBJECTS.has(mappedInput)) {
                result = { execute: () => applyOperatorTextObj(state.operator as VimOperator, state.scope, mappedInput, state.count) };
            } else {
                result = { next: { type: 'idle' } };
            }
        } else if (state.type === 'replace') {
            if (mappedInput.length === 1) {
                result = { execute: () => applyReplaceChar(mappedInput, state.count) };
            } else {
                result = { next: { type: 'idle' } };
            }
        } else if (state.type === 'indent') {
            if (mappedInput === state.operator) {
                result = { execute: () => applyIndent(state.operator as '>' | '<', state.count) };
            } else {
                result = { next: { type: 'idle' } };
            }
        }

        if (result?.execute) {
            result.execute();
        }
        if (result?.next) {
            commandStateRef.current = result.next;
        } else if (result?.execute) {
            resetState();
        }

        if (mappedInput === '?' && state.type === 'idle') {
            onChange('?');
        }
    }, [
        value,
        cursorOffset,
        setCursorOffset,
        onChange,
        onSubmit,
        onExit,
        onUndo,
        getPosition,
        getOffset,
        getLogicalLineBounds,
        findCharacter,
        moveNextVimWord,
        movePrevVimWord,
        moveEndOfVimWord,
        moveNextWORD,
        movePrevWORD,
        moveEndOfWORD,
        getTextObjectRange,
        performOperator,
        macros,
        enterInsert,
        setVimMode
    ]);

    const handleVisualModeInput = useCallback((key: any, input: string) => {
        // Very basic visual mode for now
        if (input === 'h' || key.leftArrow) setCursorOffset(Math.max(0, cursorOffset - 1));
        if (input === 'l' || key.rightArrow) setCursorOffset(Math.min(value.length - 1, cursorOffset + 1));
        if (input === 'j' || key.downArrow) {
            const { line, column } = getPosition(cursorOffset);
            setCursorOffset(getOffset(line + 1, column));
        }
        if (input === 'k' || key.upArrow) {
            const { line, column } = getPosition(cursorOffset);
            setCursorOffset(getOffset(line - 1, column));
        }

        if (input === 'y' || input === 'd' || input === 'x') {
            if (visualStart !== null) {
                const from = Math.min(visualStart, cursorOffset);
                const to = Math.max(visualStart, cursorOffset) + 1;
                const selectedText = value.slice(from, to);
                internalVimStateRef.current.register = selectedText;
                internalVimStateRef.current.registerIsLinewise = false;

                if (input === 'd' || input === 'x') {
                    onChange(value.slice(0, from) + value.slice(to));
                    setCursorOffset(from);
                }

                setVisualStart(null);
                enterNormal();
            }
        }
        if (input === 'i' || input === 'a') {
            // Text objects in visual mode (not implemented)
        }
    }, [value, cursorOffset, visualStart, setCursorOffset, onChange, getPosition, getOffset, enterNormal]);

    const handleKey = useCallback((input: string, key: any) => {
        if (!enabled) return false;

        if (mode === 'INSERT') {
            if (key.escape) {
                const insertedText = insertStateRef.current.insertedText;
                if (insertedText) {
                    internalVimStateRef.current.lastChange = { type: 'insert', text: insertedText };
                }
                let nextOffset = cursorOffset;
                if (nextOffset > 0 && value[nextOffset - 1] !== '\n') {
                    nextOffset = nextOffset - 1;
                }
                insertStateRef.current.insertedText = '';
                commandStateRef.current = { type: 'idle' };
                enterNormal(nextOffset);
                return true;
            }
            if (!key.ctrl && !key.meta) {
                if (key.backspace || key.delete) {
                    if (insertStateRef.current.insertedText.length > 0) {
                        insertStateRef.current.insertedText = insertStateRef.current.insertedText.slice(0, -1);
                    }
                } else if (input && !key.return) {
                    insertStateRef.current.insertedText += input;
                }
            }
            return false;
        }

        if (mode === 'VISUAL' && key.escape) {
            setVisualStart(null);
            enterNormal(cursorOffset);
            return true;
        }

        if (mode === 'NORMAL') {
            handleNormalModeKey(key, input);
            return true;
        }

        if (mode === 'VISUAL') {
            handleVisualModeInput(key, input);
            return true;
        }

        return false;
    }, [enabled, mode, cursorOffset, value, handleNormalModeKey, handleVisualModeInput, enterNormal]);

    return {
        mode,
        setMode: setVimMode,
        handleKey
    };
}
