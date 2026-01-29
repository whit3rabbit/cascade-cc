/**
 * File: src/utils/text/MarkdownRenderer.ts
 * Role: Custom Markdown renderer based on the original 2.1.19 chunk620 logic.
 * Adapts 'marked' to work with Ink components.
 */

import { Lexer } from 'marked';
import { highlight } from 'cli-highlight';
import chalk from 'chalk';

// Types for our custom renderer
interface RenderOptions {
    width?: number;
    showLineNumbers?: boolean;
}

/**
 * Custom renderer class that mimics the structure seen in chunk620.
 * It primarily handles code blocks with highlighting and text formatting.
 */
export class TerminalMarkdownRenderer {
    constructor(private options: RenderOptions = {}) { }

    /**
     * Renders markdown content to a string format suitable for Ink Text/Box components.
     * Note: Ink doesn't use HTML, so we parse tokens and return a structure
     * that MessageRenderer can use to create component trees.
     */
    public parse(content: string): ReturnType<Lexer['lex']> {
        const lexer = new Lexer();
        return lexer.lex(content);
    }

    /**
     * Highlights code blocks using cli-highlight (as seen in the deobfuscated code's dependencies).
     */
    public highlightCode(code: string, lang: string): string {
        try {
            return highlight(code, {
                language: lang || 'plaintext',
                ignoreIllegals: true
            });
        } catch (e) {
            return code;
        }
    }

    /**
     * Formats inline styles (bold, italic, code) using chalk.
     */
    public formatInline(text: string): string {
        // Simple regex-based formatter for demonstration
        // iterating on tokens would be more robust but this covers 90%
        return text
            .replace(/\*\*(.*?)\*\*/g, (_, p1) => chalk.bold(p1))
            .replace(/\*(.*?)\*/g, (_, p1) => chalk.italic(p1))
            .replace(/`([^`]+)`/g, (_, p1) => chalk.cyan(p1));
    }
}

export const markdownRenderer = new TerminalMarkdownRenderer();
