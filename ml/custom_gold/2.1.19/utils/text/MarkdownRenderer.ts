import { Lexer } from 'marked';
import { highlight } from 'cli-highlight';
import chalk from 'chalk';

interface RenderOptions {
    width?: number;
    showLineNumbers?: boolean;
}

export class TerminalMarkdownRenderer {
    private lexer: Lexer;

    constructor(private options: RenderOptions = {}) {
        this.lexer = new Lexer();
    }

    public parse(content: string): any[] {
        return this.lexer.lex(content);
    }

    public highlightCode(code: string, lang: string): string {
        try {
            return highlight(code, {
                language: lang || 'plaintext',
                ignoreIllegals: true
            });
        } catch {
            return code;
        }
    }

    /**
     * More robust inline formatter using 'marked' to correctly handle nested styles.
     */
    public formatInline(text: string): string {
        const tokens = this.lexer.inlineTokens(text);
        return this.renderInlineTokens(tokens);
    }

    private renderInlineTokens(tokens: any[]): string {
        return tokens.map(token => {
            switch (token.type) {
                case 'strong':
                    return chalk.bold(this.renderInlineTokens(token.tokens || []));
                case 'em':
                    return chalk.italic(this.renderInlineTokens(token.tokens || []));
                case 'codespan':
                    return chalk.cyan(token.text);
                case 'link':
                    return chalk.blue.underline(token.text);
                case 'text':
                    return token.text;
                case 'escape':
                    return token.text;
                case 'br':
                    return '\n';
                default:
                    return token.text || '';
            }
        }).join('');
    }

    /**
     * Helper to render lists and other block elements as strings if needed.
     */
    public renderBlockAsString(token: any): string {
        if (token.type === 'list') {
            return token.items.map((item: any, i: number) => {
                const bullet = token.ordered ? `${i + 1}. ` : '• ';
                return `${bullet}${this.formatInline(item.text)}`;
            }).join('\n');
        }
        return token.text || '';
    }
}

export const markdownRenderer = new TerminalMarkdownRenderer();
