declare module 'micromatch' {
    export function isMatch(path: string, patterns: string | string[], options?: any): boolean;
    export default isMatch;
}

declare module 'gray-matter' {
    const matter: any;
    export default matter;
}

declare module 'marked' {
    export class Lexer {
        static lex(content: string): any[];
        static lexInline(text: string): any[];
        constructor(options?: any);
        lex(content: string): any[];
        inlineTokens(text: string): any[];
    }
    export const marked: {
        lexer(content: string): any[];
    };
}

declare module 'cli-highlight';
