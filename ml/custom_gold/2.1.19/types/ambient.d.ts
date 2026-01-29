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
        constructor(options?: any);
        lex(content: string): any[];
    }
}

declare module 'cli-highlight';
