
// Logic from chunks 612-619 (Syntax Highlighting)

import hljs from "highlight.js";

/**
 * Provides syntax highlighting for various languages supported by Claude Code.
 */
export const SyntaxHighlightService = {
    /**
     * Highlights a block of code using highlight.js.
     */
    highlight(code: string, language?: string): string {
        if (language && hljs.getLanguage(language)) {
            return hljs.highlight(code, { language }).value;
        }
        return hljs.highlightAuto(code).value;
    },

    /**
     * Registers additional language definitions (like GML/ISBL).
     */
    registerLanguages() {
        // Logic to register custom language definitions from chunks
        // hljs.registerLanguage('gml', gmlDefinition);
    }
};
