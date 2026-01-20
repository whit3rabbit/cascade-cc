export class MaxFileReadTokenExceededError extends Error {
    tokenCount: number;
    maxTokens: number;

    constructor(tokenCount: number, maxTokens: number) {
        super(
            `File content (${tokenCount} tokens) exceeds maximum allowed tokens (${maxTokens}). Please use offset and limit parameters to read specific portions of the file, or use the GrepTool to search for specific content.`
        );
        this.tokenCount = tokenCount;
        this.maxTokens = maxTokens;
        this.name = "MaxFileReadTokenExceededError";
    }
}
