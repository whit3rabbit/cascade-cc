/**
 * Character-count based content truncation logic, aligned with the original 2.1.19 implementation.
 */

export const CONTENT_TRUNCATION_LIMIT = 10000;

export interface TruncationResult {
    totalLines: number;
    truncatedContent: string;
    isImage: boolean;
}

export function isBase64Image(content: string): boolean {
    return /^data:image\/[a-z0-9.+_-]+;base64,/i.test(content);
}

/**
 * Truncates content based on character count.
 * Mirrored from QP1 in chunk1125.
 */
export function truncateContent(content: string, limit: number = CONTENT_TRUNCATION_LIMIT): TruncationResult {
    const isImage = isBase64Image(content);
    if (isImage) {
        return {
            totalLines: 1,
            truncatedContent: content,
            isImage
        };
    }

    const totalLines = content.split('\n').length;

    if (content.length <= limit) {
        return {
            totalLines,
            truncatedContent: content,
            isImage
        };
    }

    const visibleContent = content.slice(0, limit);
    const truncatedLines = content.slice(limit).split('\n').length;

    // Original format: ... [X lines truncated] ...
    const truncatedContent = `${visibleContent}\n\n... [${truncatedLines} lines truncated] ...`;

    return {
        totalLines,
        truncatedContent,
        isImage
    };
}
