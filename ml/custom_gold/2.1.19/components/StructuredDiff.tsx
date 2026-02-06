
import React, { useMemo } from 'react';
import { Box, Text } from 'ink';
import * as Diff from 'diff';

interface StructuredDiffProps {
    filePath?: string;
    oldContent?: string;
    newContent?: string;
    // Alternative: pass direct hunks if we already have them
    hunks?: any[];
    mode?: 'diff' | 'write';
    maxPreviewLines?: number;
}

const DEFAULT_PREVIEW_LINES = 10;

function normalizePreviewLines(content: string): string[] {
    return content.split(/\r\n|\n|\r/);
}

function renderPreviewLine(line: string): string {
    // Preserve empty lines in previews by rendering a single space.
    return line === '' ? ' ' : line;
}

export const StructuredDiff: React.FC<StructuredDiffProps> = ({
    filePath,
    oldContent = '',
    newContent = '',
    hunks,
    mode = 'diff',
    maxPreviewLines = DEFAULT_PREVIEW_LINES
}) => {
    const diffHunks = useMemo(() => {
        if (mode === 'write') return [];
        if (hunks) return hunks;
        // Create a patch. FileName, OldFileName, OldStr, NewStr, OldHeader, NewHeader, Context
        // We probably just want the structural differences.
        // If we just want visual diff of two strings:
        if (!oldContent && !newContent) return [];

        // We use createPatch to get standard unified diff format which handles context lines automatically (default 4)
        const patchStr = Diff.createPatch(filePath || 'file', oldContent, newContent, '', '', { context: 3 });
        return Diff.parsePatch(patchStr);
    }, [oldContent, newContent, filePath, hunks, mode]);

    if (mode === 'write') {
        const hasContent = newContent !== undefined && newContent !== '';
        const lines = hasContent ? normalizePreviewLines(newContent) : [];
        const previewLines = lines.slice(0, maxPreviewLines);
        const remaining = Math.max(0, lines.length - previewLines.length);

        return (
            <Box flexDirection="column" borderStyle="single" borderColor="gray" paddingX={1}>
                <Text bold underline>{filePath || 'Unknown file'}</Text>
                {hasContent ? (
                    previewLines.map((line, i) => (
                        <Text key={i} dimColor wrap="wrap">
                            {renderPreviewLine(line)}
                        </Text>
                    ))
                ) : (
                    <Text dimColor>(No content)</Text>
                )}
                {remaining > 0 && (
                    <Text dimColor>… +{remaining} lines</Text>
                )}
            </Box>
        );
    }

    return (
        <Box flexDirection="column" borderStyle="single" borderColor="gray" paddingX={1}>
            <Text bold underline>{filePath || 'Unknown file'}</Text>
            {diffHunks.map((hunk: any, i: number) => (
                <Box key={i} flexDirection="column" marginTop={0}>
                    {hunk.hunks.map((block: any, j: number) => (
                        <Box key={j} flexDirection="column" marginTop={j > 0 ? 1 : 0}>
                            <Text dimColor>
                                @@ -{block.oldStart},{block.oldLines} +{block.newStart},{block.newLines} @@
                            </Text>
                            {block.lines.map((line: string, k: number) => {
                                const isAdd = line.startsWith('+');
                                const isRem = line.startsWith('-');
                                const color = isAdd ? 'green' : (isRem ? 'red' : undefined);
                                return (
                                    <Text key={k} color={color} wrap="wrap">
                                        {line.replace(/\n/g, '↵')}
                                    </Text>
                                );
                            })}
                        </Box>
                    ))}
                </Box>
            ))}
            {diffHunks.length === 0 && (
                <Text dimColor>No changes detected</Text>
            )}
        </Box>
    );
};
