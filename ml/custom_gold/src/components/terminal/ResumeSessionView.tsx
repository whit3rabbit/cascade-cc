
// Logic from chunk_596.ts (Resume Session UI)

import React from "react";
import { Box, Text } from "ink";

/**
 * Shown when a previous session is detected and can be resumed.
 */
export function ResumeSessionView({ command }: { command: string }) {
    return (
        <Box flexDirection="column" borderStyle="round" borderColor="claude" paddingX={1}>
            <Text bold color="claude">Previous session detected</Text>
            <Box marginY={1}>
                <Text>To resume, run:</Text>
                <Text bold color="suggestion"> {command}</Text>
            </Box>
            <Text dimColor>(Command copied to clipboard)</Text>
        </Box>
    );
}
