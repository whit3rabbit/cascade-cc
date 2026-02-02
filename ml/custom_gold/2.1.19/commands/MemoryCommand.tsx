import React from 'react';
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

interface MemoryCommandProps {
    cwd: string;
    onDone: (message: string, options?: { display: 'system' }) => void;
}

export const MemoryCommand: React.FC<MemoryCommandProps> = ({
    cwd,
    onDone
}) => {
    React.useEffect(() => {
        const memoryPath = join(cwd, 'MEMORY.md');
        if (existsSync(memoryPath)) {
            try {
                const content = readFileSync(memoryPath, 'utf8');
                onDone(`**Current MEMORY.md Content:**\n\n${content}`, { display: 'system' });
            } catch (err: any) {
                onDone(`Error reading MEMORY.md: ${err.message}`, { display: 'system' });
            }
        } else {
            onDone('_MEMORY.md not found in current directory._', { display: 'system' });
        }
    }, [cwd]);

    return null;
};
