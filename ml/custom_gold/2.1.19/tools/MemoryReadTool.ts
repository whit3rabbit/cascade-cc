/**
 * File: src/tools/MemoryReadTool.ts
 * Role: Reads from the persistent memory store.
 */

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { getBaseConfigDir } from '../utils/shared/runtimeAndEnv.js';

export class MemoryReadTool {
    static name = "MemoryReadTool";

    async call(): Promise<any> {
        const memoryPath = join(getBaseConfigDir(), 'memory.json');
        if (!existsSync(memoryPath)) {
            return { memory: {} };
        }

        try {
            const content = readFileSync(memoryPath, 'utf8');
            return { memory: JSON.parse(content) };
        } catch (error) {
            return { error: `Failed to read memory: ${error instanceof Error ? error.message : String(error)}` };
        }
    }
}
