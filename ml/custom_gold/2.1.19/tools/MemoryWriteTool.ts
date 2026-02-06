/**
 * File: src/tools/MemoryWriteTool.ts
 * Role: Writes to the persistent memory store.
 */

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { getBaseConfigDir } from '../utils/shared/runtimeAndEnv.js';

export interface MemoryWriteInput {
    key: string;
    value: any;
}

export class MemoryWriteTool {
    static name = "MemoryWriteTool";

    async call(input: MemoryWriteInput): Promise<any> {
        const { key, value } = input;
        const memoryPath = join(getBaseConfigDir(), 'memory.json');
        let memory: Record<string, any> = {};

        if (existsSync(memoryPath)) {
            try {
                const content = readFileSync(memoryPath, 'utf8');
                memory = JSON.parse(content);
            } catch {
                console.warn("[MemoryWriteTool] Failed to read existing memory, starting fresh.");
            }
        }

        memory[key] = value;

        try {
            writeFileSync(memoryPath, JSON.stringify(memory, null, 2));
            return { success: true, key };
        } catch (error) {
            return { error: `Failed to write memory: ${error instanceof Error ? error.message : String(error)}` };
        }
    }
}
