/**
 * File: src/utils/fs/FileParser.ts
 * Role: Utilities for parsing files (JSON, YAML, Text, Binary) for use in tools.
 */

import { load as loadYaml } from 'js-yaml';
import { parse as parseJson5 } from 'json5';

const SUPPORTED_TEXT_TYPES = /\.(txt|htm|html|md|xml|js|json|yaml|yml|css|scss|less|svg)$/i;
const SUPPORTED_IMAGE_TYPES = /\.(jpeg|jpg|gif|png|bmp|ico)$/i;

export interface ToolHandler {
    canParse: (ext: string) => boolean;
    parse: (data: Buffer | string) => any;
}

export const FileParser = {
    /**
     * Parses file data based on its filename extension.
     */
    parse(fileData: Buffer, fileName: string): any {
        const ext = fileName.split('.').pop()?.toLowerCase();

        try {
            if (ext === 'json') {
                return JSON.parse(fileData.toString('utf8'));
            }
            if (ext === 'json5') {
                return parseJson5(fileData.toString('utf8'));
            }
            if (ext === 'yaml' || ext === 'yml') {
                return loadYaml(fileData.toString('utf8'));
            }

            if (SUPPORTED_TEXT_TYPES.test(fileName)) {
                return fileData.toString('utf8');
            }

            if (SUPPORTED_IMAGE_TYPES.test(fileName)) {
                return fileData; // Return as buffer for images
            }
        } catch (err: any) {
            console.warn(`[FileParser] Failed to parse ${fileName}: ${err.message}`);
        }

        return fileData; // Fallback to raw buffer
    }
};

/**
 * Legacy handlers from deobfuscated code, preserved for compatibility.
 */
export const UserToolHandlers: Record<string, ToolHandler> = {
    json: {
        canParse: (ext: string) => ext === '.json' || ext === '.json5',
        parse: (data: Buffer | string) => parseJson5(data.toString())
    },
    yaml: {
        canParse: (ext: string) => ext === '.yaml' || ext === '.yml',
        parse: (data: Buffer | string) => loadYaml(data.toString())
    }
};
