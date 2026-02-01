
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync, promises as fs } from 'fs';
import { createRequire } from 'module';
import { EnvService } from '../../services/config/EnvService.js';

const require = createRequire(import.meta.url);
const Parser = require('web-tree-sitter');

let parser: any;
const languages = new Map<string, any>();

export async function initTreeSitter() {
    if (parser) return;

    const __dirname = dirname(fileURLToPath(import.meta.url));
    // Check root and dist locations for wasm files
    const possiblePaths = [
        join(__dirname, '../../../assets'), // assets from src/utils/shared
        join(__dirname, '../../../../assets'), // assets from dist/utils/shared
        join(process.cwd(), 'assets'),
        join(__dirname, '../../..'), // root from src/utils/shared
        join(__dirname, '../../../../'), // root from dist/utils/shared (bundled)
        process.cwd()
    ];

    let wasmPath: string | undefined;
    let bashWasmPath: string | undefined;

    for (const p of possiblePaths) {
        if (existsSync(join(p, 'tree-sitter.wasm'))) {
            wasmPath = join(p, 'tree-sitter.wasm');
        }
        if (existsSync(join(p, 'tree-sitter-bash.wasm'))) {
            bashWasmPath = join(p, 'tree-sitter-bash.wasm');
        }
    }

    if (!wasmPath || !bashWasmPath) {
        console.warn('Tree-sitter WASM files not found. Shell parsing functionality will be limited.');
        return;
    }

    await (Parser as any).init({
        locateFile: () => wasmPath!
    });

    parser = new (Parser as any)();

    // Load available languages
    const skipFiles = ['tree-sitter.wasm'];
    for (const p of possiblePaths) {
        if (!existsSync(p)) continue;
        const files = await fs.readdir(p);
        for (const file of files) {
            if (file.startsWith('tree-sitter-') && file.endsWith('.wasm') && !skipFiles.includes(file)) {
                const langName = file.replace('tree-sitter-', '').replace('.wasm', '');
                try {
                    const langPath = join(p, file);
                    const language = await (Parser as any).Language.load(langPath);
                    languages.set(langName, language);
                } catch (e) {
                    if (EnvService.isTruthy("DEBUG_TREESITTER")) {
                        console.error(`[TreeSitter] Failed to load language ${langName}:`, e);
                    }
                }
            }
        }
    }

    // Default to bash if available
    if (languages.has('bash')) {
        parser.setLanguage(languages.get('bash'));
    }
}

export function parse(code: string, language?: string) {
    if (!parser) return null;
    const lang = language ? languages.get(language) : languages.get('bash');
    if (lang) {
        parser.setLanguage(lang);
    }
    return parser.parse(code);
}

export function parseBash(script: string) {
    return parse(script, 'bash');
}

export { Parser };
