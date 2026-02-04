import { join } from 'path';
import { readFile } from 'fs/promises';
import { Watcher } from '../../utils/fs/Watcher.js';

export interface ProjectConfig {
    allowedTools?: string[];
    disallowedTools?: string[];
    // Add other known config fields here
    [key: string]: any;
}

export class ProjectConfigHandler {
    private config: ProjectConfig = {};
    private watcher: Watcher;
    private configPath: string;

    constructor(cwd: string) {
        this.configPath = join(cwd, '.claude.json');
        this.watcher = new Watcher();
        this.init();
    }

    private async init() {
        await this.loadConfig();
        this.watcher.watch(this.configPath);
        this.watcher.on('change', () => {
            console.log('[Config] Reloading .claude.json due to change');
            this.loadConfig();
        });
    }

    private async loadConfig() {
        try {
            const content = await readFile(this.configPath, 'utf-8');
            this.config = JSON.parse(content);
        } catch (error: any) {
            // Ignore ENOENT (file not found), log others
            if (error.code !== 'ENOENT') {
                console.error(`[Config] Failed to load .claude.json: ${error.message}`);
            } else {
                this.config = {};
            }
        }
    }

    public getConfig(): ProjectConfig {
        return this.config;
    }

    public dispose() {
        this.watcher.close();
    }
}
