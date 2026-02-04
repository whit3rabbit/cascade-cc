import { ProjectConfigHandler, ProjectConfig } from './ProjectConfigHandler.js';

class ConfigManager {
    private static instance: ConfigManager;
    private handler: ProjectConfigHandler | null = null;

    private constructor() { }

    static getInstance(): ConfigManager {
        if (!ConfigManager.instance) {
            ConfigManager.instance = new ConfigManager();
        }
        return ConfigManager.instance;
    }

    initialize(cwd: string) {
        if (this.handler) {
            this.handler.dispose();
        }
        this.handler = new ProjectConfigHandler(cwd);
    }

    getConfig(): ProjectConfig {
        if (!this.handler) {
            return {};
        }
        return this.handler.getConfig();
    }

    dispose() {
        if (this.handler) {
            this.handler.dispose();
            this.handler = null;
        }
    }
}

export const configManager = ConfigManager.getInstance();
