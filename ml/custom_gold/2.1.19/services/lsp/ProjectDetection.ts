import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

export interface ProjectType {
    type: string;
    description: string;
    recommendedLsp: string;
    configFile: string;
}

const PROJECT_TYPES: ProjectType[] = [
    {
        type: 'typescript',
        description: 'TypeScript Project',
        recommendedLsp: 'typescript-language-server',
        configFile: 'package.json'
    },
    {
        type: 'go',
        description: 'Go Project',
        recommendedLsp: 'gopls',
        configFile: 'go.mod'
    },
    {
        type: 'rust',
        description: 'Rust Project',
        recommendedLsp: 'rust-analyzer',
        configFile: 'Cargo.toml'
    },
    {
        type: 'python',
        description: 'Python Project',
        recommendedLsp: 'pyright',
        configFile: 'pyproject.toml'
    },
    {
        type: 'python-requirements',
        description: 'Python Project',
        recommendedLsp: 'pyright',
        configFile: 'requirements.txt'
    },
    {
        type: 'java-maven',
        description: 'Java Project (Maven)',
        recommendedLsp: 'jdtls',
        configFile: 'pom.xml'
    },
    {
        type: 'java-gradle',
        description: 'Java Project (Gradle)',
        recommendedLsp: 'jdtls',
        configFile: 'build.gradle'
    }
];

export async function detectProjectType(rootPath: string): Promise<ProjectType | null> {
    for (const project of PROJECT_TYPES) {
        if (existsSync(join(rootPath, project.configFile))) {
            return project;
        }
    }
    return null;
}
