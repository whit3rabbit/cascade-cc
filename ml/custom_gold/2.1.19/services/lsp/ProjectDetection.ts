import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';

const execAsync = promisify(exec);

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

/**
 * Generates a summary of the project, including type, version, and Git info.
 * Refined using Git logic from chunk1271.
 */
export async function getProjectSummary(rootPath: string): Promise<string> {
    const project = await detectProjectType(rootPath);
    const summaryArr: string[] = [];

    if (project) {
        summaryArr.push(project.description);
    }

    // Try to get info from package.json
    if (existsSync(join(rootPath, 'package.json'))) {
        try {
            const pkg = JSON.parse(readFileSync(join(rootPath, 'package.json'), 'utf-8'));
            if (pkg.name) summaryArr.push(pkg.name);
            if (pkg.version) summaryArr.push(`v${pkg.version}`);
        } catch (e) { }
    }

    // Check git via command line for richer info (as done in chunk1271)
    try {
        const { stdout: isRepo } = await execAsync('git rev-parse --is-inside-work-tree', { cwd: rootPath });
        if (isRepo.trim() === 'true') {
            try {
                const { stdout: remoteUrl } = await execAsync('git remote get-url origin', { cwd: rootPath });
                const match = remoteUrl.trim().match(/github\.com[:/]([^/]+\/[^/]+)(\.git)?$/);
                if (match) {
                    summaryArr.push(`GitHub: ${match[1].replace(/\.git$/, '')}`);
                } else {
                    summaryArr.push("Git repo");
                }
            } catch {
                summaryArr.push("Local Git repo");
            }
        }
    } catch {
        if (existsSync(join(rootPath, '.git'))) {
            summaryArr.push("(Git detected)");
        }
    }

    return summaryArr.length > 0 ? summaryArr.join(' - ') : "Unknown Project";
}
