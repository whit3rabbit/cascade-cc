import fs from "fs";
import path from "path";
import { generateSlug } from "../../utils/shared/idUtils.js";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";

// In-memory cache for plan paths
const planPathCache = new Map<string, string>();

/**
 * Gets or creates a plan file path for the given context.
 */
export function getPlanPath(contextId: string): string {
    // If contextId is provided, check if we've already cached a plan for it
    if (!contextId) {
        contextId = process.cwd();
    }

    // Check global cache
    if (planPathCache.has(contextId)) {
        return planPathCache.get(contextId)!;
    }

    const plansDir = getPlansDir();
    let slug = "";

    // Try to find an unused slug
    for (let i = 0; i < 10; i++) {
        slug = generateSlug();
        const planPath = path.join(plansDir, `${slug}.md`);
        if (!fs.existsSync(planPath)) {
            break;
        }
    }

    planPathCache.set(contextId, slug);
    return slug;
}

export function cachePlanPath(contextId: string, slug: string): void {
    planPathCache.set(contextId, slug);
}

function getPlansDir(): string {
    const configDir = getConfigDir();
    const plansDir = path.join(configDir, "plans");

    if (!fs.existsSync(plansDir)) {
        try {
            fs.mkdirSync(plansDir, { recursive: true });
        } catch (error) {
            console.error(error);
        }
    }
    return plansDir;
}

function resolvePlanPath(slug: string): string {
    return path.join(getPlansDir(), `${slug}.md`);
}

/**
 * Loads a plan from disk by its slug.
 */
export function loadPlan(slug: string): string | null {
    const filePath = resolvePlanPath(slug);
    if (!fs.existsSync(filePath)) return null;

    try {
        return fs.readFileSync(filePath, "utf-8");
    } catch (error) {
        console.error(error);
        return null;
    }
}

/**
 * Checks if a plan exists for the messages (checking for slug property).
 */
export function hasPlan(completion: any): boolean {
    const slug = completion.messages?.find((m: any) => m.slug)?.slug;
    if (!slug) return false;

    cachePlanPath(process.cwd(), slug);
    const planPath = resolvePlanPath(slug);
    return fs.existsSync(planPath);
}

// --- Task Persistence ---

function getTasksDir(): string {
    const cwd = process.cwd();
    // Assuming .claude directory structure in project root
    return path.join(cwd, ".claude", "tasks");
}

function ensureTasksDir(): void {
    const tasksDir = getTasksDir();
    if (!fs.existsSync(tasksDir)) {
        fs.mkdirSync(tasksDir, { recursive: true });
    }
}

export function getTaskOutputPath(taskId: string): string {
    return path.join(getTasksDir(), `${taskId}.output`);
}

/**
 * Appends output to a task's output file.
 */
export function appendTaskOutput(taskId: string, output: string): void {
    try {
        ensureTasksDir();
        const outputPath = getTaskOutputPath(taskId);
        const outputDir = path.dirname(outputPath);

        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        fs.appendFileSync(outputPath, output, "utf8");
    } catch (error) {
        console.error(error);
    }
}

/**
 * Reads task output starting from a specific offset.
 */
export function readTaskOutput(taskId: string, offset: number): { content: string; newOffset: number } {
    try {
        const outputPath = getTaskOutputPath(taskId);
        if (!fs.existsSync(outputPath)) {
            return { content: "", newOffset: offset };
        }

        const stats = fs.statSync(outputPath);
        if (stats.size <= offset) {
            return { content: "", newOffset: offset };
        }

        const content = fs.readFileSync(outputPath, "utf8").slice(offset);
        return { content, newOffset: stats.size };
    } catch (error) {
        console.error(error);
        return { content: "", newOffset: offset };
    }
}

export function getAllTaskOutput(taskId: string): string {
    try {
        const outputPath = getTaskOutputPath(taskId);
        if (!fs.existsSync(outputPath)) return "";
        return fs.readFileSync(outputPath, "utf8");
    } catch (error) {
        console.error(error);
        return "";
    }
}

export function initTaskOutput(taskId: string): string {
    ensureTasksDir();
    const outputPath = getTaskOutputPath(taskId);
    if (!fs.existsSync(outputPath)) {
        fs.writeFileSync(outputPath, "", "utf8");
    }
    return outputPath;
}

export function cleanTaskOutputs(): void {
    try {
        const tasksDir = getTasksDir();
        if (!fs.existsSync(tasksDir)) return;

        const files = fs.readdirSync(tasksDir);
        for (const file of files) {
            if (file.endsWith(".output")) {
                try {
                    fs.unlinkSync(path.join(tasksDir, file));
                } catch { }
            }
        }
    } catch { }
}


export function getTaskIdFromPath(filePath: string): string | null {
    const tasksDir = getTasksDir() + "/";
    if (filePath.startsWith(tasksDir) && filePath.endsWith(".output")) {
        const id = filePath.slice(tasksDir.length, -7); // remove .output
        // Validate ID format (basic alphanum check from original code)
        if (id.length > 0 && id.length <= 20 && /^[a-zA-Z0-9_-]+$/.test(id)) {
            return id;
        }
    }
    return null;
}

/**
 * Helper to find the project root (where .claude or CLAUDE.md resides).
 */
export function getProjectRoot(startDir: string): string {
    let currentDir = startDir;
    while (true) {
        if (fs.existsSync(path.join(currentDir, ".claude")) || fs.existsSync(path.join(currentDir, "CLAUDE.md"))) {
            return currentDir;
        }
        const parentDir = path.dirname(currentDir);
        if (parentDir === currentDir) {
            // Reached root
            return startDir; // Fallback to startDir if not found
        }
        currentDir = parentDir;
    }
}
