import { z } from "zod";
import * as fs from "node:fs";
import * as path from "node:path";
import { getProjectRoot } from "../../utils/shared/pathUtils.js";
// In chunk call t1(), likely getting project root.

const SessionIndexEntrySchema = z.object({
    sessionId: z.string(),
    leafUuid: z.string(),
    fullPath: z.string(),
    fileMtime: z.number(),
    firstPrompt: z.string().optional(),
    customTitle: z.string().optional(),
    summary: z.string().optional(),
    tag: z.string().optional(),
    messageCount: z.number(),
    created: z.string(),
    modified: z.string(),
    gitBranch: z.string().optional(),
    projectPath: z.string(),
    isSidechain: z.boolean().optional()
});

const SessionIndexSchema = z.object({
    version: z.number(),
    entries: z.array(SessionIndexEntrySchema)
});

type SessionIndex = z.infer<typeof SessionIndexSchema>;
type SessionIndexEntry = z.infer<typeof SessionIndexEntrySchema>;

const INDEX_FILENAME = "sessions-index.json";
const INDEX_VERSION = 1;

export class SessionIndexManager {
    static getIndexFilePath(projectPath: string): string {
        return path.join(projectPath, INDEX_FILENAME); // Adjust path logic if it's in .claude/sessions
    }

    static readIndex(projectPath: string): SessionIndex | null {
        const filePath = this.getIndexFilePath(projectPath);
        try {
            if (!fs.existsSync(filePath)) return null;
            const content = fs.readFileSync(filePath, "utf-8");
            const index = JSON.parse(content);
            if (index.version !== INDEX_VERSION || !Array.isArray(index.entries)) {
                return null;
            }
            return index;
        } catch {
            return null;
        }
    }

    static writeIndex(projectPath: string, index: SessionIndex): boolean {
        const filePath = this.getIndexFilePath(projectPath);
        const tempPath = `${filePath}.tmp`;
        try {
            if (!fs.existsSync(projectPath)) fs.mkdirSync(projectPath, { recursive: true });
            fs.writeFileSync(tempPath, JSON.stringify(index, null, 2), { encoding: 'utf-8' });
            fs.renameSync(tempPath, filePath);
            return true;
        } catch {
            try { if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath); } catch { }
            return false;
        }
    }

    static updateEntry(projectPath: string, sessionId: string, updates: Partial<SessionIndexEntry>) {
        const index = this.readIndex(projectPath);
        if (!index) return;

        const entry = index.entries.find(e => e.sessionId === sessionId);
        if (!entry) return;

        if (updates.customTitle !== undefined) entry.customTitle = updates.customTitle;
        if (updates.tag !== undefined) entry.tag = updates.tag;

        // Check file mtime if needed
        try {
            const stats = fs.statSync(entry.fullPath);
            entry.fileMtime = stats.mtimeMs;
        } catch { }

        this.writeIndex(projectPath, index);
    }

    static async rebuildIndex(projectPath: string) {
        // Logic from q12 to scan disk and update index
    }
}
