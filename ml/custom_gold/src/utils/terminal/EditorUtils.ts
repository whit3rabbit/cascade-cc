
import { execSync } from 'child_process';
import { existsSync, readFileSync, writeFileSync, unlinkSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { randomUUID } from 'crypto';

// rr logic
export const getEditor = () => {
    if (process.env.VISUAL?.trim()) return process.env.VISUAL.trim();
    if (process.env.EDITOR?.trim()) return process.env.EDITOR.trim();
    if (process.platform === "win32") return "start /wait notepad";

    // Check for common editors
    for (const editor of ["code", "vi", "nano"]) {
        try {
            const cmd = (process.platform as string) === "win32" ? "where" : "which";
            execSync(`${cmd} ${editor}`, { stdio: "ignore" });
            return editor;
        } catch { }
    }
    return null;
};

const waitFlags: Record<string, string> = {
    code: "code -w",
    subl: "subl --wait"
};

const nonInteractiveEditors = ["code", "subl", "atom", "gedit", "notepad++", "notepad"];

function isNonInteractive(editor: string) {
    const bin = editor.split(" ")[0] ?? "";
    return nonInteractiveEditors.some(e => bin.includes(e));
}

// yC0
export function openFileInEditor(filePath: string, inkInstance?: any): string | null {
    const editor = getEditor();
    if (!editor || !existsSync(filePath)) return null;

    const useWaitFlag = waitFlags[editor] || editor;
    const nonInteractive = isNonInteractive(editor);

    try {
        if (inkInstance) {
            inkInstance.pause();
            inkInstance.suspendStdin();
        }

        // Potential TTY handling (from chunk_486:661)
        if (!nonInteractive) {
            process.stdout.write("\x1B[?1049h\x1B[?1004l\x1B[0m\x1B[?25h\x1B[2J\x1B[H");
        }

        execSync(`${useWaitFlag} "${filePath}"`, { stdio: "inherit" });

        return readFileSync(filePath, "utf-8");
    } catch (err) {
        return null;
    } finally {
        if (!nonInteractive) {
            process.stdout.write("\x1B[?1049l\x1B[?1004h\x1B[?25l");
        }
        if (inkInstance) {
            inkInstance.resumeStdin();
            inkInstance.resume();
        }
    }
}

// sI1
export function editStringInEditor(content: string, prefix: string = "claude-prompt", suffix: string = ".md"): string | null {
    const tmpFile = join(tmpdir(), `${prefix}-${randomUUID()}${suffix}`);
    try {
        writeFileSync(tmpFile, content, { encoding: "utf-8", flush: true });
        // We don't have easy access to inkInstance here, but we can try to find it in global state if needed.
        // For now, assuming direct call or passing it.
        const result = openFileInEditor(tmpFile);
        if (result === null) return null;

        // Trim single trailing newline but keep double
        if (result.endsWith('\n') && !result.endsWith('\n\n')) {
            return result.slice(0, -1);
        }
        return result;
    } finally {
        try { if (existsSync(tmpFile)) unlinkSync(tmpFile); } catch { }
    }
}
