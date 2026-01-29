/**
 * File: src/utils/shared/screenshotUtils.ts
 * Role: Platform-specific logic for clipboard image and screenshot handling.
 */

import { join } from "node:path";
import { readFile, unlink } from "node:fs/promises";
import { executeBashCommand, spawnBashCommand } from "./bashUtils.js";

/**
 * Gets the error message when no image is found in the clipboard.
 */
export function getClipboardImageErrorMessage(): string {
    const platform = process.platform;
    const messages: Record<string, string> = {
        darwin: "No image found in clipboard. Use Cmd + Ctrl + Shift + 4 to copy a screenshot to clipboard.",
        win32: "No image found in clipboard. Use Print Screen to copy a screenshot to clipboard.",
        linux: "No image found in clipboard. Use appropriate screenshot tool to copy a screenshot to clipboard."
    };
    return messages[platform] || messages.linux!;
}

/**
 * Gets the temporary directory path for screenshots.
 */
export function getScreenshotTempDir(): string {
    const platform = process.platform;
    const defaultTempDir = platform === "win32" ? process.env.TEMP || "C:\\Temp" : "/tmp";
    return process.env.CLAUDE_CODE_TMPDIR || defaultTempDir;
}

interface ScreenshotCommands {
    checkImage: string;
    saveImage: string;
    getPath: string;
    deleteFile: string;
}

/**
 * Constructs the screenshot file paths and commands based on the platform.
 */
export function getScreenshotPaths(tempDir: string): { commands: ScreenshotCommands; screenshotPath: string } {
    const screenshotFilename = "claude_cli_latest_screenshot.png";
    const platform = process.platform;
    const screenshotPath = join(tempDir, screenshotFilename);

    const allCommands: Record<string, ScreenshotCommands> = {
        darwin: {
            checkImage: "osascript -e 'the clipboard as «class PNGf»'",
            saveImage: `osascript -e 'set png_data to (the clipboard as «class PNGf»)' -e 'set fp to open for access POSIX file "${screenshotPath}" with write permission' -e 'write png_data to fp' -e 'close access fp'`,
            getPath: "osascript -e 'get POSIX path of (the clipboard as «class furl»)'",
            deleteFile: `rm -f "${screenshotPath}"`
        },
        linux: {
            checkImage: 'xclip -selection clipboard -t TARGETS -o 2>/dev/null | grep -E "image/(png|jpeg|jpg|gif|webp)" || wl-paste -l 2>/dev/null | grep -E "image/(png|jpeg|jpg|gif|webp)"',
            saveImage: `xclip -selection clipboard -t image/png -o > "${screenshotPath}" 2>/dev/null || wl-paste --type image/png > "${screenshotPath}"`,
            getPath: "xclip -selection clipboard -t text/plain -o 2>/dev/null || wl-paste 2>/dev/null",
            deleteFile: `rm -f "${screenshotPath}"`
        },
        win32: {
            checkImage: 'powershell -NoProfile -Command "(Get-Clipboard -Format Image) -ne $null"',
            saveImage: `powershell -NoProfile -Command "$img = Get-Clipboard -Format Image; if ($img) { $img.Save('${screenshotPath.replace(/\\/g, "\\\\")}', [System.Drawing.Imaging.ImageFormat]::Png) }"`,
            getPath: 'powershell -NoProfile -Command "Get-Clipboard"',
            deleteFile: `del /f "${screenshotPath}"`
        }
    };

    return {
        commands: allCommands[platform] || allCommands.linux!,
        screenshotPath,
    };
}

/**
 * Checks if the clipboard contains a PNG image (macOS only).
 */
export async function hasPngInClipboard(): Promise<boolean> {
    if (process.platform !== "darwin") return false;
    try {
        const result = await executeBashCommand("osascript -e 'the clipboard as «class PNGf»'");
        return result.exitCode === 0;
    } catch {
        return false;
    }
}

/**
 * Determines the media type of an image buffer.
 */
export function getMediaTypeFromBuffer(buffer: Buffer): string {
    if (buffer.length < 4) return "image/png";

    if (buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4E && buffer[3] === 0x47) {
        return "image/png";
    }
    if (buffer[0] === 0xFF && buffer[1] === 0xD8 && buffer[2] === 0xFF) {
        return "image/jpeg";
    }
    if (buffer[0] === 0x47 && buffer[1] === 0x49 && buffer[2] === 0x46) {
        return "image/gif";
    }
    if (buffer[0] === 0x52 && buffer[1] === 0x49 && buffer[2] === 0x46 && buffer[3] === 0x46) {
        if (buffer.length >= 12 && buffer[8] === 0x57 && buffer[9] === 0x45 && buffer[10] === 0x42 && buffer[11] === 0x50) {
            return "image/webp";
        }
    }
    return "image/png";
}

/**
 * Retrieves the screenshot from the clipboard and saves it to a file.
 */
export async function getScreenshotFromClipboard(): Promise<{ base64: string; mediaType: string } | null> {
    const tempDir = getScreenshotTempDir();
    const { commands, screenshotPath } = getScreenshotPaths(tempDir);

    try {
        // Check for image
        const checkRes = await executeBashCommand(commands.checkImage);
        if (checkRes.exitCode !== 0) return null;

        // Save image
        await executeBashCommand(commands.saveImage);

        const buffer = await readFile(screenshotPath);
        const base64 = buffer.toString("base64");
        const mediaType = getMediaTypeFromBuffer(buffer);

        // Cleanup
        await unlink(screenshotPath).catch(() => { });

        return { base64, mediaType };
    } catch (error) {
        console.error("[Screenshot] Error getting screenshot:", error);
        return null;
    }
}
