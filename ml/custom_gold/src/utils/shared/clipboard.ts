import { execSync } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { resizeImage } from "./imageOptimizer.js";

/**
 * Clipboard image capturing utilities.
 * Deobfuscated from QA1, TDB, and related in chunk_215.ts.
 */

const SCREENSHOT_FILENAME = "claude_cli_latest_screenshot.png";

function getScreenshotPath(): string {
    const platform = os.platform();
    if (platform === "win32") {
        const temp = process.env.TEMP || "C:\\Temp";
        return path.join(temp, SCREENSHOT_FILENAME);
    }
    return path.join("/tmp", SCREENSHOT_FILENAME);
}

function getClipboardCommands() {
    const platform = os.platform();
    const screenshotPath = getScreenshotPath();

    const commands = {
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

    return (commands as any)[platform] || commands.linux;
}

/**
 * Detects if an image is in the clipboard and captures it.
 * Deobfuscated from QA1 in chunk_215.ts.
 */
export async function captureClipboardImage() {
    const cmds = getClipboardCommands();
    const screenshotPath = getScreenshotPath();

    try {
        execSync(cmds.checkImage, { stdio: "ignore" });
        execSync(cmds.saveImage, { stdio: "ignore" });

        if (!fs.existsSync(screenshotPath)) return null;

        const buffer = fs.readFileSync(screenshotPath);
        const resized = await resizeImage(buffer, buffer.length, "png");

        // Clean up
        try {
            execSync(cmds.deleteFile, { stdio: "ignore" });
        } catch (e) { }

        return {
            base64: resized.buffer.toString("base64"),
            mediaType: `image/${resized.mediaType}`,
            dimensions: resized.dimensions
        };
    } catch (e) {
        return null;
    }
}

/**
 * Gets the raw path from the clipboard if it exists.
 */
export function getClipboardPath(): string | null {
    const cmds = getClipboardCommands();
    try {
        const output = execSync(cmds.getPath, { encoding: "utf-8" }).trim();
        return output || null;
    } catch (e) {
        return null;
    }
}

/**
 * Detects the media type of a buffer by inspecting magic numbers.
 */
export function detectImageMediaType(buffer: Buffer): string {
    if (buffer.length < 4) return "image/png";
    if (buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4E && buffer[3] === 0x47) return "image/png";
    if (buffer[0] === 0xFF && buffer[1] === 0xD8 && buffer[2] === 0xFF) return "image/jpeg";
    if (buffer[0] === 0x47 && buffer[1] === 0x49 && buffer[2] === 0x46) return "image/gif";
    if (buffer[0] === 0x52 && buffer[1] === 0x49 && buffer[2] === 0x46 && buffer[3] === 0x46) {
        if (buffer.length >= 12 && buffer[8] === 0x57 && buffer[9] === 0x45 && buffer[10] === 0x42 && buffer[11] === 0x50) return "image/webp";
    }
    return "image/png";
}
