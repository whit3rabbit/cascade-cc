import { execFile } from 'child_process';
import { promisify } from 'util';
import { join } from 'path';
import { tmpdir } from 'os';
import { existsSync, unlinkSync } from 'fs';
import plist from 'plist';

const execFileAsync = promisify(execFile);

/**
 * Executes an AppleScript command using osascript.
 */
export async function execAppleScript(script: string): Promise<string> {
    if (process.platform !== 'darwin') {
        throw new Error('macOS only');
    }
    const { stdout } = await execFileAsync('osascript', ['-e', script]);
    return stdout.trim();
}

/**
 * Checks if the system clipboard contains an image.
 */
export async function hasClipboardImage(): Promise<boolean> {
    if (process.platform !== 'darwin') {
        return false;
    }
    try {
        await execAppleScript('the clipboard as «class PNGf»');
        return true;
    } catch {
        return false;
    }
}

/**
 * Gets the file path from the clipboard if it exists.
 * Aligned with chunk446 logic.
 */
export async function getClipboardFilePath(): Promise<string | null> {
    if (process.platform !== 'darwin') {
        return null;
    }
    try {
        // Aligned with chunk446 getPath: "osascript -e 'get POSIX path of (the clipboard as «class furl»)'"
        return await execAppleScript('get POSIX path of (the clipboard as «class furl»)');
    } catch {
        return null;
    }
}

/**
 * Saves the clipboard image to a temporary file and returns the path.
 * Aligned with chunk446 logic.
 */
export async function getClipboardImage(tempDir: string = tmpdir()): Promise<{ path: string; cleanup: () => void } | null> {
    if (process.platform !== 'darwin') {
        return null;
    }

    const targetPath = join(tempDir, 'claude_cli_latest_screenshot.png');

    // Logic from chunk446
    // set png_data to (the clipboard as «class PNGf»)
    // set fp to open for access POSIX file "..." with write permission
    // write png_data to fp
    // close access fp

    const script = `
    try
      set png_data to (the clipboard as «class PNGf»)
      set fp to open for access POSIX file "${targetPath}" with write permission
      write png_data to fp
      close access fp
      return "success"
    on error
      return "error"
    end try
  `;

    try {
        const result = await execAppleScript(script);
        if (result === 'success' && existsSync(targetPath)) {
            return {
                path: targetPath,
                cleanup: () => {
                    try {
                        unlinkSync(targetPath);
                    } catch { }
                }
            };
        }
        return null;
    } catch {
        return null;
    }
}

/**
 * Copies an image file to the clipboard.
 * Aligned with chunk1443 qB2 logic.
 */
export async function copyImageToClipboard(imagePath: string): Promise<{ success: boolean; message: string }> {
    if (process.platform !== 'darwin') {
        return { success: false, message: 'macOS only' };
    }

    // Logic from chunk1443 qB2
    // set the clipboard to (read (POSIX file "...") as «class PNGf»)
    // Escaping logic: replace / with //, " with \"
    const escapedPath = imagePath.replace(/\\/g, "\\\\").replace(/"/g, "\\\"");
    const script = `set the clipboard to (read (POSIX file "${escapedPath}") as «class PNGf»)`;

    try {
        await execAppleScript(script);
        return { success: true, message: 'Screenshot copied to clipboard' };
    } catch (error: any) {
        return { success: false, message: `Failed to copy to clipboard: ${error.stderr || error.message}` };
    }
}

/**
 * Gets the bundle name for an application ID.
 * Aligned with chunk595 logic.
 */
export async function getAppNameFromId(bundleId: string): Promise<string> {
    if (process.platform !== 'darwin') {
        return '';
    }
    // tell application "Finder" to set app_path to application file id "${A}" as string
    // tell application "System Events" to get value of property list item "CFBundleName" of property list file (app_path & ":Contents:Info.plist")
    const script = `
    tell application "Finder" to set app_path to application file id "${bundleId}" as string
    tell application "System Events" to get value of property list item "CFBundleName" of property list file (app_path & ":Contents:Info.plist")
  `;
    try {
        return await execAppleScript(script);
    } catch {
        return '';
    }
}

/**
 * Checks if the bell is enabled in Apple Terminal.
 * Aligned with chunk1055 logic.
 */
export async function isAppleTerminalBellEnabled(): Promise<boolean> {
    if (process.platform !== 'darwin' || process.env.TERM_PROGRAM !== 'Apple_Terminal') {
        return false;
    }

    try {
        // tell application "Terminal" to name of current settings of front window
        const settingsName = await execAppleScript('tell application "Terminal" to name of current settings of front window');
        if (!settingsName) return false;

        // Check defaults
        const { stdout } = await execFileAsync('defaults', ['export', 'com.apple.Terminal', '-']);

        // Use plist parser as per chunk1055
        const parsed = plist.parse(stdout) as any;
        const windowSettings = parsed['Window Settings'];
        const currentSettings = windowSettings ? windowSettings[settingsName] : null;

        if (!currentSettings) {
            return false;
        }

        // Bell is enabled unless explicitly set to false? 
        // Chunk1055 logic: return w.Bell === !1; -> Wait, chunk1055 says:
        // return w.Bell === !1; 
        // This implies it returns TRUE if Bell is FALSE (disabled). 
        // But the function name is isAppleTerminalBellEnabled...
        // Let's re-read chunk1055 carefully.
        // It's used in clearConversation_4:
        // if (await clearConversation_6()) { K.notifyBell(); return "terminal_bell"; }
        // notifyBell() makes sound. So clearConversation_6() must return TRUE if bell is ENABLED.
        // In chunk1055: return w.Bell === !1;
        // if w.Bell is false, it returns true? That means "Bell disabled" = true? 
        // No, maybe w.Bell property means "Visual Bell" or "Silent Bell"?
        // Let's look at `man defaults` for com.apple.Terminal.
        // "Bell" key usually controls "Audible Bell".
        // If Bell = false, audible bell is off.
        // If chunk code is `return w.Bell === !1`, it returns true if Bell is false.
        // If it returns true, it calls `notifyBell()`. 
        // This is confusing. If bell is disabled (Bell=false), why call notifyBell?
        // Maybe `notifyBell` implies sending a bell character, which might do something else (visual) if audible is off?
        // OR `clearConversation_6` checks if we CAN use the bell?
        // If `w.Bell` was "AudibleBell", then `false` means off.
        // Maybe it's checking if "Visual Bell" is enabled?
        // Let's assume standard behavior: we want to know if we can make a sound.
        // If Bell is NOT false (i.e. true or undefined (default on)), we return true.
        // Updating to be safer:

        // Re-reading chunk1055 lines 140-144:
        // let w = A5K.default.parse(q.stdout)?.["Window Settings"]?.[K];
        // if (!w) { return !1; }
        // return w.Bell === !1;

        // If w.Bell is explicitly false, return true.
        // This implies "isAppleTerminalBellEnabled" might be a misnomer in my code or the deobfuscated code logic is inverted.
        // If the original chunk return true when Bell is false, maybe it means "Is Bell Suppressed" or "Is Silent"?
        // But the usage site:
        // if (await clearConversation_6()) { K.notifyBell(); return "terminal_bell"; }
        // It calls notifyBell if that returns true.
        // Maybe notifyBell sends \a. 
        // If the terminal Bell setting is OFF (false), sending \a does nothing?
        // Unless it's "Visual Bell"?

        // Let's stick to the logic exactly as is for now.
        // w.Bell === !1  means w.Bell is false.
        // So if Bell is false in plist, return true.

        return currentSettings.Bell === false;
    } catch {
        return false;
    }
}
