import { exec } from "child_process";
import { promisify } from "util";
import * as plist from "plist";
import { log } from "../logger/loggerService.js";
import { getSettings } from "../terminal/settings.js";

const execAsync = promisify(exec);
const logger = log("NotificationService");

export async function sendNotification(message: string, title?: string) {
    const formattedTitle = title ? `${title}:\n${message}` : message;
    const settings = getSettings("userSettings");

    // Check configured channel from settings, default to auto
    const channel = settings.preferredNotifChannel || "auto";
    let methodUsed = "none";

    try {
        await handleNotification(channel, message, title || "Claude Code", formattedTitle, (method) => {
            methodUsed = method;
        });
    } catch (error) {
        logger.error(`Failed to send notification: ${error}`);
    }

    // Telemetry could go here if implemented
    logger.debug(`Notification sent via ${methodUsed} (configured: ${channel})`);
}

async function handleNotification(
    channel: string,
    message: string,
    title: string,
    formattedTitle: string,
    setMethod: (m: string) => void
) {
    const term = process.env.TERM_PROGRAM;

    switch (channel) {
        case "auto":
            if (term === "Apple_Terminal") {
                if (await isTerminalBellEnabled()) {
                    ringBell();
                    setMethod("terminal_bell");
                } else {
                    setMethod("no_method_available");
                }
            } else if (term === "iTerm.app") {
                notifyIterm2(formattedTitle);
                setMethod("iterm2");
            } else if (term === "kitty") {
                notifyKitty(message, title);
                setMethod("kitty");
            } else if (term === "ghostty") {
                notifyGhostty(message, title);
                setMethod("ghostty");
            } else {
                setMethod("no_method_available");
            }
            break;
        case "iterm2":
            notifyIterm2(formattedTitle);
            setMethod("iterm2");
            break;
        case "terminal_bell":
            ringBell();
            setMethod("terminal_bell");
            break;
        case "iterm2_with_bell":
            notifyIterm2(formattedTitle);
            ringBell();
            setMethod("iterm2_with_bell");
            break;
        case "kitty":
            notifyKitty(message, title);
            setMethod("kitty");
            break;
        case "notifications_disabled":
            setMethod("disabled");
            break;
        default:
            setMethod("unknown_channel");
    }
}

function notifyIterm2(message: string) {
    try {
        process.stdout.write(`\x1B]9;\n\n${message}\x07`);
    } catch { }
}

function notifyKitty(message: string, title: string) {
    try {
        const id = Math.floor(Math.random() * 10000);
        process.stdout.write(`\x1B]99;i=${id}:d=0:p=title;${title}\x1B\\`);
        process.stdout.write(`\x1B]99;i=${id}:p=body;${message}\x1B\\`);
        process.stdout.write(`\x1B]99;i=${id}:d=1:a=focus;\x1B\\`);
    } catch { }
}

function notifyGhostty(message: string, title: string) {
    try {
        process.stdout.write(`\x1B]777;notify;${title};${message}\x07`);
    } catch { }
}

function ringBell() {
    process.stdout.write("\x07");
}

async function isTerminalBellEnabled(): Promise<boolean> {
    try {
        if (process.env.TERM_PROGRAM !== "Apple_Terminal") return false;

        // Get the name of the current settings profile
        const script = 'tell application "Terminal" to name of current settings of front window';
        const { stdout: profileName } = await execAsync(`osascript -e '${script}'`, { timeout: 2000 });
        const trimmedProfile = profileName.trim();

        if (!trimmedProfile) return false;

        // Export settings to defaults format (which is XML plist)
        // Command: defaults export com.apple.Terminal -
        const { stdout: plistContent } = await execAsync("defaults export com.apple.Terminal -", { timeout: 2000 });

        // Parse Plist
        const settings = plist.parse(plistContent) as any;
        const windowSettings = settings?.["Window Settings"]?.[trimmedProfile];

        if (!windowSettings) return false;

        // Check Bell property
        return windowSettings.Bell !== false;
    } catch (error) {
        logger.debug(`Error checking terminal bell status: ${error}`);
        return false;
    }
}
