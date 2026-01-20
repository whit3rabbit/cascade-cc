import { platform } from "os";
import { spawn } from "child_process";

export async function openInBrowser(url: string): Promise<boolean> {
    const osPlatform = platform();
    let command = "";
    let args: string[] = [];

    switch (osPlatform) {
        case "darwin":
            command = "open";
            args = ["-a", "Google Chrome", url];
            break;
        case "win32":
            command = "rundll32";
            args = ["url,OpenURL", url];
            break;
        case "linux":
            const browsers = ["google-chrome", "google-chrome-stable"];
            for (const browser of browsers) {
                try {
                    // Try spawning
                    // Logic from z82 loop
                    return true;
                } catch { }
            }
            return false;
        default:
            return false;
    }

    try {
        const child = spawn(command, args);
        return new Promise((resolve) => {
            child.on("close", (code) => resolve(code === 0));
            child.on("error", () => resolve(false));
        });
    } catch {
        return false;
    }
}
