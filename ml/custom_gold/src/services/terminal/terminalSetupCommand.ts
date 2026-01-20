import { isSupportedTerminal, setupTerminalPrompt } from "./terminalSetup.js";
import {
    installWezTermKeybinding,
    installGhosttyKeybinding,
    installIdeKeybinding
} from "./keybindingInstallers.js";

/**
 * Implementation of the /terminal-setup command.
 * Deobfuscated from $B3 and related in chunk_216.ts.
 */

export const terminalSetupCommand = {
    type: "local-jsx",
    name: "terminal-setup",
    userFacingName() { return "terminal-setup"; },
    description: "Install Shift+Enter key binding for advanced prompts (or Option+Enter in Apple Terminal).",
    isEnabled: () => true,
    isHidden: false,

    async call(print: (msg: string) => void, context: any) {
        const term = process.env.TERM_PROGRAM;

        if (!isSupportedTerminal()) {
            const platform = (process.platform === "darwin" ? "macos" : process.platform === "win32" ? "windows" : "linux");
            let supportedList = "";
            if (platform === "macos") {
                supportedList = "   • macOS: iTerm2, Apple Terminal\n";
            } else if (platform === "windows") {
                supportedList = "   • Windows: Windows Terminal\n";
            }

            const msg = `Terminal setup cannot be run from ${term || "your current terminal"}.

This command configures a convenient Shift+Enter shortcut for multi-line prompts.
Note: You can already use backslash (\\) + return to add newlines.

To set up the shortcut (optional):
1. Exit tmux/screen temporarily
2. Run /terminal-setup directly in one of these terminals:
${supportedList}   • IDE: VSCode, Cursor, Windsurf, Zed
   • Other: Ghostty, WezTerm, Kitty, Alacritty, Warp
3. Return to tmux/screen - settings will persist`;
            print(msg);
            return null;
        }

        // Call shared setup logic
        setupTerminalPrompt();

        // Trigger specific installers based on terminal
        let result = "";
        switch (term) {
            case "iTerm.app":
                // result = await installIterm2Keybinding();
                result = "iTerm2 setup successful (simulated).";
                break;
            case "Apple_Terminal":
                // result = await installAppleTerminalConfig();
                result = "Apple Terminal setup successful (simulated).";
                break;
            case "vscode":
                result = await installIdeKeybinding("VSCode");
                break;
            case "cursor":
                result = await installIdeKeybinding("Cursor");
                break;
            case "windsurf":
                result = await installIdeKeybinding("Windsurf");
                break;
            case "ghostty":
                result = await installGhosttyKeybinding();
                break;
            case "WezTerm":
                result = await installWezTermKeybinding();
                break;
            default:
                result = `Automatic setup for ${term} is not yet implemented. Use manual config.`;
        }

        print(result);
        return null;
    }
};

export default terminalSetupCommand;
