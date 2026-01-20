/**
 * Simple terminal color utility providing ANSI escape sequences.
 */
export class Colours {
    static enabled = false;
    static reset = "";
    static bright = "";
    static dim = "";
    static red = "";
    static green = "";
    static yellow = "";
    static blue = "";
    static magenta = "";
    static cyan = "";
    static white = "";
    static grey = "";

    /**
     * Checks if colors should be enabled based on TTY status and color depth.
     */
    static isEnabled(stream: NodeJS.WriteStream): boolean {
        return !!(
            stream.isTTY &&
            (typeof stream.getColorDepth === "function" ? stream.getColorDepth() > 2 : true)
        );
    }

    /**
     * Refreshes the color codes based on current terminal capabilities.
     */
    static refresh() {
        this.enabled = this.isEnabled(process.stderr);

        if (!this.enabled) {
            this.reset = "";
            this.bright = "";
            this.dim = "";
            this.red = "";
            this.green = "";
            this.yellow = "";
            this.blue = "";
            this.magenta = "";
            this.cyan = "";
            this.white = "";
            this.grey = "";
        } else {
            this.reset = "\x1b[0m";
            this.bright = "\x1b[1m";
            this.dim = "\x1b[2m";
            this.red = "\x1b[31m";
            this.green = "\x1b[32m";
            this.yellow = "\x1b[33m";
            this.blue = "\x1b[34m";
            this.magenta = "\x1b[35m";
            this.cyan = "\x1b[36m";
            this.white = "\x1b[37m";
            this.grey = "\x1b[90m";
        }
    }
}

// Initial refresh
Colours.refresh();
