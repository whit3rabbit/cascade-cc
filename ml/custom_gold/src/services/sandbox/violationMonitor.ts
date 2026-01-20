import { spawn, ChildProcess } from "node:child_process";
import { EventEmitter } from "node:events";

/**
 * Interface for a sandbox violation.
 * Deobfuscated from the ViolationManager (OLA) logic.
 */
export interface SandboxViolation {
    line: string;
    command?: string;
    encodedCommand?: string;
    timestamp: Date;
}

/**
 * Manages and collects sandbox violations.
 * Deobfuscated from OLA in chunk_221.ts.
 */
export class ViolationManager extends EventEmitter {
    private violations: SandboxViolation[] = [];
    private totalCount = 0;
    private readonly maxSize = 100;

    addViolation(violation: SandboxViolation) {
        this.violations.push(violation);
        this.totalCount++;
        if (this.violations.length > this.maxSize) {
            this.violations = this.violations.slice(-this.maxSize);
        }
        this.emit("violation", violation);
        this.emit("update", this.violations);
    }

    getViolations(limit?: number): SandboxViolation[] {
        if (limit === undefined) return [...this.violations];
        return this.violations.slice(-limit);
    }

    getCount(): number {
        return this.violations.length;
    }

    getTotalCount(): number {
        return this.totalCount;
    }

    clear() {
        this.violations = [];
        this.emit("update", this.violations);
    }

    /**
     * Helper to filter violations by a specific command.
     */
    getViolationsForCommand(command: string): SandboxViolation[] {
        const encoded = Buffer.from(command.slice(0, 100)).toString("base64");
        return this.violations.filter(v => v.encodedCommand === encoded);
    }
}

/**
 * Monitor macOS sandbox violations via 'log stream'.
 * Deobfuscated from UEB in chunk_221.ts.
 */
export function monitorMacosViolations(
    onViolation: (v: SandboxViolation) => void,
    options: {
        ignoreViolations?: Record<string, string[]>,
        logTag?: string
    } = {}
): () => void {
    const { ignoreViolations, logTag = "CLAUDE_SBX" } = options;

    // Start log stream looking for sandbox messages
    const predicate = logTag ? `(eventMessage ENDSWITH "${logTag}")` : '(process == "sandbox-exec")';
    const logProcess = spawn("log", ["stream", "--predicate", predicate, "--style", "compact"]);

    logProcess.stdout?.on("data", (data: Buffer) => {
        const lines = data.toString().split("\n");
        const violationLine = lines.find(l => l.includes("Sandbox:") && l.includes("deny"));
        const commandLine = lines.find(l => l.startsWith("CMD64_"));

        if (!violationLine) return;

        // Extract the denial details
        const denialMatch = violationLine.match(/Sandbox:\s+(.+)$/);
        if (!denialMatch?.[1]) return;
        const denialMsg = denialMatch[1];

        let command: string | undefined;
        let encodedCommand: string | undefined;

        if (commandLine) {
            const cmdMatch = commandLine.match(/CMD64_(.+?)_END/);
            if (cmdMatch?.[1]) {
                encodedCommand = cmdMatch[1];
                try {
                    command = Buffer.from(encodedCommand, "base64").toString("utf8");
                } catch { }
            }
        }

        // Skip ignored violations
        if (ignoreViolations) {
            const globalIgnores = ignoreViolations["*"] || [];
            if (globalIgnores.some(pattern => denialMsg.includes(pattern))) return;

            if (command) {
                for (const [cmdPattern, patterns] of Object.entries(ignoreViolations)) {
                    if (cmdPattern !== "*" && command.includes(cmdPattern)) {
                        if (patterns.some(p => denialMsg.includes(p))) return;
                    }
                }
            }
        }

        // Common system noise to ignore
        if (denialMsg.includes("mDNSResponder") ||
            denialMsg.includes("mach-lookup com.apple.diagnosticd") ||
            denialMsg.includes("mach-lookup com.apple.analyticsd")) {
            return;
        }

        onViolation({
            line: denialMsg,
            command,
            encodedCommand,
            timestamp: new Date()
        });
    });

    logProcess.stderr?.on("data", (data: Buffer) => {
        // Standard debug logging would go here
    });

    return () => {
        logProcess.kill("SIGTERM");
    };
}
