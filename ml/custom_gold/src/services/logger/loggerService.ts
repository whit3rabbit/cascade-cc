import { EventEmitter } from "node:events";
import * as process from "node:process";
import * as util from "node:util";
import * as fs from "node:fs";
import { join } from "node:path";
import { Colours } from "../../utils/shared/terminalColors.js";
import { getSessionId } from "../session/globalState.js";
import { getConfigDir } from "../../utils/shared/pathUtils.js";
import { filterLogLine, parseFilterString } from "../logging/filterUtils.js";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";

export enum LogSeverity {
    DEFAULT = "DEFAULT",
    DEBUG = "DEBUG",
    INFO = "INFO",
    WARNING = "WARNING",
    ERROR = "ERROR"
}

export type LogMetadata = Record<string, any> & { severity?: LogSeverity };

export class AdhocDebugLogger extends EventEmitter {
    constructor(public namespace: string, private upstream?: (meta: LogMetadata, ...args: any[]) => void) {
        super();
    }

    get func() {
        return Object.assign(this.invoke.bind(this), {
            instance: this,
            on: (event: string, listener: (...args: any[]) => void) => this.on(event, listener),
            debug: (...args: any[]) => this.invokeSeverity(LogSeverity.DEBUG, ...args),
            info: (...args: any[]) => this.invokeSeverity(LogSeverity.INFO, ...args),
            warn: (...args: any[]) => this.invokeSeverity(LogSeverity.WARNING, ...args),
            error: (...args: any[]) => this.invokeSeverity(LogSeverity.ERROR, ...args),
            sublog: (subNamespace: string) => log(subNamespace, this.func)
        });
    }

    private invoke(meta: LogMetadata, ...args: any[]) {
        if (this.upstream) {
            this.upstream(meta, ...args);
        }
        this.emit("log", meta, args);
    }

    private invokeSeverity(severity: LogSeverity, ...args: any[]) {
        this.invoke({ severity }, ...args);
    }
}

export const placeholder = new AdhocDebugLogger("", () => { }).func;

export abstract class DebugLogBackendBase {
    protected cached = new Map<string, (...args: any[]) => void>();
    protected filters: string[] = [];
    private filtersSet = false;

    constructor() {
        let enables = process.env[LOGGER_CONFIG.nodeEnables] ?? "*";
        if (enables === "all") enables = "*";
        this.filters = enables.split(",");
    }

    log(namespace: string, meta: LogMetadata, ...args: any[]) {
        try {
            if (!this.filtersSet) {
                this.setFilters();
                this.filtersSet = true;
            }
            let logger = this.cached.get(namespace);
            if (!logger) {
                logger = this.makeLogger(namespace);
                this.cached.set(namespace, logger);
            }
            logger(meta, ...args);
        } catch (err) {
            console.error(err);
        }
    }

    abstract makeLogger(namespace: string): (meta: LogMetadata, ...args: any[]) => void;
    abstract setFilters(): void;
}

class ConsoleDebugBackend extends DebugLogBackendBase {
    private enabledRegexp = /.*/;

    isEnabled(namespace: string): boolean {
        return this.enabledRegexp.test(namespace);
    }

    makeLogger(namespace: string) {
        if (!this.isEnabled(namespace)) return () => { };

        return (meta: LogMetadata, ...args: any[]) => {
            const nsColor = `${Colours.green}${namespace}${Colours.reset}`;
            const pidColor = `${Colours.yellow}${process.pid}${Colours.reset}`;
            let sevColor: string;

            switch (meta.severity) {
                case LogSeverity.ERROR:
                    sevColor = `${Colours.red}${meta.severity}${Colours.reset}`;
                    break;
                case LogSeverity.INFO:
                    sevColor = `${Colours.magenta}${meta.severity}${Colours.reset}`;
                    break;
                case LogSeverity.WARNING:
                    sevColor = `${Colours.yellow}${meta.severity}${Colours.reset}`;
                    break;
                default:
                    sevColor = meta.severity ?? LogSeverity.DEFAULT;
                    break;
            }

            const formatted = util.formatWithOptions({ colors: Colours.enabled }, ...args);
            const metaClone = { ...meta };
            delete metaClone.severity;
            const metaStr = Object.keys(metaClone).length ? JSON.stringify(metaClone) : "";
            const metaColor = metaStr ? `${Colours.grey}${metaStr}${Colours.reset}` : "";

            console.error("%s [%s|%s] %s%s", pidColor, nsColor, sevColor, formatted, metaStr ? ` ${metaColor}` : "");
        };
    }

    setFilters() {
        const pattern = this.filters
            .join(",")
            .replace(/[|\\{}()[\]^$+?.]/g, "\\$&")
            .replace(/\*/g, ".*")
            .replace(/,/g, "$|^");
        this.enabledRegexp = new RegExp(`^${pattern}$`, "i");
    }
}

export function getNodeBackend() {
    return new ConsoleDebugBackend();
}

/**
 * File-based debug logging backend (logic from chunk_3.ts)
 */
class FileDebugBackend extends DebugLogBackendBase {
    private filter: any = null;
    private logFile: string | null = null;
    private bufferedWriter: any = null;

    constructor() {
        super();
        const debugFlag = process.argv.find(a => a.startsWith("--debug="));
        const filterStr = debugFlag ? debugFlag.substring(8) : (process.env.DEBUG || "");
        this.filter = parseFilterString(filterStr);

        // Initialize log file
        const sessionId = getSessionId();
        const logDir = join(getConfigDir(), "debug");
        this.logFile = process.env.CLAUDE_CODE_DEBUG_LOGS_DIR ?? join(logDir, `${sessionId}.txt`);

        if (typeof process !== 'undefined') {
            try {
                if (!fs.existsSync(logDir)) fs.mkdirSync(logDir, { recursive: true });
                // Maintain 'latest' symlink
                const latestLink = join(logDir, "latest");
                if (fs.existsSync(latestLink)) fs.unlinkSync(latestLink);
                fs.symlinkSync(this.logFile, latestLink);
            } catch { }
        }
    }

    makeLogger(namespace: string) {
        return (meta: LogMetadata, ...args: any[]) => {
            if (!filterLogLine(namespace, this.filter)) return;

            const timestamp = new Date().toISOString();
            const severity = meta.severity ?? LogSeverity.DEBUG;
            const formatted = util.format(...args);
            const line = `${timestamp} [${severity}] ${namespace}: ${formatted}\n`;

            if (process.argv.includes("--debug-to-stderr")) {
                process.stderr.write(line);
                return;
            }

            try {
                if (this.logFile) fs.appendFileSync(this.logFile, line);
            } catch { }
        };
    }

    setFilters() {
        // Handled in constructor for this backend
    }
}

export function getFileBackend() {
    return new FileDebugBackend();
}

/**
 * Interface with the 'debug' library if needed.
 */
class DebugPkgBackend extends DebugLogBackendBase {
    constructor(private debugPkg: any) {
        super();
    }

    makeLogger(namespace: string) {
        const d = this.debugPkg(namespace);
        return (meta: LogMetadata, ...args: any[]) => {
            d(args[0], ...args.slice(1));
        };
    }

    setFilters() {
        const current = process.env.NODE_DEBUG ?? "";
        process.env.NODE_DEBUG = `${current}${current ? "," : ""}${this.filters.join(",")}`;
    }
}

export function getDebugBackend(debugPkg: any) {
    return new DebugPkgBackend(debugPkg);
}

class StructuredLogBackend extends DebugLogBackendBase {
    constructor(private upstream: DebugLogBackendBase = getNodeBackend()) {
        super();
    }

    makeLogger(namespace: string) {
        const upstreamLogger = this.upstream.makeLogger(namespace);
        return (meta: LogMetadata, ...args: any[]) => {
            const severity = meta.severity ?? LogSeverity.INFO;
            const structured = {
                severity,
                message: util.format(...args),
                ...meta
            };
            upstreamLogger(meta, JSON.stringify(structured));
        };
    }

    setFilters() {
        this.upstream.setFilters();
    }
}

export function getStructuredBackend(upstream?: DebugLogBackendBase) {
    return new StructuredLogBackend(upstream);
}

export const LOGGER_CONFIG = {
    nodeEnables: "GOOGLE_SDK_NODE_LOGGING"
};

const loggerInstances = new Map<string, AdhocDebugLogger>();
let activeBackend: DebugLogBackendBase | undefined;

export function setBackend(backend: DebugLogBackendBase) {
    activeBackend = backend;
    loggerInstances.clear();
}

export function log(namespace: string, parent?: any): any {
    if (!process.env[LOGGER_CONFIG.nodeEnables]) return placeholder;
    if (!namespace) return placeholder;

    if (parent && parent.instance && parent.instance.namespace) {
        namespace = `${parent.instance.namespace}:${namespace}`;
    }

    let instance = loggerInstances.get(namespace);
    if (instance) return instance.func;

    if (activeBackend === null) return placeholder;
    if (activeBackend === undefined) {
        // Prioritize GOOGLE_SDK_NODE_LOGGING for structured logging
        if (process.env[LOGGER_CONFIG.nodeEnables]) {
            activeBackend = getNodeBackend();
        } else {
            // Default to file-based debug logging for the CLI
            activeBackend = getFileBackend();
        }
    }

    const downstream = (() => {
        let currentBackend: DebugLogBackendBase | undefined;
        return (meta: LogMetadata, ...args: any[]) => {
            if (currentBackend !== activeBackend) {
                if (activeBackend === null) return;
                if (activeBackend === undefined) activeBackend = getNodeBackend();
                currentBackend = activeBackend;
            }
            activeBackend?.log(namespace, meta, ...args);
        };
    })();

    instance = new AdhocDebugLogger(namespace, downstream);
    loggerInstances.set(namespace, instance);
    return instance.func;
}

export function logError(namespace: string, error: any, ...args: any[]) {
    log(namespace).error(error, ...args);
}

/**
 * High-level exception logger that also triggers telemetry (logic from chunk_4.ts t)
 */
export function logException(error: Error, metadata: any = {}) {
    const message = error.stack || error.message;
    log("error").error(`${error.name}: ${message}`, metadata);

    // Trigger telemetry event
    logTelemetryEvent("error_reported", {
        name: error.name,
        message,
        ...metadata,
        timestamp: new Date().toISOString()
    }).catch(() => { });
}
