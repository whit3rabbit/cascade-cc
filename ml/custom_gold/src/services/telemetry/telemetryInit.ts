import { getTelemetryContext } from "../../utils/shared/envContext.js";
import { TelemetryExporter } from "./telemetryExporter.js";

let exporter: TelemetryExporter | null = null;
let isInitialized = false;

/**
 * Initializes the telemetry system.
 * Deobfuscated from jYB in chunk_188.ts.
 */
export async function initTelemetry() {
    if (isInitialized) return;

    exporter = new TelemetryExporter({
        maxBatchSize: 200,
        batchDelayMs: 5000
    });

    isInitialized = true;

    process.on("beforeExit", async () => {
        await shutdownTelemetry();
    });
}

/**
 * Shuts down the telemetry system and flushes pending events.
 */
export async function shutdownTelemetry() {
    if (!isInitialized || !exporter) return;
    await exporter.shutdown();
    isInitialized = false;
}

/**
 * Logs a telemetry event.
 * Deobfuscated from bp1 in chunk_188.ts.
 */
export async function logTelemetryEvent(eventName: string, metadata: any = {}) {
    if (!isInitialized || !exporter) await initTelemetry();

    const context = await getTelemetryContext();
    const event = {
        eventName,
        timestamp: new Date().toISOString(),
        context,
        metadata
    };

    exporter?.export([{ body: event }], (result) => {
        if (result.error) {
            // Silently ignore telemetry failures in production
            if (process.env.DEBUG_TELEMETRY) console.error("Telemetry export failed", result.error);
        }
    });
}

/**
 * Logs a Growthbook experiment variation assignment.
 * Deobfuscated from _YB in chunk_188.ts.
 */
export async function logGrowthbookExperiment(experimentId: string, variationId: number, attributes: any) {
    await logTelemetryEvent("growthbook_experiment", {
        experimentId,
        variationId,
        attributes
    });
}
