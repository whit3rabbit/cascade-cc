/**
 * File: src/services/telemetry/OtelInit.ts
 * Role: Initializes OpenTelemetry MeterProvider, TracerProvider, and LoggerProvider.
 */

import { diag, DiagConsoleLogger, DiagLogLevel, metrics } from "@opentelemetry/api";
import * as baseResources from "@opentelemetry/resources";
import {
    SEMRESATTRS_SERVICE_NAME,
    SEMRESATTRS_SERVICE_VERSION,
    SEMRESATTRS_TELEMETRY_SDK_NAME,
    SEMRESATTRS_TELEMETRY_SDK_LANGUAGE,
    SEMRESATTRS_TELEMETRY_SDK_VERSION
} from "@opentelemetry/semantic-conventions";
import * as traceSdk from "@opentelemetry/sdk-trace-node";
import * as metricSdk from "@opentelemetry/sdk-metrics";
import * as logSdk from "@opentelemetry/sdk-logs";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { OTLPMetricExporter } from "@opentelemetry/exporter-metrics-otlp-http";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-http";

import { BUILD_INFO } from '../../constants/build.js';
import { getOtlpHttpConfig } from "./OtlpConfig.js";
import { EnvService } from '../config/EnvService.js';

/**
 * Bootstraps OpenTelemetry with standard resource attributes and OTLP exporters.
 */
export function initializeOtel() {
    // 1. Configure Diagnostics
    const debugOtel = EnvService.isTruthy('DEBUG_OTEL');
    diag.setLogger(new DiagConsoleLogger(), debugOtel ? DiagLogLevel.DEBUG : DiagLogLevel.ERROR);

    // 2. Define Shared Resource
    const resource = baseResources.resourceFromAttributes({
        [SEMRESATTRS_SERVICE_NAME]: "claude-code",
        [SEMRESATTRS_SERVICE_VERSION]: BUILD_INFO.VERSION,
        [SEMRESATTRS_TELEMETRY_SDK_NAME]: "opentelemetry",
        [SEMRESATTRS_TELEMETRY_SDK_LANGUAGE]: "nodejs",
        [SEMRESATTRS_TELEMETRY_SDK_VERSION]: "2.2.0", // SDK version
        "os.platform": process.platform,
        "host.arch": process.arch,
        "claude.install_method": EnvService.get("CLAUDE_CODE_INSTALL_METHOD") || "npm"
    });

    // 3. Setup Tracing
    const traceConfig = getOtlpHttpConfig('TRACES');
    if (traceConfig.url) {
        const traceExporter = new OTLPTraceExporter(traceConfig as any);
        const tracerProvider = new traceSdk.NodeTracerProvider({
            resource,
            spanProcessors: [new traceSdk.SimpleSpanProcessor(traceExporter)]
        });
        tracerProvider.register();
    }

    // 4. Setup Metrics
    const metricConfig = getOtlpHttpConfig('METRICS');
    const metricExporter = metricConfig.url ? new OTLPMetricExporter(metricConfig as any) : null;
    const meterProvider = new metricSdk.MeterProvider({
        resource,
        readers: metricExporter ? [
            new metricSdk.PeriodicExportingMetricReader({
                exporter: metricExporter,
                exportIntervalMillis: 60000,
            })
        ] : [],
    });
    metrics.setGlobalMeterProvider(meterProvider);

    // 5. Setup Logs
    const logConfig = getOtlpHttpConfig('LOGS');
    if (logConfig.url) {
        const logExporter = new OTLPLogExporter(logConfig as any);
        const loggerProvider = new logSdk.LoggerProvider({
            resource,
            processors: [new logSdk.SimpleLogRecordProcessor(logExporter)]
        });
        // Note: Global logger provider registration might vary by SDK version
    }

    // Shutdown handler
    const shutdown = async () => {
        try {
            await meterProvider.shutdown();
        } catch (e) {
            diag.error("Error shutting down Otel:", e);
        }
    };

    process.on('SIGINT', async () => {
        await shutdown();
        process.exit(0);
    });

    return { meterProvider, shutdown };
}

export const initializeTelemetry = initializeOtel;
