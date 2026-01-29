/**
 * File: src/utils/nativeHostRunner.ts
 * Role: Implementation of the Chrome Native Messaging protocol for browser extension integration.
 */

import { appendFileSync } from "node:fs";

let logFilePath: string | undefined;

/**
 * Logs an error to a file and console, specifically for the Native Host context.
 */
export function logNativeHostError(message: string, ...args: any[]): void {
    if (logFilePath) {
        const timestamp = new Date().toISOString();
        const logEntry = `[${timestamp}] [NativeHost] ${message} ${args.join(", ")}\n`;
        try {
            appendFileSync(logFilePath, logEntry);
        } catch {
            // Ignore silent failures
        }
    }
    console.error(`[NativeHost] ${message}`, ...args);
}

/**
 * Sends a message to the browser extension using the Native Messaging protocol.
 * Protocol: 4-byte length prefix (Little Endian) followed by JSON data.
 */
export function sendMessageToExtension(message: any): void {
    const payload = JSON.stringify(message);
    const buffer = Buffer.from(payload, "utf-8");
    const lengthBuffer = Buffer.alloc(4);
    lengthBuffer.writeUInt32LE(buffer.length, 0);

    process.stdout.write(lengthBuffer);
    process.stdout.write(buffer);
}

/**
 * Interface for reading messages from the extension.
 */
export interface NativeMessageReader {
    read(): Promise<any | null>;
}

/**
 * Interface for handling messages from the extension.
 */
export interface NativeMessageHandler {
    start(): Promise<void>;
    stop(): Promise<void>;
    handleMessage(message: any): Promise<void>;
}

/**
 * Main execution loop for the Native Host.
 */
export async function runNativeHost(reader: NativeMessageReader, handler: NativeMessageHandler): Promise<void> {
    logNativeHostError("Native Host Starting...");

    try {
        await handler.start();

        while (true) {
            const message = await reader.read();
            if (message === null) {
                logNativeHostError("End of stream reached or error occurred.");
                break;
            }
            await handler.handleMessage(message);
        }
    } catch (error: any) {
        logNativeHostError("Fatal error in Native Host loop:", error);
    } finally {
        await handler.stop();
        logNativeHostError("Native Host Stopped.");
    }
}
