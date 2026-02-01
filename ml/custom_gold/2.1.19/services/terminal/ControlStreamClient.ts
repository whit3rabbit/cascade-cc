/**
 * File: src/services/terminal/ControlStreamClient.ts
 * Role: Orchestrates the connection between a transport (WebSocket/Terminal) and the ControlStreamService.
 */

import { PassThrough } from "node:stream";
import { ControlStreamService, ControlStreamServiceOptions } from "./ControlStreamService.js";
import { WebSocketTransport } from "./WebSocketTransport.js";
import { getAuthHeaders } from "../auth/AuthService.js";
import { EnvService } from '../config/EnvService.js';

/**
 * Client for managing the control stream lifecycle.
 */
export class ControlStreamClient extends ControlStreamService {
    private inputStream: PassThrough;
    private url: URL;
    private options: ControlStreamServiceOptions;
    private transport: WebSocketTransport | null = null;

    constructor(url: string, initialInput: string[] = [], options: ControlStreamServiceOptions = {}) {
        const inputStream = new PassThrough({ encoding: "utf8" });

        // Pass a wrapper around this.writeToTransport to handle the async nature correctly in the super constructor hook if needed
        // but super expects (line: string) => void.
        const outputHandler = (msg: string) => {
            this.writeToTransport(msg).catch(err => {
                console.error("Failed to write to transport:", err);
            });
        };

        super(inputStream, outputHandler, options.replayUserMessages);

        this.inputStream = inputStream;
        this.url = new URL(url);
        this.options = options;

        this.initializeConnection(initialInput);
    }

    async initializeConnection(initialInput: string[]): Promise<void> {
        const headers = await getAuthHeaders();



        this.transport = new WebSocketTransport(this.url.toString(),
            { headers: { ...headers, "x-claude-session-id": EnvService.get("CLAUDE_SESSION_ID") || "" } }
        );

        this.transport.onData((data: any) => this.inputStream.write(data));
        this.transport.onClose(() => this.inputStream.end());

        await this.transport.connect();

        // Feed initial messages if any
        for (const line of initialInput) {
            this.inputStream.write(line + "\n");
        }
    }

    async writeToTransport(message: any): Promise<void> {
        if (this.transport) {
            // Check if message is already string, otherwise stringify
            const payload = typeof message === 'string' ? message : JSON.stringify(message);
            this.transport.write(payload); // WebSocketTransport.write expects any, but usually handles string/json
        }
    }

    close(): void {
        this.transport?.close();
        this.inputStream.end();
    }
}
