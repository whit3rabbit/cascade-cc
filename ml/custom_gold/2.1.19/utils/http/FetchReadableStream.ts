/**
 * File: src/utils/http/FetchReadableStream.ts
 * Role: A ReadableStream implementation that mimics the Fetch API body's behavior.
 */

import { Readable as NodeReadable } from "node:stream";
import { RequestAbortedError } from "../shared/errors.js";

const kConsume = Symbol("kConsume");
const kAbort = Symbol("kAbort");
const kContentType = Symbol("kContentType");
const kContentLength = Symbol("kContentLength");

interface FetchReadableStreamOptions {
    resume?: (size: number) => void;
    abort: () => void;
    contentType?: string;
    contentLength?: number;
    highWaterMark?: number;
}

interface ConsumeState {
    type: 'text' | 'json' | 'blob' | 'bytes' | 'arrayBuffer';
    resolve: (value: any) => void;
    reject: (reason: any) => void;
    body: Buffer[];
    length: number;
}

/**
 * Enhanced ReadableStream for fetch-like interactions in a Node.js environment.
 */
export class FetchReadableStream extends NodeReadable {
    private [kAbort]: () => void;
    private [kConsume]: ConsumeState | null = null;
    private [kContentType]: string;
    private [kContentLength]?: number;

    constructor({ resume, abort, contentType = "", contentLength, highWaterMark = 65536 }: FetchReadableStreamOptions) {
        super({ autoDestroy: true, read: resume, highWaterMark });
        this[kAbort] = abort;
        this[kContentType] = contentType;
        this[kContentLength] = contentLength;
    }

    override destroy(err?: Error | null): this {
        let error = err;
        if (!error && !this.readableEnded) {
            error = new RequestAbortedError();
        }

        if (error) {
            this[kAbort]();
        }
        return super.destroy(error ?? undefined);
    }

    async text(): Promise<string> { return this.consume("text"); }
    async json(): Promise<any> { return this.consume("json"); }
    async blob(): Promise<Blob> { return this.consume("blob"); }
    async bytes(): Promise<Uint8Array> { return this.consume("bytes"); }
    async arrayBuffer(): Promise<ArrayBuffer> { return this.consume("arrayBuffer"); }

    private async consume(type: ConsumeState['type']): Promise<any> {
        return new Promise((resolve, reject) => {
            if (this.readableEnded || this.destroyed) {
                return reject(new Error("Stream already closed or destroyed"));
            }

            const state: ConsumeState = { type, resolve, reject, body: [], length: 0 };
            this[kConsume] = state;

            this.on("data", (chunk: Buffer) => {
                state.body.push(chunk);
                state.length += chunk.length;
            });

            this.on("end", () => {
                const { body, length, type, resolve } = state;
                const buffer = Buffer.concat(body, length);

                try {
                    switch (type) {
                        case "text":
                            resolve(buffer.toString('utf8'));
                            break;
                        case "json":
                            resolve(JSON.parse(buffer.toString('utf8')));
                            break;
                        case "bytes":
                            resolve(new Uint8Array(buffer));
                            break;
                        case "arrayBuffer":
                            // Buffer.buffer might be shared, so we slice it to get exactly the right data
                            resolve(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
                            break;
                        case "blob":
                            resolve(new Blob([buffer], { type: this[kContentType] }));
                            break;
                    }
                } catch (e) {
                    reject(e);
                }
            });

            this.on("error", (err) => reject(err));
            this.resume();
        });
    }
}
