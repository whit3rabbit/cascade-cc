import * as http from "node:http";
import * as https from "node:https";
import * as net from "node:net";
import { URL } from "node:url";
import { createSocksServer, Socks5Server } from "../../vendor/socks5Server.js";

/**
 * Sandbox proxy utilities.
 * Deobfuscated from uFB, rFB in chunk_219.ts.
 */

export interface NetworkFilter {
    filter(port: number, host: string, socket?: net.Socket): Promise<boolean>;
}

function logDebug(msg: string, level: "info" | "warn" | "error" = "info") {
    if (!process.env.SRT_DEBUG) return;
    const prefix = "[SandboxDebug]";
    switch (level) {
        case "error": console.error(`${prefix} ${msg}`); break;
        case "warn": console.warn(`${prefix} ${msg}`); break;
        default: console.info(`${prefix} ${msg}`); break;
    }
}

/**
 * Creates an HTTP proxy that filters requests.
 */
export function createHttpProxy(filter: NetworkFilter): http.Server {
    const server = http.createServer();

    // Handle HTTPS via CONNECT
    server.on("connect", async (req, socket, head) => {
        socket.on("error", (err) => logDebug(`Client socket error: ${err.message}`, "error"));

        try {
            const [host, portStr] = req.url!.split(":");
            const port = portStr ? parseInt(portStr, 10) : 443;

            if (!host || isNaN(port)) {
                logDebug(`Invalid CONNECT request: ${req.url}`, "error");
                socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
                return;
            }

            if (!await filter.filter(port, host, socket as net.Socket)) {
                logDebug(`Connection blocked to ${host}:${port}`, "error");
                socket.end("HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nX-Proxy-Error: blocked-by-allowlist\r\n\r\nConnection blocked by network allowlist");
                return;
            }

            const target = net.connect(port, host, () => {
                socket.write("HTTP/1.1 200 Connection Established\r\n\r\n");
                target.pipe(socket);
                socket.pipe(target);
            });

            target.on("error", (err) => {
                logDebug(`CONNECT tunnel failed: ${err.message}`, "error");
                socket.end("HTTP/1.1 502 Bad Gateway\r\n\r\n");
            });

            socket.on("end", () => target.end());
            target.on("end", () => socket.end());
        } catch (err) {
            logDebug(`Error handling CONNECT: ${err}`, "error");
            socket.end("HTTP/1.1 500 Internal Server Error\r\n\r\n");
        }
    });

    // Handle standard HTTP requests
    server.on("request", async (req, res) => {
        try {
            const url = new URL(req.url!, `http://${req.headers.host}`);
            const host = url.hostname;
            const port = url.port ? parseInt(url.port, 10) : (url.protocol === "https:" ? 443 : 80);

            if (!await filter.filter(port, host, req.socket)) {
                logDebug(`HTTP request blocked to ${host}:${port}`, "error");
                res.writeHead(403, {
                    "Content-Type": "text/plain",
                    "X-Proxy-Error": "blocked-by-allowlist"
                });
                res.end("Connection blocked by network allowlist");
                return;
            }

            const proxyReq = (url.protocol === "https:" ? https : http).request({
                hostname: host,
                port: port,
                path: url.pathname + url.search,
                method: req.method,
                headers: {
                    ...req.headers,
                    host: url.host
                }
            }, (proxyRes) => {
                res.writeHead(proxyRes.statusCode!, proxyRes.headers);
                proxyRes.pipe(res);
            });

            proxyReq.on("error", (err) => {
                logDebug(`Proxy request failed: ${err.message}`, "error");
                if (!res.headersSent) {
                    res.writeHead(502, { "Content-Type": "text/plain" });
                    res.end("Bad Gateway");
                }
            });

            req.pipe(proxyReq);
        } catch (err) {
            logDebug(`Error handling HTTP request: ${err}`, "error");
            res.writeHead(500, { "Content-Type": "text/plain" });
            res.end("Internal Server Error");
        }
    });

    return server;
}

/**
 * Creates a SOCKS5 proxy that filters requests.
 */
export function createSocksProxy(filter: NetworkFilter) {
    const server = createSocksServer();

    server.rulesetValidator = async (conn) => {
        try {
            const { destAddress, destPort } = conn;
            logDebug(`Connection request to ${destAddress}:${destPort}`);

            if (!destAddress || !destPort || !await filter.filter(destPort, destAddress)) {
                logDebug(`Connection blocked to ${destAddress}:${destPort}`, "error");
                return false;
            }

            logDebug(`Connection allowed to ${destAddress}:${destPort}`);
            return true;
        } catch (err) {
            logDebug(`Error validating connection: ${err}`, "error");
            return false;
        }
    };

    return {
        server,
        getPort() {
            const addr = (server as any).server?.address();
            return addr?.port;
        },
        async listen(port: number, hostname?: string): Promise<number> {
            return new Promise((resolve, reject) => {
                server.listen(port, hostname, () => {
                    const p = this.getPort();
                    if (p) {
                        logDebug(`SOCKS proxy listening on ${hostname || "all"}:${p}`);
                        resolve(p);
                    } else {
                        reject(new Error("Failed to get SOCKS proxy port"));
                    }
                });
            });
        },
        async close() {
            return new Promise<void>((resolve, reject) => {
                server.close((err) => {
                    if (err) {
                        const msg = err.message.toLowerCase();
                        if (msg.includes("not running") || msg.includes("already closed")) {
                            return resolve();
                        }
                        return reject(err);
                    }
                    resolve();
                });
            });
        }
    };
}
