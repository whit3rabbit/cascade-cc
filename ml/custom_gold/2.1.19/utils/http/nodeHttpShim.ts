/**
 * File: src/utils/http/nodeHttpShim.ts
 * Role: Shims and utilities for Node.js native HTTP/HTTPS modules.
 */

import { Agent, createServer, request as httpRequest, RequestOptions } from "node:http";
import { request as httpsRequest } from "node:https";
import { connect as netConnect } from "node:net";
import { URL } from "node:url";

/**
 * Converts a URL object into an options object compatible with http.request.
 * 
 * @param url - The URL object to convert.
 * @returns An options object for HTTP requests.
 */
export function urlToHttpOptions(url: URL): RequestOptions & { [key: string]: any } {
    const options: RequestOptions & { [key: string]: any } = {
        protocol: url.protocol,
        hostname: url.hostname?.startsWith("[") ? url.hostname.slice(1, -1) : url.hostname,
        hash: url.hash,
        search: url.search,
        pathname: url.pathname,
        path: `${url.pathname || ""}${url.search || ""}`,
        href: url.href,
    };

    if (url.port) {
        options.port = Number(url.port);
    }

    if (url.username || url.password) {
        options.auth = `${decodeURIComponent(url.username || "")}:${decodeURIComponent(url.password || "")}`;
    }

    return options;
}

export {
    Agent,
    createServer,
    httpRequest,
    httpsRequest,
    netConnect,
    URL as URLConstructor,
};
