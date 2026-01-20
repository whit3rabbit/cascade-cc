
/**
 * Based on chunk_477.ts:401-1473
 * This is a vendored and simplified version of uri-js logic.
 */

export interface URIComponents {
    scheme?: string;
    authority?: string;
    userinfo?: string;
    host?: string;
    port?: number | string;
    path?: string;
    query?: string;
    fragment?: string;
}

const SCHEME = /^[a-z][a-z0-9+\-.]*:/i;

export function parse(uriString: string): URIComponents {
    const components: URIComponents = {};
    let remaining = uriString;

    // Very simplified parser based on the spirit of the obfuscated code
    const schemeMatch = remaining.match(SCHEME);
    if (schemeMatch) {
        components.scheme = schemeMatch[0].slice(0, -1).toLowerCase();
        remaining = remaining.slice(schemeMatch[0].length);
    }

    if (remaining.startsWith('//')) {
        remaining = remaining.slice(2);
        const authorityEnd = remaining.match(/[\/\?#]/);
        const authority = authorityEnd ? remaining.slice(0, authorityEnd.index) : remaining;
        components.authority = authority;
        remaining = authorityEnd ? remaining.slice(authorityEnd.index) : "";

        const userinfoEnd = authority.indexOf('@');
        if (userinfoEnd !== -1) {
            components.userinfo = authority.slice(0, userinfoEnd);
            const hostPort = authority.slice(userinfoEnd + 1);
            const portStart = hostPort.lastIndexOf(':');
            if (portStart !== -1) {
                components.host = hostPort.slice(0, portStart);
                components.port = hostPort.slice(portStart + 1);
            } else {
                components.host = hostPort;
            }
        } else {
            const portStart = authority.lastIndexOf(':');
            if (portStart !== -1) {
                components.host = authority.slice(0, portStart);
                components.port = authority.slice(portStart + 1);
            } else {
                components.host = authority;
            }
        }
    }

    const fragmentStart = remaining.indexOf('#');
    if (fragmentStart !== -1) {
        components.fragment = remaining.slice(fragmentStart + 1);
        remaining = remaining.slice(0, fragmentStart);
    }

    const queryStart = remaining.indexOf('?');
    if (queryStart !== -1) {
        components.query = remaining.slice(queryStart + 1);
        remaining = remaining.slice(0, queryStart);
    }

    components.path = remaining;

    return components;
}

export function serialize(components: URIComponents): string {
    let uri = "";
    if (components.scheme) uri += components.scheme + ":";
    if (components.authority || components.host) {
        uri += "//";
        if (components.userinfo) uri += components.userinfo + "@";
        if (components.host) uri += components.host;
        if (components.port) uri += ":" + components.port;
    }
    if (components.path) uri += components.path;
    if (components.query) uri += "?" + components.query;
    if (components.fragment) uri += "#" + components.fragment;
    return uri;
}

export function resolve(base: string, relative: string): string {
    // Simplified resolution logic
    if (relative.match(SCHEME)) return relative;
    if (relative.startsWith('//')) return parse(base).scheme + ":" + relative;

    const baseParsed = parse(base);
    if (relative.startsWith('/')) {
        return serialize({ ...baseParsed, path: relative, query: undefined, fragment: undefined });
    }

    // Relative path resolution
    const basePath = baseParsed.path || "/";
    const lastSlash = basePath.lastIndexOf('/');
    const newPath = basePath.slice(0, lastSlash + 1) + relative;
    return serialize({ ...baseParsed, path: newPath, query: undefined, fragment: undefined });
}
