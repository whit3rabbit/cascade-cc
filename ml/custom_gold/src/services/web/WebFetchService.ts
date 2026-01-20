
import axios, { AxiosError } from 'axios';
import { LRUCache } from 'lru-cache';
import TurndownService from '../../vendor/turndown.js';

// Allowed Domains (RI1 from chunk_475)
const ALLOWED_DOMAINS = new Set([
    "platform.claude.com", "code.claude.com", "modelcontextprotocol.io", "github.com/anthropics",
    "docs.python.org", "en.cppreference.com", "docs.oracle.com", "learn.microsoft.com",
    "developer.mozilla.org", "go.dev", "pkg.go.dev", "www.php.net", "docs.swift.org",
    "kotlinlang.org", "ruby-doc.org", "doc.rust-lang.org", "www.typescriptlang.org",
    "react.dev", "angular.io", "vuejs.org", "nextjs.org", "expressjs.com", "nodejs.org",
    "bun.sh", "jquery.com", "getbootstrap.com", "tailwindcss.com", "d3js.org", "threejs.org",
    "redux.js.org", "webpack.js.org", "jestjs.io", "reactrouter.com", "docs.djangoproject.com",
    "flask.palletsprojects.com", "fastapi.tiangolo.com", "pandas.pydata.org", "numpy.org",
    "www.tensorflow.org", "pytorch.org", "scikit-learn.org", "matplotlib.org",
    "requests.readthedocs.io", "jupyter.org", "laravel.com", "symfony.com", "wordpress.org",
    "docs.spring.io", "hibernate.org", "tomcat.apache.org", "gradle.org", "maven.apache.org",
    "asp.net", "dotnet.microsoft.com", "nuget.org", "blazor.net", "reactnative.dev",
    "docs.flutter.dev", "developer.apple.com", "developer.android.com", "keras.io",
    "spark.apache.org", "huggingface.co", "www.kaggle.com", "www.mongodb.com", "redis.io",
    "www.postgresql.org", "dev.mysql.com", "www.sqlite.org", "graphql.org", "prisma.io",
    "docs.aws.amazon.com", "cloud.google.com", "kubernetes.io", "www.docker.com",
    "www.terraform.io", "www.ansible.com", "vercel.com/docs", "docs.netlify.com",
    "devcenter.heroku.com/", "cypress.io", "selenium.dev", "docs.unity.com",
    "docs.unrealengine.com", "git-scm.com", "nginx.org", "httpd.apache.org"
]);

const MAX_CACHE_SIZE = 50 * 1024 * 1024; // 50MB
const CACHE_TTL = 15 * 60 * 1000; // 15 min
const MAX_CONTENT_LENGTH = 10 * 1024 * 1024; // 10MB
const TRUNCATE_LENGTH = 100000;

export class DomainBlockedError extends Error {
    constructor(domain: string) {
        super(`Claude Code is unable to fetch from ${domain}`);
        this.name = "DomainBlockedError";
    }
}

export class DomainCheckFailedError extends Error {
    constructor(domain: string) {
        super(`Unable to verify if domain ${domain} is safe to fetch. This may be due to network restrictions or enterprise security policies blocking claude.ai.`);
        this.name = "DomainCheckFailedError";
    }
}

// Cache
const fetchCache = new LRUCache<string, any>({
    maxSize: MAX_CACHE_SIZE,
    sizeCalculation: (val: any) => Buffer.byteLength(val.content),
    ttl: CACHE_TTL
});

// Domain Check (Ci5)
async function checkDomainSafety(domain: string): Promise<{ status: 'allowed' | 'blocked' | 'check_failed', error?: Error }> {
    try {
        const response = await axios.get(`https://claude.ai/api/web/domain_info?domain=${encodeURIComponent(domain)}`);
        if (response.status === 200) {
            return response.data.can_fetch === true ? { status: 'allowed' } : { status: 'blocked' };
        }
        return { status: 'check_failed', error: new Error(`Domain check returned status ${response.status}`) };
    } catch (err: any) {
        return { status: 'check_failed', error: err };
    }
}

// Validate URL (zi5)
function isValidUrl(url: string): boolean {
    if (url.length > 2000) return false;
    try {
        const u = new URL(url);
        if (u.username || u.password) return false;
        if (u.hostname.split('.').length < 2) return false;
        return true;
    } catch {
        return false;
    }
}

// Same Origin Check ($i5)
function isSameOrigin(url1: string, url2: string): boolean {
    try {
        const u1 = new URL(url1);
        const u2 = new URL(url2);
        if (u1.protocol !== u2.protocol) return false;
        if (u1.port !== u2.port) return false;
        if (u1.username || u1.password) return false;
        const stripWww = (s: string) => s.replace(/^www\./, '');
        return stripWww(u1.hostname) === stripWww(u2.hostname);
    } catch {
        return false;
    }
}

// Raw Fetch (Og2)
async function fetchUrlRaw(url: string, signal: AbortSignal, checkRedirect: (orig: string, newUrl: string) => boolean): Promise<any> {
    try {
        const response = await axios.get(url, {
            signal,
            maxRedirects: 0,
            responseType: 'arraybuffer',
            maxContentLength: MAX_CONTENT_LENGTH,
            headers: {
                'Accept': 'text/markdown, text/html, */*'
            }
        });
        return response;
    } catch (err: any) {
        if (axios.isAxiosError(err) && err.response && [301, 302, 307, 308].includes(err.response.status)) {
            const location = err.response.headers.location;
            if (!location) throw new Error("Redirect missing Location header");
            const redirectUrl = new URL(location, url).toString();
            if (checkRedirect(url, redirectUrl)) {
                return fetchUrlRaw(redirectUrl, signal, checkRedirect);
            } else {
                return {
                    type: 'redirect',
                    originalUrl: url,
                    redirectUrl,
                    statusCode: err.response.status
                };
            }
        }
        throw err;
    }
}

// Main Fetch Function (Mg2)
export async function performWebFetch(url: string, abortSignal: AbortSignal, options?: any) {
    if (!isValidUrl(url)) throw new Error("Invalid URL");

    const cached = fetchCache.get(url);
    if (cached) return cached;

    let targetUrl = url;
    try {
        const u = new URL(url);
        if (u.protocol === 'http:') {
            u.protocol = 'https:';
            targetUrl = u.toString();
        }
        const hostname = u.hostname;

        // Skip preflight check if configured? (zQ().skipWebFetchPreflight)
        // Check permissions logic here or in caller?
        // Implementing safety check
        const safety = await checkDomainSafety(hostname);
        if (safety.status === 'blocked') throw new DomainBlockedError(hostname);
        if (safety.status === 'check_failed') throw new DomainCheckFailedError(hostname);
    } catch (err: any) {
        if (err instanceof DomainBlockedError || err instanceof DomainCheckFailedError) throw err;
        // log err
    }

    const response = await fetchUrlRaw(targetUrl, abortSignal, isSameOrigin);
    if (response.type === 'redirect') return response;

    const contentBuffer = Buffer.from(response.data);
    const contentType = response.headers['content-type'] || '';
    let contentStr = contentBuffer.toString('utf8'); // Simplification for now, charset detection needed

    if (contentType.includes('text/html')) {
        // @ts-ignore
        const turndown = new TurndownService({});
        contentStr = turndown.turndown(contentStr);
    }

    if (contentStr.length > TRUNCATE_LENGTH) { // && !A31() (isPaid?)
        contentStr = contentStr.substring(0, TRUNCATE_LENGTH) + "...[content truncated]";
    }

    const result = {
        bytes: Buffer.byteLength(contentBuffer),
        code: response.status,
        codeText: response.statusText,
        content: contentStr,
        contentType
    };

    fetchCache.set(url, result);
    return result;
}

export { ALLOWED_DOMAINS };
