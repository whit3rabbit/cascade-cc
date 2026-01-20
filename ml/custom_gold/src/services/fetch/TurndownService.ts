import TurndownService from "turndown";
import axios from "axios";
import { log } from "../logger/loggerService.js";

const logger = log("TurndownService");

export const FETCH_CONFIG = {
    MAX_URL_LENGTH: 2000,
    MAX_CONTENT_LENGTH: 10 * 1024 * 1024, // 10MB
};

// Domain Allowlist from RI1 in chunk_475.ts
export const ALLOWED_DOMAINS = new Set([
    "platform.claude.com",
    "code.claude.com",
    "modelcontextprotocol.io",
    "github.com/anthropics",
    "docs.python.org",
    "en.cppreference.com",
    "docs.oracle.com",
    "learn.microsoft.com",
    "developer.mozilla.org",
    "go.dev",
    "pkg.go.dev",
    "www.php.net",
    "docs.swift.org",
    "kotlinlang.org",
    "ruby-doc.org",
    "doc.rust-lang.org",
    "www.typescriptlang.org",
    "react.dev",
    "angular.io",
    "vuejs.org",
    "nextjs.org",
    "expressjs.com",
    "nodejs.org",
    "bun.sh",
    "jquery.com",
    "getbootstrap.com",
    "tailwindcss.com",
    "d3js.org",
    "threejs.org",
    "redux.js.org",
    "webpack.js.org",
    "jestjs.io",
    "reactrouter.com",
    "docs.djangoproject.com",
    "flask.palletsprojects.com",
    "fastapi.tiangolo.com",
    "pandas.pydata.org",
    "numpy.org",
    "www.tensorflow.org",
    "pytorch.org",
    "scikit-learn.org",
    "matplotlib.org",
    "requests.readthedocs.io",
    "jupyter.org",
    "laravel.com",
    "symfony.com",
    "wordpress.org",
    "docs.spring.io",
    "hibernate.org",
    "tomcat.apache.org",
    "gradle.org",
    "maven.apache.org",
    "asp.net",
    "dotnet.microsoft.com",
    "nuget.org",
    "blazor.net",
    "reactnative.dev",
    "docs.flutter.dev",
    "developer.apple.com",
    "developer.android.com",
    "keras.io",
    "spark.apache.org",
    "huggingface.co",
    "www.kaggle.com",
    "www.mongodb.com",
    "redis.io",
    "www.postgresql.org",
    "dev.mysql.com",
    "www.sqlite.org",
    "graphql.org",
    "prisma.io",
    "docs.aws.amazon.com",
    "cloud.google.com",
    "kubernetes.io",
    "www.docker.com",
    "www.terraform.io",
    "www.ansible.com",
    "vercel.com/docs",
    "docs.netlify.com",
    "devcenter.heroku.com/",
    "cypress.io",
    "selenium.dev",
    "docs.unity.com",
    "docs.unrealengine.com",
    "git-scm.com",
    "nginx.org",
    "httpd.apache.org"
]);

/**
 * Checks if a domain is allowed based on the allowlist.
 * Logic from Lg2 in chunk_475.ts.
 */
export function isDomainAllowed(url: string): boolean {
    try {
        const parsed = new URL(url);
        const { hostname, pathname } = parsed;
        for (const domain of ALLOWED_DOMAINS) {
            if (domain.includes("/")) {
                const [host, ...pathParts] = domain.split("/");
                const pathPrefix = "/" + pathParts.join("/");
                if (hostname === host && pathname.startsWith(pathPrefix)) return true;
            } else if (hostname === domain) {
                return true;
            }
        }
        return false;
    } catch {
        return false;
    }
}

/**
 * Validates a URL for length and components.
 * Logic from zi5 in chunk_475.ts.
 */
export function isValidUrl(url: string): boolean {
    if (url.length > FETCH_CONFIG.MAX_URL_LENGTH) return false;
    try {
        const parsed = new URL(url);
        if (parsed.username || parsed.password) return false;
        if (parsed.hostname.split(".").length < 2) return false;
        return true;
    } catch {
        return false;
    }
}

/**
 * Checks if two URLs belong to the same domain (ignoring www).
 * Logic from $i5 in chunk_475.ts.
 */
export function isSameDomain(url1: string, url2: string): boolean {
    try {
        const u1 = new URL(url1);
        const u2 = new URL(url2);
        if (u1.protocol !== u2.protocol || u1.port !== u2.port) return false;
        if (u1.username || u1.password) return false;

        const normalize = (h: string) => h.replace(/^www\./, "");
        return normalize(u1.hostname) === normalize(u2.hostname);
    } catch {
        return false;
    }
}

/**
 * Performs a fetch request using axios, handling redirects and size limits.
 * Logic from Og2 in chunk_475.ts.
 */
export async function performFetch(
    url: string,
    signal?: AbortSignal,
    allowRedirects: (oldUrl: string, newUrl: string) => boolean = () => true
): Promise<any> {
    try {
        const response = await axios.get(url, {
            signal,
            maxRedirects: 0,
            responseType: "arraybuffer",
            maxContentLength: FETCH_CONFIG.MAX_CONTENT_LENGTH,
            headers: {
                "Accept": "text/markdown, text/html, */*"
            }
        });
        return response;
    } catch (error: any) {
        if (axios.isAxiosError(error) && error.response && [301, 302, 307, 308].includes(error.response.status)) {
            const location = error.response.headers.location;
            if (!location) throw new Error("Redirect missing Location header");
            const nextUrl = new URL(location, url).toString();
            if (allowRedirects(url, nextUrl)) {
                return performFetch(nextUrl, signal, allowRedirects);
            } else {
                return {
                    type: "redirect",
                    originalUrl: url,
                    redirectUrl: nextUrl,
                    statusCode: error.response.status
                };
            }
        }
        throw error;
    }
}

/**
 * Checks if a domain is allowed via Anthropic's API.
 * Logic from Ci5 in chunk_475.ts.
 */
export async function checkDomainInfo(domain: string): Promise<{ status: "allowed" | "blocked" | "check_failed"; error?: any }> {
    try {
        const response = await axios.get(`https://claude.ai/api/web/domain_info?domain=${encodeURIComponent(domain)}`);
        if (response.status === 200) {
            return response.data.can_fetch === true ? { status: "allowed" } : { status: "blocked" };
        }
        return { status: "check_failed", error: new Error(`Domain check returned status ${response.status}`) };
    } catch (error) {
        logger.error(`Domain check failed: ${error}`);
        return { status: "check_failed", error };
    }
}

/**
 * Custom Turndown Service with specific rules matching chunk_475.ts.
 */
export function createTurndownService() {
    const td = new TurndownService({
        headingStyle: "atx",
        hr: "* * *",
        bulletListMarker: "*",
        codeBlockStyle: "fenced",
        fence: "```",
        emDelimiter: "_",
        strongDelimiter: "**",
        linkStyle: "inlined",
        linkReferenceStyle: "full"
    });

    // Custom Rules from Sz in chunk_475.ts
    td.addRule("paragraph", {
        filter: "p",
        replacement: (content) => `\n\n${content}\n\n`
    });

    td.addRule("lineBreak", {
        filter: "br",
        replacement: (content, node, options) => (options as any).br + "\n"
    });

    td.addRule("heading", {
        filter: ["h1", "h2", "h3", "h4", "h5", "h6"],
        replacement: (content, node, options) => {
            const level = Number(node.nodeName.charAt(1));
            return `\n\n${"#".repeat(level)} ${content}\n\n`;
        }
    });

    td.addRule("blockquote", {
        filter: "blockquote",
        replacement: (content) => {
            const trimmed = content.trim().replace(/^/gm, "> ");
            return `\n\n${trimmed}\n\n`;
        }
    });

    td.addRule("list", {
        filter: ["ul", "ol"],
        replacement: (content, node) => {
            if (node.parentNode?.nodeName === "LI" && node.parentNode.lastElementChild === node) {
                return `\n${content}`;
            }
            return `\n\n${content}\n\n`;
        }
    });

    td.addRule("listItem", {
        filter: "li",
        replacement: (content, node, options) => {
            const cleaned = content.replace(/^\n+/, "").replace(/\n+$/, "\n").replace(/\n/gm, "\n    ");
            let marker = (options as any).bulletListMarker + "   ";
            const parent = node.parentNode;
            if (parent?.nodeName === "OL") {
                const start = (parent as any).getAttribute("start");
                const index = Array.prototype.indexOf.call(parent.children, node);
                marker = (start ? Number(start) + index : index + 1) + ".  ";
            }
            return marker + cleaned + (node.nextSibling && !/\n$/.test(cleaned) ? "\n" : "");
        }
    });

    td.addRule("indentedCodeBlock", {
        filter: (node, options) => (options as any).codeBlockStyle === "indented" && node.nodeName === "PRE" && node.firstChild?.nodeName === "CODE",
        replacement: (content, node) => {
            const code = node.firstChild?.textContent || "";
            return `\n\n    ${code.replace(/\n/g, "\n    ")}\n\n`;
        }
    });

    td.addRule("fencedCodeBlock", {
        filter: (node, options) => (options as any).codeBlockStyle === "fenced" && node.nodeName === "PRE" && node.firstChild?.nodeName === "CODE",
        replacement: (content, node, options) => {
            const firstChild = node.firstChild as HTMLElement;
            const className = firstChild.getAttribute("class") || "";
            const language = (className.match(/language-(\S+)/) || [null, ""])[1];
            const code = firstChild.textContent || "";
            const fenceChar = (options as any).fence.charAt(0);

            // Calculate required fence length
            let fenceSize = 3;
            const re = new RegExp("^" + fenceChar + "{3,}", "gm");
            let match;
            while ((match = re.exec(code))) {
                if (match[0].length >= fenceSize) fenceSize = match[0].length + 1;
            }
            const fence = fenceChar.repeat(fenceSize);

            return `\n\n${fence}${language}\n${code.replace(/\n$/, "")}\n${fence}\n\n`;
        }
    });

    td.addRule("inlineLink", {
        filter: (node, options) => (options as any).linkStyle === "inlined" && node.nodeName === "A" && (node as HTMLElement).getAttribute("href") !== null,
        replacement: (content, node) => {
            const href = (node as HTMLElement).getAttribute("href") || "";
            const title = (node as HTMLElement).getAttribute("title");
            const escapedHref = href.replace(/([()])/g, "\\$1");
            const titlePart = title ? ` "${title.replace(/"/g, '\\"')}"` : "";
            return `[${content}](${escapedHref}${titlePart})`;
        }
    });

    td.addRule("emphasis", {
        filter: ["em", "i"],
        replacement: (content, node, options) => {
            if (!content.trim()) return "";
            return (options as any).emDelimiter + content + (options as any).emDelimiter;
        }
    });

    td.addRule("strong", {
        filter: ["strong", "b"],
        replacement: (content, node, options) => {
            if (!content.trim()) return "";
            return (options as any).strongDelimiter + content + (options as any).strongDelimiter;
        }
    });

    td.addRule("code", {
        filter: (node) => {
            const hasSibling = node.previousSibling || node.nextSibling;
            const isPreChild = node.parentNode?.nodeName === "PRE" && !hasSibling;
            return node.nodeName === "CODE" && !isPreChild;
        },
        replacement: (content) => {
            if (!content) return "";
            const cleaned = content.replace(/\r?\n|\r/g, " ");
            const extraSpace = /^`|^ .*?[^ ].* $|`$/.test(cleaned) ? " " : "";
            const backticks = "`";
            // Handle nested backticks (simplified from chunk_475)
            let fence = backticks;
            while (cleaned.includes(fence)) fence += "`";
            return fence + extraSpace + cleaned + extraSpace + fence;
        }
    });

    td.addRule("image", {
        filter: "img",
        replacement: (content, node) => {
            const alt = (node as HTMLElement).getAttribute("alt") || "";
            const src = (node as HTMLElement).getAttribute("src") || "";
            const title = (node as HTMLElement).getAttribute("title");
            const titlePart = title ? ` "${title}"` : "";
            return src ? `![${alt}](${src}${titlePart})` : "";
        }
    });

    return td;
}

/**
 * Converts HTML string to Markdown.
 */
export function convertHtmlToMarkdown(html: string): string {
    const td = createTurndownService();
    return td.turndown(html);
}
