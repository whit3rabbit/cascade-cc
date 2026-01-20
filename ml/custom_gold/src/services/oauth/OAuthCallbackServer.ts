import { createServer, Server, IncomingMessage, ServerResponse } from "http";
import { log, logError } from "../logger/loggerService.js";
import { getAuthHeaders, getEnvHelper } from "../auth/AuthConfig.js";
import { axiosInstance } from "../fetch/axiosService.js";
import { getUserAgent } from "../auth/UserAgent.js";

const env = getEnvHelper();

// Helper for debug logging
const debug = (msg: string) => log("oauth").debug(msg);

export class OAuthCallbackServer {
    private localServer: Server;
    private port: number = 0;
    private promiseResolver: ((value: string) => void) | null = null;
    private promiseRejecter: ((reason?: any) => void) | null = null;
    private expectedState: string | null = null;
    private pendingResponse: ServerResponse | null = null;
    private callbackPath: string;

    constructor(callbackPath: string = "/callback") {
        this.localServer = createServer();
        this.callbackPath = callbackPath;
    }

    async start(port?: number): Promise<number> {
        return new Promise((resolve, reject) => {
            this.localServer.once("error", (err: Error) => {
                reject(new Error(`Failed to start OAuth callback server: ${err.message}`));
            });

            this.localServer.listen(port ?? 0, "localhost", () => {
                const address = this.localServer.address();
                if (address && typeof address !== 'string') {
                    this.port = address.port;
                    resolve(this.port);
                } else {
                    reject(new Error("Failed to get server address"));
                }
            });
        });
    }

    getPort(): number {
        return this.port;
    }

    hasPendingResponse(): boolean {
        return this.pendingResponse !== null;
    }

    async waitForAuthorization(state: string, onStart: () => void): Promise<string> {
        return new Promise((resolve, reject) => {
            this.promiseResolver = resolve;
            this.promiseRejecter = reject;
            this.expectedState = state;
            this.startLocalListener(onStart);
        });
    }

    handleSuccessRedirect(code: string, customHandler?: (res: ServerResponse, code: string) => void) {
        if (!this.pendingResponse) return;

        if (customHandler) {
            customHandler(this.pendingResponse, code);
            this.pendingResponse = null;
            return;
        }

        const location = isClaudeAi(code) ? env.CLAUDEAI_SUCCESS_URL : env.CONSOLE_SUCCESS_URL;
        this.pendingResponse.writeHead(302, { Location: location });
        this.pendingResponse.end();
        this.pendingResponse = null;
    }

    handleErrorRedirect() {
        if (!this.pendingResponse) return;

        const location = env.CLAUDEAI_SUCCESS_URL;
        this.pendingResponse.writeHead(302, { Location: location });
        this.pendingResponse.end();
        this.pendingResponse = null;
    }

    startLocalListener(onStart: () => void) {
        this.localServer.on("request", this.handleRedirect.bind(this));
        this.localServer.on("error", this.handleError.bind(this));
        onStart();
    }

    handleRedirect(req: IncomingMessage, res: ServerResponse) {
        const url = new URL(req.url || "", `http://${req.headers.host || "localhost"}`);
        if (url.pathname !== this.callbackPath) {
            res.writeHead(404);
            res.end();
            return;
        }

        const code = url.searchParams.get("code") ?? undefined;
        const state = url.searchParams.get("state") ?? undefined;
        this.validateAndRespond(code, state, res);
    }

    validateAndRespond(code: string | undefined, state: string | undefined, res: ServerResponse) {
        if (!code) {
            res.writeHead(400);
            res.end("Authorization code not found");
            this.reject(new Error("No authorization code received"));
            return;
        }

        if (state !== this.expectedState) {
            res.writeHead(400);
            res.end("Invalid state parameter");
            this.reject(new Error("Invalid state parameter"));
            return;
        }

        this.pendingResponse = res;
        this.resolve(code);
    }

    handleError(err: Error) {
        logError("oauth", err, "OAuth server error");
        this.close();
        this.reject(err);
    }

    resolve(code: string) {
        if (this.promiseResolver) {
            this.promiseResolver(code);
            this.promiseResolver = null;
            this.promiseRejecter = null;
        }
    }

    reject(reason: any) {
        if (this.promiseRejecter) {
            this.promiseRejecter(reason);
            this.promiseResolver = null;
            this.promiseRejecter = null;
        }
    }

    close() {
        if (this.pendingResponse) {
            this.handleErrorRedirect();
        }
        if (this.localServer) {
            this.localServer.removeAllListeners();
            this.localServer.close();
        }
    }
}

function isClaudeAi(code: string): boolean {
    return true;
}

export async function fetchGroveConfig() {
    try {
        const auth = await getAuthHeaders();
        if ('error' in auth) {
            debug(`Failed to get auth headers: ${auth.error}`);
            return null;
        }

        const response = await axiosInstance.get(`${env.BASE_API_URL}/api/claude_code_grove`, {
            headers: {
                ...auth.headers,
                "User-Agent": getUserAgent()
            }
        });

        const {
            grove_enabled,
            domain_excluded,
            notice_is_grace_period,
            notice_reminder_frequency
        } = response.data;

        return {
            grove_enabled,
            domain_excluded: domain_excluded ?? false,
            notice_is_grace_period: notice_is_grace_period ?? true,
            notice_reminder_frequency
        };
    } catch (err) {
        debug(`Failed to fetch Grove notice config: ${err}`);
        return null;
    }
}
