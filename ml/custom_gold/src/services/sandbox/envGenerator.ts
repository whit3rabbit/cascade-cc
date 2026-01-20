import * as os from "node:os";

/**
 * Generates environment variables for the sandbox to use internal proxies.
 * Deobfuscated from yA1 in chunk_220.ts.
 */
export function generateSandboxEnv(httpProxyPort?: number, socksProxyPort?: number): string[] {
    const env = ["SANDBOX_RUNTIME=1", "TMPDIR=/tmp/claude"];

    if (!httpProxyPort && !socksProxyPort) return env;

    const noProxy = [
        "localhost",
        "127.0.0.1",
        "::1",
        "*.local",
        ".local",
        "169.254.0.0/16",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16"
    ].join(",");

    env.push(`NO_PROXY=${noProxy}`);
    env.push(`no_proxy=${noProxy}`);

    if (httpProxyPort) {
        const httpProxy = `http://localhost:${httpProxyPort}`;
        env.push(`HTTP_PROXY=${httpProxy}`);
        env.push(`HTTPS_PROXY=${httpProxy}`);
        env.push(`http_proxy=${httpProxy}`);
        env.push(`https_proxy=${httpProxy}`);

        // Cloud SDK specific
        env.push("CLOUDSDK_PROXY_TYPE=https");
        env.push("CLOUDSDK_PROXY_ADDRESS=localhost");
        env.push(`CLOUDSDK_PROXY_PORT=${httpProxyPort}`);
    }

    if (socksProxyPort) {
        const socksProxy = `socks5h://localhost:${socksProxyPort}`;
        env.push(`ALL_PROXY=${socksProxy}`);
        env.push(`all_proxy=${socksProxy}`);
        env.push(`FTP_PROXY=${socksProxy}`);
        env.push(`ftp_proxy=${socksProxy}`);
        env.push(`RSYNC_PROXY=localhost:${socksProxyPort}`);
        env.push(`GRPC_PROXY=${socksProxy}`);
        env.push(`grpc_proxy=${socksProxy}`);

        if (os.platform() === "darwin") {
            env.push(`GIT_SSH_COMMAND="ssh -o ProxyCommand='nc -X 5 -x localhost:${socksProxyPort} %h %p'"`);
        }

        // Docker proxy (uses HTTP bridge to SOCKS if needed)
        const dockerProxyPort = httpProxyPort || socksProxyPort;
        env.push(`DOCKER_HTTP_PROXY=http://localhost:${dockerProxyPort}`);
        env.push(`DOCKER_HTTPS_PROXY=http://localhost:${dockerProxyPort}`);
    }

    return env;
}
