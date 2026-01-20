import * as net from "node:net";
import { EventEmitter } from "node:events";

/**
 * SOCKS5 Protocol constants.
 * Deobfuscated from aFB in chunk_219.ts.
 */

export enum SocksCommand {
    CONNECT = 1,
    BIND = 2,
    UDP_ASSOCIATE = 3
}

export enum SocksStatus {
    REQUEST_GRANTED = 0,
    GENERAL_FAILURE = 1,
    CONNECTION_NOT_ALLOWED = 2,
    NETWORK_UNREACHABLE = 3,
    HOST_UNREACHABLE = 4,
    CONNECTION_REFUSED = 5,
    TTL_EXPIRED = 6,
    COMMAND_NOT_SUPPORTED = 7,
    ADDRESS_TYPE_NOT_SUPPORTED = 8
}

export interface SocksServerOptions {
    auth?: {
        username: string;
        password?: string;
    };
    port?: number;
    hostname?: string;
}

export class SocksConnection {
    public username?: string;
    public password?: string;
    public command?: SocksCommand;
    public destAddress?: string;
    public destPort?: number;
    public metadata: Record<string, any> = {};

    constructor(
        private server: Socks5Server,
        public socket: net.Socket
    ) {
        socket.on("error", () => { });
        socket.pause();
        this.handleGreeting();
    }

    private async readBytes(n: number): Promise<Buffer> {
        return new Promise((resolve) => {
            let buffer = Buffer.allocUnsafe(n);
            let offset = 0;

            const onData = (chunk: Buffer) => {
                let toCopy = Math.min(chunk.length, n - offset);
                chunk.copy(buffer, offset, 0, toCopy);
                offset += toCopy;

                if (offset < n) return;

                this.socket.removeListener("data", onData);
                // Push back remaining data
                if (chunk.length > toCopy) {
                    this.socket.unshift(chunk.subarray(toCopy));
                }
                resolve(buffer);
                this.socket.pause();
            };

            this.socket.on("data", onData);
            this.socket.resume();
        });
    }

    private async handleGreeting() {
        const version = (await this.readBytes(1)).readUInt8();
        if (version !== 5) return this.socket.destroy();

        const nMethods = (await this.readBytes(1)).readUInt8();
        if (nMethods === 0 || nMethods > 128) return this.socket.destroy();

        const methods = await this.readBytes(nMethods);
        const requiredMethod = this.server.authHandler ? 2 : 0; // 0 = No auth, 2 = User/Pass

        if (!methods.includes(requiredMethod)) {
            this.socket.write(Buffer.from([5, 0xff]));
            return this.socket.destroy();
        }

        this.socket.write(Buffer.from([5, requiredMethod]));

        if (this.server.authHandler) {
            this.handleUserPassword();
        } else {
            this.handleConnectionRequest();
        }
    }

    private async handleUserPassword() {
        // Protocol version for sub-auth
        await this.readBytes(1);

        const ulen = (await this.readBytes(1)).readUInt8();
        const uname = (await this.readBytes(ulen)).toString();

        const plen = (await this.readBytes(1)).readUInt8();
        const pass = (await this.readBytes(plen)).toString();

        this.username = uname;
        this.password = pass;

        let decided = false;
        const accept = () => {
            if (decided) return;
            decided = true;
            this.socket.write(Buffer.from([1, 0]));
            this.handleConnectionRequest();
        };
        const reject = () => {
            if (decided) return;
            decided = true;
            this.socket.write(Buffer.from([1, 1]));
            this.socket.destroy();
        };

        if (this.server.authHandler) {
            const result = await this.server.authHandler(this, accept, reject);
            if (result === true) accept();
            else if (result === false) reject();
        }
    }

    private async handleConnectionRequest() {
        await this.readBytes(1); // Skip version
        const cmdByte = (await this.readBytes(1))[0];
        this.command = cmdByte as SocksCommand;

        if (!this.server.supportedCommands.has(this.command)) {
            this.socket.write(Buffer.from([5, SocksStatus.COMMAND_NOT_SUPPORTED, 0, 1, 0, 0, 0, 0, 0, 0]));
            return this.socket.destroy();
        }

        await this.readBytes(1); // Skip RSV
        const atyp = (await this.readBytes(1)).readUInt8();
        let address = "";

        switch (atyp) {
            case 1: // IPv4
                address = (await this.readBytes(4)).join(".");
                break;
            case 3: // Domain name
                const len = (await this.readBytes(1)).readUInt8();
                address = (await this.readBytes(len)).toString();
                break;
            case 4: // IPv6
                const ipv6 = await this.readBytes(16);
                for (let i = 0; i < 16; i += 2) {
                    if (i > 0) address += ":";
                    address += ipv6.readUInt16BE(i).toString(16);
                }
                break;
            default:
                this.socket.destroy();
                return;
        }

        const port = (await this.readBytes(2)).readUInt16BE();
        this.destAddress = address;
        this.destPort = port;

        let decided = false;
        const accept = () => {
            if (decided) return;
            decided = true;
            this.establish();
        };
        const reject = () => {
            if (decided) return;
            decided = true;
            this.socket.write(Buffer.from([5, SocksStatus.CONNECTION_NOT_ALLOWED, 0, 1, 0, 0, 0, 0, 0, 0]));
            this.socket.destroy();
        };

        if (this.server.rulesetValidator) {
            const result = await this.server.rulesetValidator(this, accept, reject);
            if (result === true) accept();
            else if (result === false) reject();
        } else {
            accept();
        }
    }

    private establish() {
        this.socket.removeListener("error", () => { });
        this.server.connectionHandler(this, (status: keyof typeof SocksStatus) => {
            const statusByte = SocksStatus[status];
            this.socket.write(Buffer.from([5, statusByte, 0, 1, 0, 0, 0, 0, 0, 0]));
            if (status !== "REQUEST_GRANTED") {
                this.socket.destroy();
            }
        });
        this.socket.resume();
    }
}

export type AuthHandler = (conn: SocksConnection, accept: () => void, reject: () => void) => Promise<boolean | void> | boolean | void;
export type RulesetValidator = (conn: SocksConnection, accept: () => void, reject: () => void) => Promise<boolean | void> | boolean | void;
export type ConnectionHandler = (conn: SocksConnection, reply: (status: keyof typeof SocksStatus) => void) => void;

export class Socks5Server {
    private server: net.Server;
    public supportedCommands = new Set<SocksCommand>([SocksCommand.CONNECT]);
    public authHandler?: AuthHandler;
    public rulesetValidator?: RulesetValidator;
    public connectionHandler: ConnectionHandler = defaultConnectionHandler;

    constructor() {
        this.server = net.createServer((socket) => {
            socket.setNoDelay();
            new SocksConnection(this, socket);
        });
    }

    listen(port: number, hostname?: string, callback?: () => void) {
        this.server.listen(port, hostname, callback);
        return this;
    }

    close(callback?: (err?: Error) => void) {
        this.server.close(callback);
        return this;
    }

    useDefaultConnectionHandler() {
        this.connectionHandler = defaultConnectionHandler;
        return this;
    }
}

function defaultConnectionHandler(conn: SocksConnection, reply: (status: keyof typeof SocksStatus) => void) {
    if (conn.command !== SocksCommand.CONNECT) {
        return reply("COMMAND_NOT_SUPPORTED");
    }

    if (conn.destPort === undefined) {
        return reply("GENERAL_FAILURE");
    }

    const target = net.createConnection({
        host: conn.destAddress,
        port: conn.destPort
    });

    target.setNoDelay();
    let done = false;

    target.on("error", (err: any) => {
        if (done) return;
        switch (err.code) {
            case "ENOTFOUND":
            case "ETIMEDOUT":
            case "EHOSTUNREACH":
                reply("HOST_UNREACHABLE");
                break;
            case "ENETUNREACH":
                reply("NETWORK_UNREACHABLE");
                break;
            case "ECONNREFUSED":
                reply("CONNECTION_REFUSED");
                break;
            default:
                reply("GENERAL_FAILURE");
        }
    });

    target.on("ready", () => {
        done = true;
        reply("REQUEST_GRANTED");
        conn.socket.pipe(target).pipe(conn.socket);
    });

    conn.socket.on("close", () => target.destroy());
}

export function createSocksServer(options?: SocksServerOptions): Socks5Server {
    const srv = new Socks5Server();
    if (options?.auth) {
        srv.authHandler = async (conn) => {
            return conn.username === options.auth?.username && conn.password === options.auth?.password;
        };
    }
    if (options?.port) {
        srv.listen(options.port, options.hostname);
    }
    return srv;
}
