import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { McpServerManager } from "./McpServerManager.js";
import { getMcpClient, disconnectMcpServer } from "./McpClientManager.js";
import { fetchMcpTools, fetchMcpResources, fetchMcpPrompts } from "./McpDiscovery.js";
import { log } from "../logger/loggerService.js";

const logger = log("mcp-connection");

interface McpState {
    clients: any[];
    tools: any[];
    resources: Record<string, any[]>;
    commands: any[];
}

interface McpContextType extends McpState {
    reconnectMcpServer: (name: string) => Promise<any>;
    toggleMcpServer: (name: string) => Promise<void>;
    retryTimers: Map<string, NodeJS.Timeout>;
}

const McpContext = createContext<McpContextType | null>(null);

const RECONNECT_ATTEMPTS = 5;
const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30000;

export function McpManagerProvider({ children, dynamicMcpConfig, isStrictMcpConfig }: any) {
    const [state, setState] = useState<McpState>({
        clients: [],
        tools: [],
        resources: {},
        commands: []
    });

    const retryTimers = useRef<Map<string, NodeJS.Timeout>>(new Map());

    // Update state helper
    const updateClientState = useCallback((name: string, type: string, error?: string, clientData?: any) => {
        setState(prev => {
            const existingIndex = prev.clients.findIndex(c => c.name === name);
            const clients = [...prev.clients];

            const newClient = clientData || (existingIndex >= 0 ? clients[existingIndex] : { name, config: {} });

            // Merge updates
            const updatedClient = {
                ...newClient,
                type,
                error
            };

            if (existingIndex >= 0) {
                clients[existingIndex] = updatedClient;
            } else {
                clients.push(updatedClient);
            }

            return { ...prev, clients };
        });
    }, []);

    // Initial load
    useEffect(() => {
        let mounted = true;

        async function loadServers() {
            try {
                const { servers, errors } = await McpServerManager.getAllMcpServers();
                if (!mounted) return;

                // Initialize clients state
                const initialClients = Object.entries(servers).map(([name, config]) => ({
                    name,
                    config,
                    type: 'pending'
                }));

                setState(prev => ({ ...prev, clients: initialClients }));

                // Connect logic (could be batched)
                for (const client of initialClients) {
                    connectServer(client.name, client.config).catch(err => {
                        logger.error(`Failed to connect to ${client.name}:`, err);
                    });
                }
            } catch (err) {
                logger.error("Failed to load MCP servers:", err);
            }
        }

        loadServers();

        return () => {
            mounted = false;
            retryTimers.current.forEach(clearTimeout);
            retryTimers.current.clear();
        };
    }, []);

    const connectServer = useCallback(async (name: string, config: any, attempt = 1) => {
        if (!config) {
            const { servers } = await McpServerManager.getAllMcpServers();
            config = servers[name];
            if (!config) throw new Error(`Configuration for ${name} not found`);
        }

        updateClientState(name, 'connecting');

        try {
            const client = await getMcpClient(name, config);

            if (client.type === 'connected') {
                // Fetch data
                const tools = await fetchMcpTools(client);
                const resources = await fetchMcpResources(client);
                const prompts = await fetchMcpPrompts(client);

                setState(prev => ({
                    ...prev,
                    tools: [...prev.tools.filter(t => !t.name.startsWith(`mcp__${name}__`)), ...tools],
                    commands: [...prev.commands.filter(c => !c.name.startsWith(`mcp__${name}__`)), ...prompts],
                    resources: { ...prev.resources, [name]: resources }
                }));

                updateClientState(name, 'connected', undefined, client);

                // Set up auto-reconnect on close
                if (client.client) {
                    client.client.onclose = () => {
                        logger.info(`Server ${name} disconnected`);
                        updateClientState(name, 'disconnected');
                        // Trigger reconnect if strictly enabled or dynamic
                        // Simple auto-reconnect logic
                        scheduleReconnect(name, config, 1);
                    };
                }

                return client;
            } else {
                throw new Error(`Client type is ${client.type}`);
            }
        } catch (err: any) {
            updateClientState(name, 'failed', err.message);
            throw err;
        }
    }, [updateClientState]);

    const scheduleReconnect = useCallback((name: string, config: any, attempt: number) => {
        if (attempt > RECONNECT_ATTEMPTS) {
            logger.warn(`Max reconnect attempts reached for ${name}`);
            updateClientState(name, 'failed', "Max reconnect attempts reached");
            return;
        }

        const delay = Math.min(INITIAL_BACKOFF_MS * Math.pow(2, attempt - 1), MAX_BACKOFF_MS);
        logger.info(`Scheduling reconnect for ${name} in ${delay}ms (attempt ${attempt})`);

        updateClientState(name, 'connecting', `Reconnecting in ${Math.round(delay / 1000)}s...`);

        if (retryTimers.current.has(name)) {
            clearTimeout(retryTimers.current.get(name)!);
        }

        const timer = setTimeout(() => {
            connectServer(name, config, attempt).catch(err => {
                scheduleReconnect(name, config, attempt + 1);
            });
        }, delay);

        retryTimers.current.set(name, timer);

    }, [connectServer, updateClientState]);

    const reconnectMcpServer = useCallback(async (name: string) => {
        const clientState = state.clients.find(c => c.name === name);
        if (!clientState) throw new Error(`Server ${name} not found`);

        if (retryTimers.current.has(name)) {
            clearTimeout(retryTimers.current.get(name)!);
            retryTimers.current.delete(name);
        }

        await disconnectMcpServer(name, clientState.config);
        updateClientState(name, 'pending');
        return connectServer(name, clientState.config, 1);
    }, [state.clients, connectServer, updateClientState]);

    const toggleMcpServer = useCallback(async (name: string) => {
        const clientState = state.clients.find(c => c.name === name);
        if (!clientState) return;

        if (clientState.type === 'disabled') {
            await reconnectMcpServer(name);
        } else {
            if (retryTimers.current.has(name)) {
                clearTimeout(retryTimers.current.get(name)!);
                retryTimers.current.delete(name);
            }
            await disconnectMcpServer(name, clientState.config);
            updateClientState(name, 'disabled');
        }
    }, [state.clients, reconnectMcpServer, updateClientState]);

    return (
        <McpContext.Provider value={{
            ...state,
            reconnectMcpServer,
            toggleMcpServer,
            retryTimers: retryTimers.current
        }}>
            {children}
        </McpContext.Provider>
    );
}

export const useReconnectMcpServer = () => {
    const ctx = useContext(McpContext);
    if (!ctx) throw new Error("useReconnectMcpServer must be used within McpManagerProvider");
    return ctx.reconnectMcpServer;
};

export const useToggleMcpServer = () => {
    const ctx = useContext(McpContext);
    if (!ctx) throw new Error("useToggleMcpServer must be used within McpManagerProvider");
    return ctx.toggleMcpServer;
};

// --- Token Parsing (CDA) ---

export function parseActiveToken(input: string, cursorPosition: number, isStrict: boolean = false) {
    if (!input) return null;
    const beforeCursor = input.substring(0, cursorPosition);

    // Check for quoted token
    if (isStrict) {
        const matchQuoted = beforeCursor.match(/@"([^"]*)"?$/);
        if (matchQuoted && matchQuoted.index !== undefined) {
            const afterCursor = input.substring(cursorPosition).match(/^[^"]*"?/);
            const suffix = afterCursor ? afterCursor[0] : "";
            return {
                token: matchQuoted[0] + suffix,
                startPos: matchQuoted.index,
                isQuoted: true
            };
        }
    }

    const regex = isStrict
        ? /(@[a-zA-Z0-9_\-./\\()[\]~]*|[a-zA-Z0-9_\-./\\()[\]~]+)$/
        : /[a-zA-Z0-9_\-./\\()[\]~]+$/;

    const match = beforeCursor.match(regex);
    if (!match || match.index === undefined) return null;

    const afterCursor = input.substring(cursorPosition).match(/^[a-zA-Z0-9_\-./\\()[\]~]+/);
    const suffix = afterCursor ? afterCursor[0] : "";

    return {
        token: match[0] + suffix,
        startPos: match.index,
        isQuoted: false
    };
}

export function formatCompletionResult(
    displayText: string,
    mode: string,
    hasAtPrefix: boolean,
    needsQuotes: boolean,
    isQuoted: boolean,
    isComplete: boolean
) {
    const suffix = isComplete ? " " : "";
    if (isQuoted || needsQuotes) {
        return mode === "bash" ? `"${displayText}"${suffix}` : `@"${displayText}"${suffix}`;
    } else if (hasAtPrefix) {
        return mode === "bash" ? `${displayText}${suffix}` : `@${displayText}${suffix}`;
    } else {
        return displayText + suffix;
    }
}
