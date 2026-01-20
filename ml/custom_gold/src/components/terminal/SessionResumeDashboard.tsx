
// Logic from chunk_557.ts (Session Resume Dashboard & Search)

import React, { useState, useEffect, useMemo, useCallback } from "react";
import { Box, Text, useInput } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { PermissionSelect } from "../permissions/PermissionComponents.js";
import { SessionTreeSelector, SessionTranscriptPreview } from "./SessionResumeView.js";

// --- Search text extraction (w77 / q77) ---
export function getMessageText(msg: any): string {
    if (msg.type !== "user" && msg.type !== "assistant") return "";
    const content = msg.message?.content;
    if (!content) return "";
    if (typeof content === "string") return content;
    if (Array.isArray(content)) {
        return content.map((block: any) => {
            if (typeof block === "string") return block;
            if (block.text && typeof block.text === "string") return block.text;
            return "";
        }).filter(Boolean).join(" ");
    }
    return "";
}

export function getSessionSearchIndex(session: any): string {
    const messagesText = (session.messages || [])
        .slice(-10) // Just last 10 for indexing
        .map(getMessageText)
        .join(" ");

    return [
        session.customTitle,
        session.summary,
        session.gitBranch,
        session.tag,
        messagesText
    ].filter(Boolean).join(" ").toLowerCase();
}

// --- Session Grouping (N77) ---
export function groupSessionsByProject(sessions: any[]): Map<string, any[]> {
    const groups = new Map<string, any[]>();
    for (const session of sessions) {
        const path = session.projectPath || "Unknown";
        const list = groups.get(path) || [];
        list.push(session);
        groups.set(path, list);
    }
    return groups;
}

// --- Dashboard Component (xH1) ---
export function SessionResumeDashboard({
    sessions,
    onSelect,
    onCancel,
    onRename
}: any) {
    const [status, setStatus] = useState<"list" | "searching" | "rename">("list");
    const [searchQuery, setSearchQuery] = useState("");
    const [focusedSession, setFocusedSession] = useState<any>(null);

    const filteredSessions = useMemo(() => {
        if (!searchQuery) return sessions;
        const query = searchQuery.toLowerCase();
        return sessions.filter((s: any) => getSessionSearchIndex(s).includes(query));
    }, [sessions, searchQuery]);

    useInput((input, key) => {
        if (status === "list") {
            if (key.escape) onCancel();
            if (input === "r" && focusedSession) setStatus("rename");
        } else if (status === "rename") {
            // Handled by TextInput
        }
    });

    const renderHeader = () => {
        if (status === "searching") {
            return (
                <Box paddingLeft={1}>
                    <Text color="suggestion">{figures.info} </Text>
                    <Text>Searching…</Text>
                </Box>
            );
        }
        if (searchQuery && filteredSessions.length > 0) {
            return (
                <Box paddingLeft={1} marginBottom={1}>
                    <Text dimColor italic>Claude found these results:</Text>
                </Box>
            );
        }
        return null;
    };

    return (
        <Box flexDirection="column" gap={1}>
            {renderHeader()}

            <SessionTreeSelector
                sessions={filteredSessions}
                onSelect={onSelect}
                onFocus={(s: any) => setFocusedSession(s)}
                onCancel={onCancel}
                currentSelectionId={focusedSession?.id}
            />

            <Box paddingLeft={2}>
                <Text dimColor>
                    {status === "rename" ? "Enter to save · Esc to cancel" :
                        status === "searching" ? "Esc to cancel" :
                            "R to rename · Type to search · Esc to exit"}
                </Text>
            </Box>
        </Box>
    );
}

// --- Clipboard Utils (_s) ---
export async function copyToClipboard(text: string): Promise<boolean> {
    // Mocking clipboard interaction
    console.log(`Copied to clipboard: ${text}`);
    return true;
}

export function getClipboardErrorMessage(): string {
    const platform = process.platform;
    if (platform === "darwin") return "Failed to copy to clipboard. Make sure 'pbcopy' is available.";
    if (platform === "win32") return "Failed to copy to clipboard. Make sure 'clip' is available.";
    return "Failed to copy to clipboard. Make sure 'xclip' or 'wl-copy' is installed.";
}
