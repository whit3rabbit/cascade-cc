
// Logic from chunk_574.ts (Export & Model Commands)

import React, { useState, useEffect } from "react";
import { Box, Text } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { Shortcut } from "../../components/shared/Shortcut.js";
import fs from "node:fs";
import path from "node:path";

// --- Export Utilities ---
export function formatDateForFilename(date: Date): string {
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}

export function getConversationPreview(messages: any[]): string {
    const userMsg = messages.find(m => m.type === "user");
    if (!userMsg) return "";
    let content = userMsg.message?.content || "";
    if (Array.isArray(content)) {
        const textPart = content.find(p => p.type === "text");
        content = textPart?.text || "";
    }
    const firstLine = String(content).trim().split("\n")[0] || "";
    return firstLine.length > 50 ? firstLine.substring(0, 50) + "..." : firstLine;
}

export function slugify(text: string): string {
    return text.toLowerCase()
        .replace(/[^a-z0-9\s-]/g, "")
        .replace(/\s+/g, "-")
        .replace(/-+/g, "-")
        .replace(/^-|-$/g, "");
}

// --- Export View ($I9) ---
export function ExportView({ content, defaultFilename, onDone }: any) {
    const [filename, setFilename] = useState(defaultFilename);
    const [status, setStatus] = useState<any>(null);

    const handleSave = () => {
        try {
            const fullPath = path.join(process.cwd(), filename);
            fs.writeFileSync(fullPath, content, "utf-8");
            onDone({ success: true, message: `Conversation exported to: ${filename}` });
        } catch (err: any) {
            setStatus({ success: false, message: `Failed to export: ${err.message}` });
        }
    };

    return (
        <Box flexDirection="column" padding={1} borderStyle="round">
            <Text bold>Export Conversation</Text>
            <Box marginTop={1}>
                <Text>Enter filename: </Text>
                <Text color="suggestion">{filename}</Text>
            </Box>
            {status && <Box marginTop={1}><Text color={status.success ? "success" : "error"}>{status.message}</Text></Box>}
            <Box marginTop={1} gap={2}>
                <Shortcut shortcut="Enter" action="save" />
                <Shortcut shortcut="Esc" action="cancel" />
            </Box>
        </Box>
    );
}

// --- Model Validator (MI9) ---
const knownModels = new Set(["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"]);

export async function validateModelExists(modelName: string): Promise<{ valid: boolean; error?: string }> {
    const normalized = modelName.toLowerCase();
    if (knownModels.has(normalized)) return { valid: true };

    // In real implementation, this would call Antropic API to check
    return { valid: false, error: `Model '${modelName}' not found or not accessible.` };
}

// --- Model Selection Components (UZ7/LZ7) ---
export function ModelStatusView({ currentModel, sessionOverride }: any) {
    if (sessionOverride) {
        return (
            <Text>
                Current model: <Text bold>{sessionOverride}</Text> (session override)
                {"\n"}Base model: {currentModel}
            </Text>
        );
    }
    return <Text>Current model: <Text bold>{currentModel}</Text></Text>;
}

