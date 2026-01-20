
// Logic from chunk_556.ts (Session Resume & Transcript UI)

import React, { useState, useMemo, useCallback, useRef, useEffect } from "react";
import { Box, Text, useInput } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { PermissionSelect } from "../permissions/PermissionComponents.js";
import { Shortcut } from "../shared/Shortcut.js";

// --- Session Tab Bar (uZ9) ---
export function SessionTabBar({
    tabs,
    selectedIndex,
    availableWidth,
    showAllProjects = false
}: any) {
    const title = showAllProjects ? "Resume (All Projects)" : "Resume";

    return (
        <Box flexDirection="row" gap={1}>
            <Text color="suggestion">{title}</Text>
            {tabs.map((tab: string, i: number) => {
                const isActive = i === selectedIndex;
                const label = tab === "All" ? tab : `#${tab.slice(0, 10)}`;
                return (
                    <Box
                        key={tab}
                        backgroundColor={isActive ? "suggestion" : undefined}
                        paddingX={1}
                    >
                        <Text
                            color={isActive ? "inverseText" : undefined}
                            bold={isActive}
                        >
                            {label}
                        </Text>
                    </Box>
                );
            })}
            <Text dimColor>(tab to cycle)</Text>
        </Box>
    );
}

// --- Session Tree Selector (uN0) ---
export function SessionTreeSelector({
    sessions,
    onSelect,
    onFocus,
    onCancel,
    currentSelectionId
}: any) {
    const [expandedIds, setExpandedIds] = useState(new Set<string>());

    const sessionList = useMemo(() => {
        const flattened: any[] = [];
        const process = (items: any[], depth = 0, parentId?: string) => {
            for (const session of items) {
                const isExpanded = expandedIds.has(session.id);
                flattened.push({
                    node: session,
                    depth,
                    isExpanded,
                    hasChildren: !!session.children?.length,
                    parentId
                });
                if (isExpanded && session.children) {
                    process(session.children, depth + 1, session.id);
                }
            }
        };
        process(sessions);
        return flattened;
    }, [sessions, expandedIds]);

    const options = useMemo(() => {
        return sessionList.map((item) => {
            let prefix = "";
            if (item.hasChildren) {
                prefix = item.isExpanded ? "▼ " : "▶ ";
            } else if (item.depth > 0) {
                prefix = "  ▸ ";
            }
            return {
                label: prefix + item.node.label,
                description: item.node.description,
                value: item.node.id
            };
        });
    }, [sessionList]);

    useInput((input, key) => {
        const currentItem = sessionList.find(s => s.node.id === currentSelectionId);
        if (!currentItem) return;

        if (key.rightArrow && currentItem.hasChildren && !currentItem.isExpanded) {
            setExpandedIds(prev => new Set([...prev, currentItem.node.id]));
        } else if (key.leftArrow) {
            if (currentItem.hasChildren && currentItem.isExpanded) {
                setExpandedIds(prev => {
                    const next = new Set(prev);
                    next.delete(currentItem.node.id);
                    return next;
                });
            } else if (currentItem.parentId) {
                onFocus(currentItem.parentId);
            }
        }
    });

    return (
        <Box flexDirection="column" paddingX={1} borderStyle="round">
            <Box marginBottom={1}>
                <Text bold>Sessions</Text>
                <Text dimColor> ({sessionList.length} items)</Text>
            </Box>
            <PermissionSelect
                options={options}
                onChange={(id) => onSelect(sessionList.find(s => s.node.id === id)?.node)}
                onFocus={onFocus}
                onCancel={onCancel}
                defaultFocusValue={currentSelectionId}
            />
            <Box marginLeft={3}>
                <Text dimColor>Esc to go back</Text>
            </Box>
        </Box>
    );
}

// --- Session Transcript View (yZ9) ---
// Note: This would normally use TranscriptView (Ds) which is more complex
export function SessionTranscriptPreview({ session, onExit, onResume }: any) {
    return (
        <Box flexDirection="column">
            <Box flexDirection="column" padding={1} borderStyle="single">
                <Text bold>Transcript for {session.id}</Text>
                <Box flexDirection="column" marginY={1}>
                    {session.messages?.map((msg: any, i: number) => (
                        <Box key={i} marginBottom={1}>
                            <Text bold color={msg.role === "user" ? "cyan" : "magenta"}>
                                {msg.role === "user" ? "User" : "Claude"}:
                            </Text>
                            <Text>{msg.content}</Text>
                        </Box>
                    ))}
                </Box>
            </Box>
            <Box
                flexDirection="column"
                borderStyle="single"
                borderTop={true}
                borderBottom={false}
                borderLeft={false}
                borderRight={false}
                paddingLeft={2}
            >
                <Text>{session.modified} · {session.messages?.length} messages</Text>
                <Box flexDirection="row" gap={2}>
                    <Shortcut shortcut="Enter" action="resume" />
                    <Shortcut shortcut="Esc" action="cancel" />
                </Box>
            </Box>
        </Box>
    );
}
