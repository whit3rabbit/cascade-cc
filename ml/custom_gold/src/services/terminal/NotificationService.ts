
import { useState, useCallback, useEffect } from "react";

export type NotificationPriority = "immediate" | "high" | "medium" | "low";

export interface Notification {
    key: string;
    text?: string | null;
    priority: NotificationPriority;
    timeoutMs?: number;
    invalidates?: string[];
    jsx?: React.ReactNode;
    color?: string;
}

const DEFAULT_TIMEOUT = 8000;

const PRIORITY_MAP: Record<NotificationPriority, number> = {
    immediate: 0,
    high: 1,
    medium: 2,
    low: 3
};

/**
 * Hook for managing terminal notifications.
 * Derived from chunk_583.ts (l8)
 */
export function useNotifications() {
    const [state, setState] = useState<{
        queue: Notification[];
        current: Notification | null;
    }>({
        queue: [],
        current: null
    });

    const [timeoutId, setTimeoutId] = useState<NodeJS.Timeout | null>(null);

    const processQueue = useCallback(() => {
        setState((prev) => {
            if (prev.current !== null || prev.queue.length === 0) return prev;

            const next = [...prev.queue].sort((a, b) =>
                (PRIORITY_MAP[a.priority] ?? 999) - (PRIORITY_MAP[b.priority] ?? 999)
            )[0];

            if (!next) return prev;

            const timer = setTimeout(() => {
                setTimeoutId(null);
                setState((curr) => {
                    if (curr.current?.key !== next.key) return curr;
                    return {
                        ...curr,
                        current: null
                    };
                });
                processQueue();
            }, next.timeoutMs ?? DEFAULT_TIMEOUT);

            setTimeoutId(timer);

            return {
                ...prev,
                queue: prev.queue.filter(n => n !== next),
                current: next
            };
        });
    }, []);

    const addNotification = useCallback((notification: Notification) => {
        if (notification.priority === "immediate") {
            if (timeoutId) {
                clearTimeout(timeoutId);
                setTimeoutId(null);
            }

            const timer = setTimeout(() => {
                setTimeoutId(null);
                setState((prev) => {
                    if (prev.current?.key !== notification.key) return prev;
                    return {
                        ...prev,
                        current: null
                    };
                });
                processQueue();
            }, notification.timeoutMs ?? DEFAULT_TIMEOUT);

            setTimeoutId(timer);

            setState((prev) => ({
                ...prev,
                current: notification,
                queue: [
                    ...(prev.current ? [prev.current] : []),
                    ...prev.queue
                ].filter(n => n.priority !== "immediate" && !notification.invalidates?.includes(n.key))
            }));
            return;
        }

        setState((prev) => {
            const alreadyQueued = prev.queue.some(n => n.key === notification.key) || prev.current?.key === notification.key;
            if (alreadyQueued) return prev;

            return {
                ...prev,
                queue: [...prev.queue.filter(n => n.priority !== "immediate" && !notification.invalidates?.includes(n.key)), notification]
            };
        });
        processQueue();
    }, [timeoutId, processQueue]);

    const removeNotification = useCallback((key: string) => {
        setState((prev) => {
            const isCurrent = prev.current?.key === key;
            const inQueue = prev.queue.some(n => n.key === key);

            if (!isCurrent && !inQueue) return prev;

            if (isCurrent && timeoutId) {
                clearTimeout(timeoutId);
                setTimeoutId(null);
            }

            return {
                ...prev,
                current: isCurrent ? null : prev.current,
                queue: prev.queue.filter(n => n.key !== key)
            };
        });
        processQueue();
    }, [timeoutId, processQueue]);

    useEffect(() => {
        if (state.queue.length > 0) processQueue();
        return () => {
            if (timeoutId) clearTimeout(timeoutId);
        };
    }, [processQueue]);

    return {
        notifications: state,
        addNotification,
        removeNotification
    };
}
