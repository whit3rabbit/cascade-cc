/**
 * File: src/services/terminal/NotificationService.tsx
 * Role: Context provider and hook for managing ephemeral notifications in the terminal UI.
 */

import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';

export interface Notification {
    key?: string;
    text: string;
    type?: 'info' | 'success' | 'warn' | 'error';
    timeoutMs?: number;
}

interface NotificationContextValue {
    notifications: Notification[];
    addNotification: (notif: Notification) => string;
    removeNotification: (key: string) => void;
    clearNotifications: () => void;
}

const NotificationContext = createContext<NotificationContextValue | null>(null);

/**
 * Provider for managing a stack of ephemeral terminal notifications.
 */
export function NotificationProvider({ children }: { children: ReactNode }) {
    const [notifications, setNotifications] = useState<Notification[]>([]);

    const addNotification = useCallback((notif: Notification) => {
        const key = notif.key || Math.random().toString(36).substring(7);
        setNotifications((prev) => [...prev, { ...notif, key }]);

        if (notif.timeoutMs) {
            setTimeout(() => {
                removeNotification(key);
            }, notif.timeoutMs);
        }
        return key;
    }, []);

    const removeNotification = useCallback((key: string) => {
        setNotifications((prev) => prev.filter((n) => n.key !== key));
    }, []);

    const clearNotifications = useCallback(() => {
        setNotifications([]);
    }, []);

    return (
        <NotificationContext.Provider value={{
            notifications,
            addNotification,
            removeNotification,
            clearNotifications
        }}>
            {children}
        </NotificationContext.Provider>
    );
}

/**
 * Hook to access and manipulate terminal notifications.
 */
export function useNotifications() {
    const context = useContext(NotificationContext);
    if (!context) {
        // Fallback for cases where Provider is not present (e.g., tests or early init)
        return {
            notifications: [],
            addNotification: (n: Notification) => {
                console.log(`[Notification] ${n.text}`);
                return "stub";
            },
            removeNotification: () => { },
            clearNotifications: () => { }
        };
    }
    return context;
}
