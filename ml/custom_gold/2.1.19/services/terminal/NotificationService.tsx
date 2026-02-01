/**
 * File: src/services/terminal/NotificationService.tsx
 * Role: Context provider and hook for managing ephemeral notifications in the terminal UI.
 */

import React, { createContext, useContext, useState, useCallback, ReactNode, useEffect } from 'react';

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

type NotificationListener = (n: Notification) => void;
class NotificationQueue {
    private listeners: Set<NotificationListener> = new Set();

    subscribe(l: NotificationListener) {
        this.listeners.add(l);
        return () => { this.listeners.delete(l); };
    }

    add(n: Notification) {
        this.listeners.forEach(l => l(n));
    }
}

export const notificationQueue = new NotificationQueue();
const NotificationContext = createContext<NotificationContextValue | null>(null);

/**
 * Provider for managing a stack of ephemeral terminal notifications.
 */
export function NotificationProvider({ children }: { children: ReactNode }) {
    const [notifications, setNotifications] = useState<Notification[]>([]);

    const removeNotification = useCallback((key: string) => {
        setNotifications((prev) => prev.filter((n) => n.key !== key));
    }, []);

    const addNotification = useCallback((notif: Notification) => {
        const key = notif.key || Math.random().toString(36).substring(7);
        setNotifications((prev) => [...prev, { ...notif, key }]);

        if (notif.timeoutMs) {
            setTimeout(() => {
                removeNotification(key);
            }, notif.timeoutMs);
        }
        return key;
    }, [removeNotification]);

    useEffect(() => {
        return notificationQueue.subscribe((n) => {
            addNotification(n);
        });
    }, [addNotification]);

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
