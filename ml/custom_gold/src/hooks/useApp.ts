import { useState, useCallback } from 'react';

interface AppState {
    notifications: any[];
    debug: boolean;
}

export function useApp() {
    const [state, setState] = useState<AppState>({
        notifications: [],
        debug: false
    });

    const addNotification = useCallback((notification: any) => {
        setState((prev: AppState) => ({ ...prev, notifications: [...prev.notifications, notification] }));
    }, []);

    return {
        state,
        dispatch: (action: any) => { },
        addNotification
    };
}
