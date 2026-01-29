/**
 * File: src/utils/platform/browserMocks.ts
 * Role: Provides basic browser-like globals (window, navigator) for Node.js environments.
 * This is sometimes required for third-party libraries that assume a browser-like environment.
 */

/**
 * Initializes global mocks for window, document, and navigator.
 */
export function initializeBrowserMocks(): void {
    const anyGlobal = global as any;
    if (typeof anyGlobal.window !== 'undefined') return;

    const mockDocument = {
        visibilityState: "visible",
        documentElement: { lang: "en" },
        addEventListener: () => { }
    };

    const mockWindow = {
        document: mockDocument,
        location: { href: "http://localhost", pathname: "/" },
        addEventListener: (event: string, cb: any) => {
            if (event === "beforeunload") {
                process.on("exit", () => {
                    if (typeof cb === 'function') {
                        cb({});
                    } else if (cb && typeof cb.handleEvent === 'function') {
                        cb.handleEvent({});
                    }
                });
            }
        },
        focus: () => { },
        innerHeight: 768,
        innerWidth: 1024,
    };

    const mockNavigator = {
        sendBeacon: () => true,
        userAgent: `Node.js/${process.version}`,
        language: "en-US",
    };

    anyGlobal.window = mockWindow;
    anyGlobal.document = mockDocument;
    anyGlobal.navigator = mockNavigator;
}

// Auto-initialize browser mocks on import.
initializeBrowserMocks();
