
import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import { getLastApiRequest, getStreamingContent } from '../claude/claudeApi.js';
import { log, logError } from '../logger/loggerService.js';
import { logTelemetryEvent } from '../telemetry/telemetryInit.js';
import { getGlobalState } from '../session/globalState.js';

// Logic from chunk_517.ts (Extra Features)

/**
 * Hook to warm up the context cache by replaying the last request periodically.
 * Based on Ne2 from chunk_517.ts
 */
export function useCacheWarming(isActive: boolean, idleThresholdMs = 240000) {
    const abortControllerRef = useRef<AbortController | null>(null);

    useEffect(() => {
        // Mock config check (from $07)
        const config = {
            enabled: false, // Defaulting to false as per chunk_517
            idleThresholdMs: idleThresholdMs,
            subsequentWarmupIntervalMs: 300000,
            maxRequests: 1
        };

        // If explicitly enabled via env or settings, we would flip this.
        // For now, mirroring the deobfuscated default logic which seems to check settings.
        const globalState = getGlobalState();
        // Assuming there's a setting for this, e.g. globalState.settings.cacheWarming
        if (process.env.CLAUDE_CACHE_WARMING_ENABLED === 'true') {
            config.enabled = true;
        }

        if (!config.enabled) return;
        if (isActive) {
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
                abortControllerRef.current = null;
            }
            return;
        }

        let requestCount = 0;
        let timeout: NodeJS.Timeout | null = null;

        const performWarmup = async () => {
            const lastRequest = getLastApiRequest();
            if (!lastRequest) {
                log('loops', "Cache warming: No previous API request to replay");
                return;
            }

            if (abortControllerRef.current) abortControllerRef.current.abort();
            abortControllerRef.current = new AbortController();

            try {
                log('loops', `Cache warming: Sending request ${requestCount + 1}/${config.maxRequests}`);

                // Construct warmup request: replay history but ask for short response
                const warmupMessages = [...lastRequest.messages, { role: "user", content: 'Reply with just "OK"' }];

                const stream = getStreamingContent({
                    ...lastRequest,
                    messages: warmupMessages,
                    max_tokens: 10,
                    signal: abortControllerRef.current.signal
                });

                let finalUsage: any = {};
                for await (const event of stream) {
                    if (event.type === 'message_stop') {
                        // usage usually comes here or in message_delta
                    }
                    if (event.usage) {
                        finalUsage = event.usage;
                    }
                }

                log('loops', "Cache warming: Request completed");
                logTelemetryEvent("tengu_cache_warming_request", {
                    warmup_number: requestCount + 1,
                    cache_read_tokens: finalUsage.cache_read_input_tokens ?? 0,
                    cache_creation_tokens: finalUsage.cache_creation_input_tokens ?? 0,
                    input_tokens: finalUsage.input_tokens,
                    output_tokens: finalUsage.output_tokens
                });

                requestCount++;
                if (requestCount < config.maxRequests) {
                    scheduleWarmup(config.subsequentWarmupIntervalMs);
                }

            } catch (err: any) {
                if (err.name !== 'AbortError') {
                    logError('loops', err, "Cache warming failed");
                }
            } finally {
                abortControllerRef.current = null;
            }
        };

        const scheduleWarmup = (delay: number) => {
            timeout = setTimeout(performWarmup, delay);
        };

        scheduleWarmup(config.idleThresholdMs);

        return () => {
            if (timeout) clearTimeout(timeout);
            if (abortControllerRef.current) {
                abortControllerRef.current.abort();
                abortControllerRef.current = null;
            }
        };
    }, [isActive, idleThresholdMs]);
}

/**
 * Hook to manage feedback survey state.
 * Based on xe2 from chunk_517.ts
 */
export function useFeedbackSurvey() {
    // Simplified logic as we lack the full survey configuration infrastructure
    const [state, setState] = useState<"closed" | "open" | "thanks">("closed");

    // In a real implementation, this would check turns, time, and global state
    // to determine when to set state to "open".

    return {
        state,
        setState,
        handleSelect: (value: any) => {
            logTelemetryEvent("tengu_feedback_survey_event", {
                event_type: "responded",
                response: value
            });
            setState("thanks");
            setTimeout(() => setState("closed"), 3000);
        }
    };
}


/**
 * Browser Tools Schemas
 * Based on xDA from chunk_517.ts
 */
export const browserTools = [
    {
        name: "javascript_tool",
        description: "Execute JavaScript code in the context of the current page. The code runs in the page's context and can interact with the DOM, window object, and page variables. Returns the result of the last expression or any thrown errors. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.",
        inputSchema: {
            type: "object",
            properties: {
                action: {
                    type: "string",
                    description: "Must be set to 'javascript_exec'"
                },
                text: {
                    type: "string",
                    description: "The JavaScript code to execute. The code will be evaluated in the page context. The result of the last expression will be returned automatically. Do NOT use 'return' statements - just write the expression you want to evaluate (e.g., 'window.myData.value' not 'return window.myData.value'). You can access and modify the DOM, call page functions, and interact with page variables."
                },
                tabId: {
                    type: "number",
                    description: "Tab ID to execute the code in. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID."
                }
            },
            required: ["action", "text", "tabId"]
        }
    }, {
        name: "read_page",
        description: "Get an accessibility tree representation of elements on the page. By default returns all elements including non-visible ones. Output is limited to 50000 characters. If the output exceeds this limit, you will receive an error asking you to specify a smaller depth or focus on a specific element using ref_id. Optionally filter for only interactive elements. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.",
        inputSchema: {
            type: "object",
            properties: {
                filter: {
                    type: "string",
                    enum: ["interactive", "all"],
                    description: 'Filter elements: "interactive" for buttons/links/inputs only, "all" for all elements including non-visible ones (default: all elements)'
                },
                tabId: {
                    type: "number",
                    description: "Tab ID to read from. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID."
                },
                depth: {
                    type: "number",
                    description: "Maximum depth of the tree to traverse (default: 15). Use a smaller depth if output is too large."
                },
                ref_id: {
                    type: "string",
                    description: "Reference ID of a parent element to read. Will return the specified element and all its children. Use this to focus on a specific part of the page when output is too large."
                }
            },
            required: ["tabId"]
        }
    }, {
        name: "find",
        description: `Find elements on the page using natural language. Can search for elements by their purpose (e.g., "search bar", "login button") or by text content (e.g., "organic mango product"). Returns up to 20 matching elements with references that can be used with other tools. If more than 20 matches exist, you'll be notified to use a more specific query. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.`,
        inputSchema: {
            type: "object",
            properties: {
                query: {
                    type: "string",
                    description: 'Natural language description of what to find (e.g., "search bar", "add to cart button", "product title containing organic")'
                },
                tabId: {
                    type: "number",
                    description: "Tab ID to search in. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID."
                }
            },
            required: ["query", "tabId"]
        }
    }, {
        name: "form_input",
        description: "Set values in form elements using element reference ID from the read_page tool. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.",
        inputSchema: {
            type: "object",
            properties: {
                ref: {
                    type: "string",
                    description: 'Element reference ID from the read_page tool (e.g., "ref_1", "ref_2")'
                },
                value: {
                    type: ["string", "boolean", "number"],
                    description: "The value to set. For checkboxes use boolean, for selects use option value or text, for other inputs use appropriate string/number"
                },
                tabId: {
                    type: "number",
                    description: "Tab ID to set form value in. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID."
                }
            },
            required: ["ref", "value", "tabId"]
        }
    }, {
        name: "computer",
        description: `Use a mouse and keyboard to interact with a web browser, and take screenshots. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.
* Whenever you intend to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.
* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your click location so that the tip of the cursor visually falls on the element that you want to click.
* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.`,
        inputSchema: {
            type: "object",
            properties: {
                action: {
                    type: "string",
                    enum: ["left_click", "right_click", "type", "screenshot", "wait", "scroll", "key", "left_click_drag", "double_click", "triple_click", "zoom", "scroll_to", "hover"],
                    description: "The action to perform:\n* `left_click`: Click the left mouse button at the specified coordinates.\n* `right_click`: Click the right mouse button at the specified coordinates to open context menus.\n* `double_click`: Double-click the left mouse button at the specified coordinates.\n* `triple_click`: Triple-click the left mouse button at the specified coordinates.\n* `type`: Type a string of text.\n* `screenshot`: Take a screenshot of the screen.\n* `wait`: Wait for a specified number of seconds.\n* `scroll`: Scroll up, down, left, or right at the specified coordinates.\n* `key`: Press a specific keyboard key.\n* `left_click_drag`: Drag from start_coordinate to coordinate.\n* `zoom`: Take a screenshot of a specific region for closer inspection.\n* `scroll_to`: Scroll an element into view using its element reference ID from read_page or find tools.\n* `hover`: Move the mouse cursor to the specified coordinates or element without clicking. Useful for revealing tooltips, dropdown menus, or triggering hover states."
                },
                coordinate: {
                    type: "array",
                    items: {
                        type: "number"
                    },
                    minItems: 2,
                    maxItems: 2,
                    description: "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates. Required for `left_click`, `right_click`, `double_click`, `triple_click`, and `scroll`. For `left_click_drag`, this is the end position."
                },
                text: {
                    type: "string",
                    description: 'The text to type (for `type` action) or the key(s) to press (for `key` action). For `key` action: Provide space-separated keys (e.g., "Backspace Backspace Delete"). Supports keyboard shortcuts using the platform\'s modifier key (use "cmd" on Mac, "ctrl" on Windows/Linux, e.g., "cmd+a" or "ctrl+a" for select all).'
                },
                duration: {
                    type: "number",
                    minimum: 0,
                    maximum: 30,
                    description: "The number of seconds to wait. Required for `wait`. Maximum 30 seconds."
                },
                scroll_direction: {
                    type: "string",
                    enum: ["up", "down", "left", "right"],
                    description: "The direction to scroll. Required for `scroll`."
                },
                scroll_amount: {
                    type: "number",
                    minimum: 1,
                    maximum: 10,
                    description: "The number of scroll wheel ticks. Optional for `scroll`, defaults to 3."
                },
                start_coordinate: {
                    type: "array",
                    items: {
                        type: "number"
                    },
                    minItems: 2,
                    maxItems: 2,
                    description: "(x, y): The starting coordinates for `left_click_drag`."
                },
                region: {
                    type: "array",
                    items: {
                        type: "number"
                    },
                    minItems: 4,
                    maxItems: 4,
                    description: "(x0, y0, x1, y1): The rectangular region to capture for `zoom`. Coordinates define a rectangle from top-left (x0, y0) to bottom-right (x1, y1) in pixels from the viewport origin. Required for `zoom` action. Useful for inspecting small UI elements like icons, buttons, or text."
                },
                repeat: {
                    type: "number",
                    minimum: 1,
                    maximum: 100,
                    description: "Number of times to repeat the key sequence. Only applicable for `key` action. Must be a positive integer between 1 and 100. Default is 1. Useful for navigation tasks like pressing arrow keys multiple times."
                },
                ref: {
                    type: "string",
                    description: 'Element reference ID from read_page or find tools (e.g., "ref_1", "ref_2"). Required for `scroll_to` action. Can be used as alternative to `coordinate` for click actions.'
                },
                modifiers: {
                    type: "string",
                    description: 'Modifier keys for click actions. Supports: "ctrl", "shift", "alt", "cmd" (or "meta"), "win" (or "windows"). Can be combined with "+" (e.g., "ctrl+shift", "cmd+alt"). Optional.'
                },
                tabId: {
                    type: "number",
                    description: "Tab ID to execute the action on. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID."
                }
            },
            required: ["action", "tabId"]
        }
    }, {
        name: "navigate",
        description: "Navigate to a URL, or go forward/back in browser history. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.",
        inputSchema: {
            type: "object",
            properties: {
                url: {
                    type: "string",
                    description: 'The URL to navigate to. Can be provided with or without protocol (defaults to https://). Use "forward" to go forward in history or "back" to go back in history.'
                },
                tabId: {
                    type: "number",
                    description: "Tab ID to navigate. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID."
                }
            },
            required: ["url", "tabId"]
        }
    }, {
        name: "resize_window",
        description: "Resize the current browser window to specified dimensions. Useful for testing responsive designs or setting up specific screen sizes. If you don't have a valid tab ID, use tabs_context_mcp first to get available tabs.",
        inputSchema: {
            type: "object",
            properties: {
                width: {
                    type: "number",
                    description: "Target window width in pixels"
                },
                height: {
                    type: "number",
                    description: "Target window height in pixels"
                },
                tabId: {
                    type: "number",
                    description: "Tab ID to get the window for. Must be a tab in the current group. Use tabs_context_mcp first if you don't have a valid tab ID."
                }
            },
            required: ["width", "height", "tabId"]
        }
    }, {
        name: "gif_creator",
        description: "Manage GIF recording and export for browser automation sessions. Control when to start/stop recording browser actions (clicks, scrolls, navigation), then export as an animated GIF with visual overlays (click indicators, action labels, progress bar, watermark). All operations are scoped to the tab's group. When starting recording, take a screenshot immediately after to capture the initial state as the first frame. When stopping recording, take a screenshot immediately before to capture the final state as the last frame. For export, either provide 'coordinate' to drag/drop upload to a page element, or set 'download: true' to download the GIF.",
        inputSchema: {
            type: "object",
            properties: {
                action: {
                    type: "string",
                    enum: ["start_recording", "stop_recording", "export", "clear"],
                    description: "Action to perform: 'start_recording' (begin capturing), 'stop_recording' (stop capturing but keep frames), 'export' (generate and export GIF), 'clear' (discard frames)"
                },
                tabId: {
                    type: "number",
                    description: "Tab ID to identify which tab group this operation applies to"
                },
                download: {
                    type: "boolean",
                    description: "Always set this to true for the 'export' action only. This causes the gif to be downloaded in the browser."
                },
                filename: {
                    type: "string",
                    description: "Optional filename for exported GIF (default: 'recording-[timestamp].gif'). For 'export' action only."
                },
                options: {
                    type: "object",
                    description: "Optional GIF enhancement options for 'export' action. Properties: showClickIndicators (bool), showDragPaths (bool), showActionLabels (bool), showProgressBar (bool), showWatermark (bool), quality (number 1-30). All default to true except quality (default: 10).",
                    properties: {
                        showClickIndicators: {
                            type: "boolean",
                            description: "Show orange circles at click locations (default: true)"
                        },
                        showDragPaths: {
                            type: "boolean",
                            description: "Show red arrows for drag actions (default: true)"
                        },
                        showActionLabels: {
                            type: "boolean",
                            description: "Show black labels describing actions (default: true)"
                        },
                        showProgressBar: {
                            type: "boolean",
                            description: "Show orange progress bar at bottom (default: true)"
                        },
                        showWatermark: {
                            type: "boolean",
                            description: "Show Claude logo watermark (default: true)"
                        },
                        quality: {
                            type: "number",
                            description: "GIF compression quality, 1-30 (lower = better quality, slower encoding). Default: 10"
                        }
                    }
                }
            },
            required: ["action", "tabId"]
        }
    }, {
        name: "upload_image",
        description: "Upload a previously captured screenshot or user-uploaded image to a file input or drag & drop target. Supports two approaches: (1) ref - for targeting specific elements, especially hidden file inputs, (2) coordinate - for drag & drop to visible locations like Google Docs. Provide either ref or coordinate, not both.",
        inputSchema: {
            type: "object",
            properties: {
                imageId: {
                    type: "string",
                    description: "ID of a previously captured screenshot (from the computer tool's screenshot action) or a user-uploaded image"
                },
                ref: {
                    type: "string",
                    description: 'Element reference ID from read_page or find tools (e.g., "ref_1", "ref_2"). Use this for file inputs (especially hidden ones) or specific elements. Provide either ref or coordinate, not both.'
                },
                coordinate: {
                    type: "array",
                    items: {
                        type: "number"
                    },
                    description: "Viewport coordinates [x, y] for drag & drop to a visible location. Use this for drag & drop targets like Google Docs. Provide either ref or coordinate, not both."
                },
                tabId: {
                    type: "number",
                    description: "Tab ID where the target element is located. This is where the image will be uploaded to."
                },
                filename: {
                    type: "string",
                    description: 'Optional filename for the uploaded file (default: "image.png")'
                }
            },
            required: ["imageId", "tabId"]
        }
    }
];
