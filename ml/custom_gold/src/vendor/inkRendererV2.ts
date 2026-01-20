import { ScreenBuffer, renderNodeToScreen, diffScreens } from "./inkScreenRenderer.js";
import { InkNode } from "./inkDomLayer.js";

/**
 * Buffer for rendering operations to be executed on a screen.
 * Deobfuscated from dNA in chunk_200.ts.
 */
export class RenderCommandList {
    private operations: any[] = [];

    constructor(
        public width: number,
        public height: number,
        private ink2: boolean = false
    ) { }

    blit(src: ScreenBuffer, region: any): void {
        this.operations.push({ type: "blit", src, region });
    }

    clear(region: any): void {
        this.operations.push({ type: "clear", region });
    }

    write(x: number, y: number, text: string): void {
        if (!text) return;
        this.operations.push({ type: "write", x, y, text });
    }

    clip(clip: any): void {
        this.operations.push({ type: "clip", clip });
    }

    unclip(): void {
        this.operations.push({ type: "unclip" });
    }

    /**
     * Executes the command list against a fresh screen buffer.
     */
    execute(stylePool: any): ScreenBuffer {
        const buffer = new ScreenBuffer(this.width, this.height);
        // In a full implementation, this would process operations one by one,
        // applying clipping regions and blitting source buffers.
        return buffer;
    }
}

/**
 * Tracks and minimizes terminal cursor movements.
 * Deobfuscated from Cl1 in chunk_200.ts.
 */
export class CursorTracker {
    public x = 0;
    public y = 0;
    public diff: any[] = [];

    constructor(initial: { x: number, y: number }, public viewportWidth: number) { }

    moveTo(targetX: number, targetY: number): void {
        const dx = targetX - this.x;
        const dy = targetY - this.y;
        if (dx === 0 && dy === 0) return;

        this.diff.push({ type: "cursorMove", x: targetX, y: targetY });
        this.x = targetX;
        this.y = targetY;
    }

    txn(action: (current: { x: number, y: number }) => [any[], { dx: number, dy: number }]): void {
        const [ops, delta] = action({ x: this.x, y: this.y });
        this.diff.push(...ops);
        this.x += delta.dx;
        this.y += delta.dy;
    }
}

/**
 * Class for terminal diffing and ANSI generation.
 * Deobfuscated from zl1 in chunk_200.ts.
 */
export class TerminalRenderer {
    private state = {
        fullStaticOutput: "",
        previousOutput: ""
    };

    constructor(private options: any) { }

    render(oldState: any, newState: any): any[] {
        if (!this.options.isTTY) {
            return [{ type: "stdout", content: newState.output }];
        }

        const ops: any[] = [];
        if (newState.staticOutput && newState.staticOutput !== "\n") {
            this.state.fullStaticOutput += newState.staticOutput;
            ops.push({ type: "stdout", content: newState.staticOutput });
        }

        if (newState.output !== oldState.output) {
            // Simplistic diff: clear N lines and redraw
            const oldLines = oldState.output.split("\n").length;
            ops.push({ type: "clear", count: oldLines });
            ops.push({ type: "stdout", content: newState.output });
            this.state.previousOutput = newState.output;
        }

        return ops;
    }
}

/**
 * Recursively finds progress components in a node tree.
 * Deobfuscated from lIB in chunk_200.ts.
 */
export function findProgressNode(node: InkNode): any {
    if (node.nodeName === "ink-progress") {
        return {
            state: node.attributes.state,
            percentage: node.attributes.percentage
        };
    }
    for (const child of node.childNodes) {
        const progress = findProgressNode(child);
        if (progress) return progress;
    }
    return undefined;
}

/**
 * Creates a renderer function for a component tree.
 * Deobfuscated from Hl1 in chunk_200.ts.
 */
export function createRenderer(root: InkNode, stylePool: any) {
    return (options: any) => {
        const { terminalWidth, terminalRows, isTTY, ink2 } = options;

        const commandList = new RenderCommandList(terminalWidth, terminalRows, ink2);
        // renderNodeToScreen(root, someBuffer, ...) - simplified

        return {
            output: "Rendering result...",
            outputHeight: 1,
            staticOutput: "",
            rows: terminalRows,
            columns: terminalWidth,
            cursorVisible: true,
            screen: commandList.execute(stylePool),
            viewport: { width: terminalWidth, height: terminalRows },
            cursor: { x: 0, y: 0, visible: true },
            progress: findProgressNode(root)
        };
    };
}
