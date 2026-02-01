import { Telemetry } from './Telemetry.js';

// From chunk810
export const SPINNER_VERBS = [
    "Accomplishing", "Actioning", "Actualizing", "Architecting", "Baking", "Beaming", "Beboppin'",
    "Befuddling", "Billowing", "Blanching", "Bloviating", "Boogieing", "Boondoggling", "Booping",
    "Bootstrapping", "Brewing", "Burrowing", "Calculating", "Canoodling", "Caramelizing", "Cascading",
    "Catapulting", "Cerebrating", "Channeling", "Channelling", "Choreographing", "Churning", "Clauding",
    "Coalescing", "Cogitating", "Combobulating", "Composing", "Computing", "Concocting", "Considering",
    "Contemplating", "Cooking", "Crafting", "Creating", "Crunching", "Crystallizing", "Cultivating",
    "Deciphering", "Deliberating", "Determining", "Dilly-dallying", "Discombobulating", "Doing",
    "Doodling", "Drizzling", "Ebbing", "Effecting", "Elucidating", "Embellishing", "Enchanting",
    "Envisioning", "Evaporating", "Fermenting", "Fiddle-faddling", "Finagling", "Flambéing",
    "Flibbertigibbeting", "Flowing", "Flummoxing", "Fluttering", "Forging", "Forming", "Frolicking",
    "Frosting", "Gallivanting", "Galloping", "Garnishing", "Generating", "Germinating", "Gitifying",
    "Grooving", "Gusting", "Harmonizing", "Hashing", "Hatching", "Herding", "Honking", "Hullaballooing",
    "Hyperspacing", "Ideating", "Imagining", "Improvising", "Incubating", "Inferring", "Infusing",
    "Ionizing", "Jitterbugging", "Julienning", "Kneading", "Leavening", "Levitating", "Lollygagging",
    "Manifesting", "Marinating", "Meandering", "Metamorphosing", "Misting", "Moonwalking", "Moseying",
    "Mulling", "Mustering", "Musing", "Nebulizing", "Nesting", "Noodling", "Nucleating", "Orbiting",
    "Orchestrating", "Osmosing", "Perambulating", "Percolating", "Perusing", "Philosophising",
    "Photosynthesizing", "Pollinating", "Pondering", "Pontificating", "Pouncing", "Precipitating",
    "Prestidigitating", "Processing", "Proofing", "Propagating", "Puttering", "Puzzling",
    "Quantumizing", "Razzle-dazzling", "Razzmatazzing", "Recombobulating", "Reticulating", "Roosting",
    "Ruminating", "Sautéing", "Scampering", "Schlepping", "Scurrying", "Seasoning", "Shenaniganing",
    "Shimmying", "Simmering", "Skedaddling", "Sketching", "Slithering", "Smooshing", "Sock-hopping",
    "Spelunking", "Spinning", "Sprouting", "Stewing", "Sublimating", "Swirling", "Swooping", "Symbioting",
    "Synthesizing", "Tempering", "Thinking", "Thundering", "Tinkering", "Tomfoolering", "Topsy-turvying",
    "Transfiguring", "Transmuting", "Twisting", "Undulating", "Unfurling", "Unravelling", "Vibing",
    "Waddling", "Wandering", "Warping", "Whatchamacalliting", "Whirlpooling", "Whirring", "Whisking",
    "Wibbling", "Working", "Wrangling", "Zesting", "Zigzagging"
];

export class CliActivityTracker {
    private activeOperations = new Set<string>();
    private lastUserActivityTime = 0;
    private lastCLIRecordedTime = Date.now();
    private isCLIActive = false;
    private readonly USER_ACTIVITY_TIMEOUT_MS = 5000;

    private static instance: CliActivityTracker | null = null;

    public static getInstance(): CliActivityTracker {
        if (!CliActivityTracker.instance) {
            CliActivityTracker.instance = new CliActivityTracker();
        }
        return CliActivityTracker.instance;
    }

    public recordUserActivity(): void {
        if (!this.isCLIActive && this.lastUserActivityTime !== 0) {
            const timeDiff = (Date.now() - this.lastUserActivityTime) / 1000;
            if (timeDiff > 0) {
                const telemetry = Telemetry.getInstance(); // Assuming singleton accessor exists or similar
                // Note: The original chunk calls Mk1() which likely returns a metrics collector or similar.
                // We'll assume Telemetry has something similar or just log it for now if method doesn't exist
                // logic from chunk: q.add(K, { type: "user" });

                // Placeholder for actual telemetry call if needed
                // telemetry.recordWaitTime(timeDiff, "user");
            }
        }
        this.lastUserActivityTime = Date.now();
    }

    public startCLIActivity(operationId: string): void {
        if (this.activeOperations.has(operationId)) {
            this.endCLIActivity(operationId);
        }
        const wasEmpty = this.activeOperations.size === 0;
        this.activeOperations.add(operationId);
        if (wasEmpty) {
            this.isCLIActive = true;
            this.lastCLIRecordedTime = Date.now();
        }
    }

    public endCLIActivity(operationId: string): void {
        this.activeOperations.delete(operationId);
        if (this.activeOperations.size === 0) {
            const now = Date.now();
            const timeDiff = (now - this.lastCLIRecordedTime) / 1000;
            if (timeDiff > 0) {
                // const telemetry = Telemetry.getInstance();
                // telemetry.recordWaitTime(timeDiff, "cli");
            }
            this.lastCLIRecordedTime = now;
            this.isCLIActive = false;
        }
    }

    public async trackOperation<T>(operationId: string, operation: () => Promise<T>): Promise<T> {
        this.startCLIActivity(operationId);
        try {
            return await operation();
        } finally {
            this.endCLIActivity(operationId);
        }
    }

    public getActivityStates() {
        return {
            isUserActive: (Date.now() - this.lastUserActivityTime) / 1000 < this.USER_ACTIVITY_TIMEOUT_MS / 1000,
            isCLIActive: this.isCLIActive,
            activeOperationCount: this.activeOperations.size
        };
    }

    public getRandomSpinnerVerb(): string {
        return SPINNER_VERBS[Math.floor(Math.random() * SPINNER_VERBS.length)];
    }
}
