import {
    Aggregator,
    DropAggregator,
    SumAggregator,
    LastValueAggregator,
    HistogramAggregator,
    ExponentialHistogramAggregator,
    InstrumentType,
    AggregationTemporality,
    MetricData,
    Accumulation
} from "./MetricAggregators.js";
import { callWithTimeout, TimeoutError, FlatMap, hashAttributes, isNotNullish, PromiseAllSettled, isPromiseAllSettledRejectionResult } from "./MetricUtils.js";
import { log } from "../logger/loggerService.js";

// --- Aggregation Factory & Types ---

export enum AggregationType {
    DEFAULT = 0,
    DROP = 1,
    SUM = 2,
    LAST_VALUE = 3,
    EXPLICIT_BUCKET_HISTOGRAM = 4,
    EXPONENTIAL_HISTOGRAM = 5
}

export interface Aggregation {
    createAggregator(instrument: InstrumentDescriptor): Aggregator<any>;
}

export class DropAggregation implements Aggregation {
    static DEFAULT_INSTANCE = new DropAggregator();
    createAggregator(_: InstrumentDescriptor) {
        return DropAggregation.DEFAULT_INSTANCE;
    }
}

export class SumAggregation implements Aggregation {
    static MONOTONIC_INSTANCE = new SumAggregator(true);
    static NON_MONOTONIC_INSTANCE = new SumAggregator(false);
    createAggregator(instrument: InstrumentDescriptor) {
        switch (instrument.type) {
            case InstrumentType.COUNTER:
            case InstrumentType.OBSERVABLE_COUNTER:
            case InstrumentType.HISTOGRAM:
                return SumAggregation.MONOTONIC_INSTANCE;
            default:
                return SumAggregation.NON_MONOTONIC_INSTANCE;
        }
    }
}

export class LastValueAggregation implements Aggregation {
    static DEFAULT_INSTANCE = new LastValueAggregator();
    createAggregator(_: InstrumentDescriptor) {
        return LastValueAggregation.DEFAULT_INSTANCE;
    }
}

export class HistogramAggregation implements Aggregation {
    static DEFAULT_INSTANCE = new HistogramAggregator([0, 5, 10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000], true);
    createAggregator(_: InstrumentDescriptor) {
        return HistogramAggregation.DEFAULT_INSTANCE;
    }
}

export class ExplicitBucketHistogramAggregation implements Aggregation {
    private _boundaries: number[];
    private _recordMinMax: boolean;

    constructor(boundaries?: number[], recordMinMax = true) {
        if (!boundaries) throw new Error("ExplicitBucketHistogramAggregation should be created with explicit boundaries");
        this._boundaries = [...boundaries].sort((a, b) => a - b);
        this._recordMinMax = recordMinMax;
    }
    createAggregator(_: InstrumentDescriptor) {
        return new HistogramAggregator(this._boundaries, this._recordMinMax);
    }
}

export class ExponentialHistogramAggregation implements Aggregation {
    constructor(private maxSize = 160, private recordMinMax = true) { }
    createAggregator(_: InstrumentDescriptor) {
        return new ExponentialHistogramAggregator(this.maxSize, this.recordMinMax);
    }
}

export class DefaultAggregation implements Aggregation {
    private _resolve(instrument: InstrumentDescriptor): Aggregation {
        switch (instrument.type) {
            case InstrumentType.COUNTER:
            case InstrumentType.UP_DOWN_COUNTER:
            case InstrumentType.OBSERVABLE_COUNTER:
            case InstrumentType.OBSERVABLE_UP_DOWN_COUNTER:
                return new SumAggregation();
            case InstrumentType.GAUGE:
            case InstrumentType.OBSERVABLE_GAUGE:
                return new LastValueAggregation();
            case InstrumentType.HISTOGRAM:
                if (instrument.advice?.explicitBucketBoundaries) {
                    return new ExplicitBucketHistogramAggregation(instrument.advice.explicitBucketBoundaries);
                }
                return new HistogramAggregation();
        }
        return new DropAggregation();
    }
    createAggregator(instrument: InstrumentDescriptor) {
        return this._resolve(instrument).createAggregator(instrument);
    }
}

export const DEFAULT_AGGREGATION_SELECTOR = (instrument: InstrumentDescriptor) => ({ type: AggregationType.DEFAULT });
export const DEFAULT_AGGREGATION_TEMPORALITY_SELECTOR = (instrument: InstrumentDescriptor) => AggregationTemporality.CUMULATIVE;

// --- Instruments & Descriptors ---

export enum ValueType {
    INT = 0,
    DOUBLE = 1
}

export interface InstrumentDescriptor {
    name: string;
    description: string;
    type: InstrumentType;
    unit: string;
    valueType: ValueType;
    advice?: any;
}

export function createInstrumentDescriptor(name: string, type: InstrumentType, options?: any): InstrumentDescriptor {
    return {
        name,
        type,
        description: options?.description ?? "",
        unit: options?.unit ?? "",
        valueType: options?.valueType ?? ValueType.DOUBLE,
        advice: options?.advice ?? {}
    };
}

// --- Attribute Helpers ---

export class AttributeHashMap<T> {
    private _map = new Map<string, T>();

    get(attributes: any) {
        return this._map.get(hashAttributes(attributes));
    }

    getOrDefault(attributes: any, factory: () => T): T {
        const hash = hashAttributes(attributes);
        if (this._map.has(hash)) return this._map.get(hash)!;
        const val = factory();
        this._map.set(hash, val);
        return val;
    }

    set(attributes: any, value: T) {
        this._map.set(hashAttributes(attributes), value);
    }

    has(attributes: any) {
        return this._map.has(hashAttributes(attributes));
    }

    entries(): [Record<string, any>, T][] {
        return [];
    }
}

// --- Predicates & Processors ---

export class PatternPredicate {
    constructor(private readonly _pattern: string) { }
    match(str: string): boolean {
        // Simple wildcard match: * represents anything
        // Convert glob pattern to regex
        const regex = new RegExp(`^${this._pattern.split('*').map(s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*')}$`);
        return regex.test(str);
    }
}

export class ExactPredicate {
    constructor(private readonly _pattern: string) { }
    match(str: string): boolean {
        return this._pattern === str;
    }
}

export class AttributesProcessor {
    // Abstract base or interface in reality
    process(attributes: any) {
        return attributes;
    }
}

// --- Metric Storage ---

export abstract class MetricStorage {
    constructor(protected _instrumentDescriptor: InstrumentDescriptor) { }
    getInstrumentDescriptor() { return this._instrumentDescriptor; }
    abstract record(value: number, attributes: any, context: any, time: number): void;
    abstract collect(collector: any, collectionTime: number): MetricData | null;
}

export class SyncMetricStorage extends MetricStorage {
    private _aggregator: Aggregator<any>;
    private _activeAccumulations = new Map<string, { attributes: any, accumulation: Accumulation }>();
    private _attributesProcessor: AttributesProcessor;

    constructor(descriptor: InstrumentDescriptor, aggregator: Aggregator<any>, attributesProcessor: any) {
        super(descriptor);
        this._aggregator = aggregator;
        this._attributesProcessor = attributesProcessor || new AttributesProcessor();
    }

    record(value: number, attributes: any, context: any, time: number) {
        attributes = this._attributesProcessor.process(attributes);
        const hash = hashAttributes(attributes);
        let current = this._activeAccumulations.get(hash);
        if (!current) {
            current = {
                attributes,
                accumulation: this._aggregator.createAccumulation(time)
            };
            this._activeAccumulations.set(hash, current);
        }
        (current.accumulation as any).record(value);
    }

    collect(collector: any, collectionTime: number): MetricData | null {
        const dataPoints = Array.from(this._activeAccumulations.values()).map(({ attributes, accumulation }) => {
            return [attributes, accumulation] as [Record<string, any>, Accumulation];
        });

        if (dataPoints.length === 0) return null;

        return this._aggregator.toMetricData(
            this._instrumentDescriptor,
            AggregationTemporality.CUMULATIVE,
            dataPoints,
            collectionTime
        );
    }
}

// --- Meter & Provider ---

export class MetricStorageRegistry {
    private _sharedRegistry = new Map<string, MetricStorage[]>();

    register(storage: MetricStorage) {
        const name = storage.getInstrumentDescriptor().name;
        if (!this._sharedRegistry.has(name)) {
            this._sharedRegistry.set(name, []);
        }
        this._sharedRegistry.get(name)!.push(storage);
    }

    getStorages(): MetricStorage[] {
        return FlatMap(Array.from(this._sharedRegistry.values()), (s) => s);
    }

    findOrUpdateCompatibleStorage(descriptor: InstrumentDescriptor): MetricStorage | null {
        const storages = this._sharedRegistry.get(descriptor.name);
        if (!storages) return null;
        return storages[0] || null;
    }
}

export class MeterSharedState {
    metricStorageRegistry = new MetricStorageRegistry();

    constructor(private _meterProviderSharedState: any, private _instrumentationScope: any) { }

    registerMetricStorage(descriptor: InstrumentDescriptor): MetricStorage {
        const existing = this.metricStorageRegistry.findOrUpdateCompatibleStorage(descriptor);
        if (existing) return existing;

        const aggregator = new DefaultAggregation().createAggregator(descriptor);
        const storage = new SyncMetricStorage(descriptor, aggregator, {});
        this.metricStorageRegistry.register(storage);
        return storage;
    }

    async collect(collector: any, collectionTime: number) {
        const storages = this.metricStorageRegistry.getStorages();
        const metrics = storages.map(s => s.collect(collector, collectionTime)).filter(isNotNullish);

        return {
            scopeMetrics: {
                scope: this._instrumentationScope,
                metrics
            },
            errors: []
        };
    }
}

export class MeterProviderSharedState {
    meterSharedStates = new Map<string, MeterSharedState>();
    viewRegistry = new ViewRegistry();
    metricCollectors: MetricCollector[] = [];

    constructor(public resource: any) { }

    getMeterSharedState(scope: any) {
        const key = `${scope.name}:${scope.version}`;
        if (!this.meterSharedStates.has(key)) {
            this.meterSharedStates.set(key, new MeterSharedState(this, scope));
        }
        return this.meterSharedStates.get(key)!;
    }
}

export class MeterProvider {
    private _sharedState: MeterProviderSharedState;
    private _shutdown = false;

    constructor(options: any = {}) {
        this._sharedState = new MeterProviderSharedState(options.resource ?? {});
        if (options.readers && options.readers.length > 0) {
            for (const reader of options.readers) {
                const collector = new MetricCollector(this._sharedState, reader);
                reader.setMetricProducer(collector);
                this._sharedState.metricCollectors.push(collector);
            }
        }
    }

    getMeter(name: string, version = "", options: any = {}) {
        if (this._shutdown) return new Meter({} as any); // noop
        return new Meter(this._sharedState.getMeterSharedState({ name, version, schemaUrl: options.schemaUrl }));
    }

    async shutdown() {
        this._shutdown = true;
        await Promise.all(this._sharedState.metricCollectors.map(c => c.shutdown()));
    }

    async forceFlush() {
        await Promise.all(this._sharedState.metricCollectors.map(c => c.forceFlush()));
    }
}

export class Meter {
    constructor(private _meterSharedState: MeterSharedState) { }

    createCounter(name: string, options?: any) {
        const descriptor = createInstrumentDescriptor(name, InstrumentType.COUNTER, options);
        const storage = this._meterSharedState.registerMetricStorage(descriptor);
        return new Counter(storage, descriptor);
    }

    createHistogram(name: string, options?: any) {
        const descriptor = createInstrumentDescriptor(name, InstrumentType.HISTOGRAM, options);
        const storage = this._meterSharedState.registerMetricStorage(descriptor);
        return new Histogram(storage, descriptor);
    }

    createGauge(name: string, options?: any) {
        const descriptor = createInstrumentDescriptor(name, InstrumentType.GAUGE, options);
        const storage = this._meterSharedState.registerMetricStorage(descriptor);
        return new Gauge(storage, descriptor);
    }

    createUpDownCounter(name: string, options?: any) {
        const descriptor = createInstrumentDescriptor(name, InstrumentType.UP_DOWN_COUNTER, options);
        const storage = this._meterSharedState.registerMetricStorage(descriptor);
        return new UpDownCounter(storage, descriptor);
    }
}

// --- Sync Instruments ---

export class SyncInstrument {
    constructor(private _storage: MetricStorage, private _descriptor: InstrumentDescriptor) { }
    protected _record(value: number, attributes: any = {}, context: any = {}) {
        if (typeof value !== "number") return;
        this._storage.record(value, attributes, context, Date.now());
    }
}

export class Counter extends SyncInstrument {
    add(value: number, attributes?: any, context?: any) {
        if (value < 0) return;
        this._record(value, attributes, context);
    }
}

export class UpDownCounter extends SyncInstrument {
    add(value: number, attributes?: any, context?: any) {
        this._record(value, attributes, context);
    }
}

export class Histogram extends SyncInstrument {
    record(value: number, attributes?: any, context?: any) {
        this._record(value, attributes, context);
    }
}

export class Gauge extends SyncInstrument {
    record(value: number, attributes?: any, context?: any) {
        this._record(value, attributes, context);
    }
}


// --- Collectors & Exporters ---

export interface MetricProducer {
    collect(options?: { timeoutMillis?: number }): Promise<any>;
}

export class MetricCollector implements MetricProducer {
    constructor(private _sharedState: MeterProviderSharedState, private _metricReader: MetricReader) { }

    async collect(options?: { timeoutMillis?: number }) {
        const collectionTime = Date.now();
        const resourceMetrics = {
            resource: this._sharedState.resource,
            scopeMetrics: [] as any[]
        };
        const errors: any[] = [];

        await Promise.all(Array.from(this._sharedState.meterSharedStates.values()).map(async (meterState) => {
            const result = await meterState.collect(this, collectionTime);
            if (result.scopeMetrics.metrics.length > 0) {
                resourceMetrics.scopeMetrics.push(result.scopeMetrics);
            }
        }));

        return { resourceMetrics, errors };
    }

    async shutdown() { await this._metricReader.shutdown(); }
    async forceFlush() { await this._metricReader.forceFlush(); }
}

export class MetricReader {
    private _shutdown = false;
    private _metricProducer?: MetricProducer;

    constructor(options: any = {}) { }

    setMetricProducer(producer: MetricProducer) {
        this._metricProducer = producer;
        this.onInitialized();
    }

    async collect(options?: { timeoutMillis?: number }) {
        if (this._shutdown) throw new Error("Shutdown");
        if (!this._metricProducer) throw new Error("No producer");
        return this._metricProducer.collect(options);
    }

    async shutdown(options?: { timeoutMillis?: number }) {
        if (this._shutdown) return;
        this._shutdown = true;
        await this.onShutdown();
    }

    async forceFlush(options?: { timeoutMillis?: number }) {
        await this.onForceFlush();
    }

    protected async onShutdown() { }
    protected async onForceFlush() { }
    protected onInitialized() { }
}

export interface MetricExporter {
    export(metrics: any, resultCallback: (result: any) => void): void;
    forceFlush(): Promise<void>;
    shutdown(): Promise<void>;
}

export class PeriodicExportingMetricReader extends MetricReader {
    private _interval: NodeJS.Timeout | null = null;
    private _exporter: MetricExporter;
    private _exportInterval: number;
    private _exportTimeout: number;

    constructor(options: { exporter: MetricExporter, exportIntervalMillis?: number, exportTimeoutMillis?: number }) {
        super();
        this._exporter = options.exporter;
        this._exportInterval = options.exportIntervalMillis ?? 60000;
        this._exportTimeout = options.exportTimeoutMillis ?? 30000;
    }

    protected onInitialized() {
        this._interval = setInterval(() => this._runOnce(), this._exportInterval);
        if (this._interval.unref) this._interval.unref();
    }

    private async _runOnce() {
        try {
            await callWithTimeout(this._doRun(), this._exportTimeout);
        } catch (err) { }
    }

    private async _doRun() {
        const result = await this.collect({ timeoutMillis: this._exportTimeout });
        await new Promise<void>((resolve) => {
            this._exporter.export(result, () => resolve());
        });
    }

    async onShutdown() {
        if (this._interval) clearInterval(this._interval);
        await this._exporter.shutdown();
    }
}

export class ConsoleMetricExporter implements MetricExporter {
    export(metrics: any, callback: (result: any) => void) {
        callback({ code: 0 });
    }
    async forceFlush() { }
    async shutdown() { }
}

export class InMemoryMetricExporter implements MetricExporter {
    private _metrics: any[] = [];
    export(metrics: any, callback: (result: any) => void) {
        this._metrics.push(metrics);
        callback({ code: 0 });
    }
    getMetrics() { return this._metrics; }
    reset() { this._metrics = []; }
    async forceFlush() { }
    async shutdown() { }
}

export class ViewRegistry {
    private _registeredViews: any[] = [];
    addView(view: any) { this._registeredViews.push(view); }
    findViews(instrument: any, meter: any) { return this._registeredViews; } // check matching logic
}

export class View {
    constructor(public options: any) { }
}

export class ObservableRegistry {
    observe(flow: any, timeout: number | undefined) {
        return Promise.resolve([]);
    }
}
