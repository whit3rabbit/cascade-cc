/**
 * File: src/utils/shared/collections.ts
 * Role: Efficient data structures like Priority Queue and LRU Cache.
 */

/**
 * A standard Priority Queue implementation using a binary heap.
 */
export class PriorityQueue<T> {
    private heap: T[] = [];
    private comparator: (a: T, b: T) => number;

    constructor(comparator: (a: T, b: T) => number = (a: any, b: any) => a - b) {
        this.comparator = comparator;
    }

    /**
     * Adds an item to the priority queue.
     */
    push(item: T): void {
        this.heap.push(item);
        this.siftUp();
    }

    /**
     * Removes and returns the top item from the priority queue.
     */
    pop(): T | null {
        if (this.size() === 0) return null;

        const top = this.heap[0];
        const bottom = this.heap.pop();

        if (this.size() > 0 && bottom !== undefined) {
            this.heap[0] = bottom;
            this.siftDown();
        }

        return top;
    }

    /**
     * Returns the number of items in the queue.
     */
    size(): number {
        return this.heap.length;
    }

    private siftUp(): void {
        let node = this.heap.length - 1;
        while (node > 0) {
            const parent = (node - 1) >> 1;
            if (this.comparator(this.heap[node], this.heap[parent]) < 0) {
                [this.heap[node], this.heap[parent]] = [this.heap[parent], this.heap[node]];
                node = parent;
            } else {
                break;
            }
        }
    }

    private siftDown(): void {
        let node = 0;
        while (true) {
            const left = (node << 1) + 1;
            const right = (node << 1) + 2;
            let smallest = node;

            if (left < this.heap.length && this.comparator(this.heap[left], this.heap[smallest]) < 0) {
                smallest = left;
            }
            if (right < this.heap.length && this.comparator(this.heap[right], this.heap[smallest]) < 0) {
                smallest = right;
            }

            if (smallest !== node) {
                [this.heap[node], this.heap[smallest]] = [this.heap[smallest], this.heap[node]];
                node = smallest;
            } else {
                break;
            }
        }
    }
}

/**
 * A simple LRU (Least Recently Used) Cache.
 */
export class LRUCache<K, V> {
    private cache: Map<K, V>;
    private max: number;

    constructor(max = 1000) {
        this.max = max;
        this.cache = new Map<K, V>();
    }

    /**
     * Retrieves a value from the cache and marks it as recently used.
     */
    get(key: K): V | undefined {
        if (!this.cache.has(key)) return undefined;

        const val = this.cache.get(key)!;
        this.cache.delete(key);
        this.cache.set(key, val);
        return val;
    }

    /**
     * Sets a value in the cache, evicting the oldest item if the limit is reached.
     */
    set(key: K, value: V): void {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size >= this.max) {
            // The first key in the Map is the least recently used
            const firstKey = this.cache.keys().next().value;
            if (firstKey !== undefined) {
                this.cache.delete(firstKey);
            }
        }
        this.cache.set(key, value);
    }

    /**
     * Checks if a key exists in the cache without updating its recency.
     */
    has(key: K): boolean {
        return this.cache.has(key);
    }

    /**
     * Deletes a key from the cache.
     */
    delete(key: K): void {
        this.cache.delete(key);
    }

    /**
     * Clears the cache.
     */
    clear(): void {
        this.cache.clear();
    }

    /**
     * Returns the current size of the cache.
     */
    get size(): number {
        return this.cache.size;
    }
}
