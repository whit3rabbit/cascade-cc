
import { AsyncQueue } from "./AsyncQueue.js";

/**
 * Runs multiple async generators in parallel.
 * Logic based on chunk_455.ts (xHA).
 */
export async function* runParallel<T>(
    generators: AsyncGenerator<T>[],
    concurrency: number = Infinity
): AsyncGenerator<T> {
    const queue = new AsyncQueue<T>();
    let activeCount = 0;
    let finishedCount = 0;

    const pump = async (gen: AsyncGenerator<T>) => {
        activeCount++;
        try {
            for await (const value of gen) {
                queue.enqueue(value);
            }
        } catch (err) {
            queue.error(err instanceof Error ? err : new Error(String(err)));
        } finally {
            activeCount--;
            finishedCount++;
            if (finishedCount === generators.length) {
                queue.done();
            }
        }
    };

    for (const gen of generators) {
        // Concurrency limit logic could be added here if needed, 
        // but for now we just start them all if concurrency is high.
        pump(gen);
    }

    if (generators.length === 0) {
        queue.done();
    }

    yield* queue;
}
