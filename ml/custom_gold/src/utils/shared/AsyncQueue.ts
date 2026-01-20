
export class AsyncQueue<T> {
    private queue: (T | Error)[] = [];
    private resolvers: ((value: IteratorResult<T>) => void)[] = [];
    private isDone = false;

    enqueue(item: T) {
        if (this.isDone) return;
        if (this.resolvers.length > 0) {
            const resolve = this.resolvers.shift();
            resolve?.({ value: item, done: false });
        } else {
            this.queue.push(item);
        }
    }

    error(err: Error) {
        if (this.isDone) return;
        if (this.resolvers.length > 0) {
            const resolve = this.resolvers.shift();
            resolve?.({ value: err as any, done: false }); // Technically iterator result shouldn't be error but we handle it
        } else {
            this.queue.push(err);
        }
    }

    done() {
        this.isDone = true;
        while (this.resolvers.length > 0) {
            const resolve = this.resolvers.shift();
            resolve?.({ value: undefined as any, done: true });
        }
    }

    async *[Symbol.asyncIterator](): AsyncGenerator<T> {
        while (true) {
            if (this.queue.length > 0) {
                const item = this.queue.shift();
                if (item instanceof Error) throw item;
                yield item as T;
                continue;
            }
            if (this.isDone) break;

            const next = await new Promise<IteratorResult<T>>((resolve) => {
                this.resolvers.push(resolve);
            });
            if (next.done) break;
            if (next.value instanceof Error) throw next.value;
            yield next.value;
        }
    }
}
