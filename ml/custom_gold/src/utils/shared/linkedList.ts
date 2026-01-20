/**
 * Doubly-linked list structure for navigable menus.
 * Deobfuscated from UeA in chunk_205.ts.
 */

export interface MenuItem<T> {
    label: string;
    value: T;
    description?: string;
    previous?: MenuItem<T>;
    next?: MenuItem<T>;
    index: number;
}

export class MenuLinkedList<T> extends Map<T, MenuItem<T>> {
    public first?: MenuItem<T>;
    public last?: MenuItem<T>;

    constructor(items: Array<{ label: string; value: T; description?: string }>) {
        const entries: [T, MenuItem<T>][] = [];
        let first: MenuItem<T> | undefined;
        let last: MenuItem<T> | undefined;
        let prev: MenuItem<T> | undefined;

        items.forEach((item, index) => {
            const node: MenuItem<T> = {
                label: item.label,
                value: item.value,
                description: item.description,
                previous: prev,
                next: undefined,
                index
            };

            if (prev) prev.next = node;
            if (!first) first = node;
            last = node;
            entries.push([item.value, node]);
            prev = node;
        });

        super(entries);
        this.first = first;
        this.last = last;
    }
}
