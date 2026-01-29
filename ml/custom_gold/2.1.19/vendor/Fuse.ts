/**
 * File: src/vendor/Fuse.ts
 * Role: Minimal stub for Fuse.js fuzzy search.
 */

export interface FuseOptions {
    threshold?: number;
    keys?: (string | { name: string; weight: number })[];
}

export interface SearchResult<T> {
    item: T;
    score?: number;
    matches?: any[];
}

export class Fuse<T> {
    private list: T[];
    private options: FuseOptions;

    constructor(list: T[], options: FuseOptions = {}) {
        this.list = list;
        this.options = options;
    }

    /**
     * Simple fuzzy search implementation.
     * 
     * @param {string} pattern - The search pattern.
     * @returns {SearchResult<T>[]} The search results.
     */
    search(pattern: string): SearchResult<T>[] {
        if (!pattern) {
            return this.list.map(item => ({ item }));
        }

        const lowerPattern = pattern.toLowerCase();

        // Very basic search logic based on keys or item properties
        return this.list
            .filter(item => {
                const searchStrings: string[] = [];

                if (this.options.keys) {
                    for (const key of this.options.keys) {
                        const keyName = typeof key === 'string' ? key : key.name;
                        const value = (item as any)[keyName];
                        if (typeof value === 'string') {
                            searchStrings.push(value.toLowerCase());
                        } else if (Array.isArray(value)) {
                            value.forEach(v => {
                                if (typeof v === 'string') searchStrings.push(v.toLowerCase());
                            });
                        }
                    }
                } else {
                    // Fallback to searching all string properties
                    for (const key in item) {
                        if (typeof (item as any)[key] === 'string') {
                            searchStrings.push((item as any)[key].toLowerCase());
                        }
                    }
                }

                return searchStrings.some(s => s.includes(lowerPattern));
            })
            .map(item => ({ item }));
    }
}
