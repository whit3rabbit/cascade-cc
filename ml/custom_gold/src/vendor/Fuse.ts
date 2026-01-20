
// Based on chunk_494.ts logic (Fuzzy search)

export interface SearchOptions {
    location?: number;
    distance?: number;
    threshold?: number;
    findAllMatches?: boolean;
    minMatchCharLength?: number;
    includeMatches?: boolean;
    ignoreLocation?: boolean;
    keys?: (string | { name: string; weight: number })[];
}

export interface SearchResult<T> {
    item: T;
    score?: number;
    refIndex: number;
}

export class Fuse<T> {
    private collection: T[];
    private options: SearchOptions;

    constructor(collection: T[], options: SearchOptions = {}) {
        this.collection = collection;
        this.options = {
            location: 0,
            distance: 100,
            threshold: 0.6,
            findAllMatches: false,
            minMatchCharLength: 1,
            includeMatches: false,
            ignoreLocation: false,
            ...options
        };
    }

    search(pattern: string): SearchResult<T>[] {
        if (!pattern) return [];

        const results: SearchResult<T>[] = [];
        const lowerPattern = pattern.toLowerCase();

        this.collection.forEach((item, index) => {
            let bestScore = 1;
            let found = false;

            const keys = this.options.keys || [];
            for (const key of keys) {
                const keyName = typeof key === 'string' ? key : key.name;
                const weight = typeof key === 'string' ? 1 : key.weight;

                const value = (item as any)[keyName];
                if (typeof value === 'string') {
                    const lowerValue = value.toLowerCase();
                    if (lowerValue.includes(lowerPattern)) {
                        const score = (1 - (lowerPattern.length / lowerValue.length)) / weight;
                        if (score < bestScore) bestScore = score;
                        found = true;
                    }
                } else if (Array.isArray(value)) {
                    for (const v of value) {
                        if (typeof v === 'string' && v.toLowerCase().includes(lowerPattern)) {
                            const score = (1 - (lowerPattern.length / v.length)) / weight;
                            if (score < bestScore) bestScore = score;
                            found = true;
                        }
                    }
                }
            }

            if (found) {
                results.push({
                    item,
                    score: bestScore,
                    refIndex: index
                });
            }
        });

        return results.sort((a, b) => (a.score || 0) - (b.score || 0));
    }
}
