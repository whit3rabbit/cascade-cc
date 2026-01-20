
import { useState, useMemo } from 'react';

/**
 * Hook for managing scrollable lists with selection.
 * Logic derived from ys in chunk_570.ts.
 */
export function useScroll({
    totalItems,
    selectedIndex,
    pageSize = 10
}: {
    totalItems: number;
    selectedIndex: number;
    pageSize?: number;
}) {
    // Calculate the start index to keep the selected item visible
    const startIndex = useMemo(() => {
        if (totalItems <= pageSize) return 0;

        let start = Math.max(0, selectedIndex - Math.floor(pageSize / 2));
        if (start + pageSize > totalItems) {
            start = Math.max(0, totalItems - pageSize);
        }
        return start;
    }, [totalItems, selectedIndex, pageSize]);

    const visibleLength = Math.min(pageSize, totalItems - startIndex);

    return {
        startIndex,
        getVisibleItems: <T,>(items: T[]) => items.slice(startIndex, startIndex + pageSize),
        toActualIndex: (index: number) => startIndex + index,
        scrollPosition: {
            current: selectedIndex + 1,
            total: totalItems,
            canScrollUp: startIndex > 0,
            canScrollDown: startIndex + pageSize < totalItems
        },
        needsPagination: totalItems > pageSize,
        handleSelectionChange: (idx: number, setIdx: (i: number) => void) => {
            setIdx(Math.max(0, Math.min(idx, totalItems - 1)));
        }
    };
}
