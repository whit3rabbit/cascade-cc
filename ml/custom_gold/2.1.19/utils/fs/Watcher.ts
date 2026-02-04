import { watch, FSWatcher } from 'chokidar';
import { EventEmitter } from 'events';

export class Watcher extends EventEmitter {
    private watcher: FSWatcher | null = null;
    private debounceTimers: Map<string, NodeJS.Timeout> = new Map();

    constructor() {
        super();
    }

    watch(path: string) {
        if (this.watcher) {
            this.watcher.add(path);
        } else {
            this.watcher = watch(path, {
                persistent: true,
                ignoreInitial: true,
                awaitWriteFinish: {
                    stabilityThreshold: 100,
                    pollInterval: 100
                }
            });

            this.watcher.on('add', (path: string) => this.emitChange('add', path));
            this.watcher.on('change', (path: string) => this.emitChange('change', path));
            this.watcher.on('unlink', (path: string) => this.emitChange('unlink', path));
            this.watcher.on('error', (error: any) => this.emit('error', error));
        }
    }

    unwatch(path: string) {
        if (this.watcher) {
            this.watcher.unwatch(path);
        }
    }

    close() {
        if (this.watcher) {
            this.watcher.close();
            this.watcher = null;
        }
        this.removeAllListeners();
    }

    private emitChange(type: string, path: string) {
        // Debounce if needed, but chokidar usually handles this well with awaitWriteFinish
        this.emit('change', type, path);
        this.emit('all', type, path);
    }
}
