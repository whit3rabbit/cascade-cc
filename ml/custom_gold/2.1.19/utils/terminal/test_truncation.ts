import { truncateString, getWidth } from './ansi.js';
import { truncateContent } from './contentTruncation.js';

function assert(condition: boolean, message: string) {
    if (!condition) {
        throw new Error(`Assertion failed: ${message}`);
    }
}

console.log('Running truncation tests...');

// Test getWidth
assert(getWidth('hello') === 5, 'getWidth simple');
assert(getWidth('\u001b[31mhello\u001b[39m') === 5, 'getWidth with ANSI');
assert(getWidth('こんにちは') === 10, 'getWidth with full-width characters');

// Test truncateString
const longAnsi = '\u001b[31mThis is a very long string with ANSI codes\u001b[39m';
const truncated = truncateString(longAnsi, 10);
assert(getWidth(truncated) === 10, 'truncateString width');
assert(truncated.includes('…'), 'truncateString has ellipsis');

// Test truncateContent (QP1)
const longContent = 'line1\nline2\nline3\nline4\nline5';
const res1 = truncateContent(longContent, 10);
assert(res1.truncatedContent.includes('lines truncated'), 'truncateContent message');
assert(res1.totalLines === 5, 'truncateContent total lines');

console.log('All tests passed!');
